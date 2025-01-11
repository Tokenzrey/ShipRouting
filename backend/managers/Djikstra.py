import os
import json
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import heapq  # untuk Dijkstra priority queue

import numpy as np
import pandas as pd
import igraph as ig
from filelock import FileLock
from keras.api.models import load_model
import joblib
import tensorflow as tf
# from rtree import index  # Jika ingin menambahkan R-tree

from utils import GridLocator  # Pastikan modul ini diimplementasikan
from constants import DATA_DIR, DATA_DIR_CACHE  # Pastikan konstanta ini didefinisikan

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_tf_for_production():
    """
    Mengonfigurasi TensorFlow agar memanfaatkan GPU atau multithread jika tersedia.
    Jika tidak ada GPU, akan berjalan pada CPU dengan thread tunggal.
    """
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            logger.info(f"GPU mode aktif: {gpus[0]}")
        else:
            cpus = tf.config.list_physical_devices('CPU')
            if len(cpus) > 1:
                logger.info("Multiple CPU detected. TF might use multi-thread automatically.")
            else:
                logger.info("Single CPU mode.")
    except Exception as e:
        logger.warning(f"TensorFlow GPU/threads config error: {e}. Using default single-thread CPU.")


class EdgeBatchCache:
    """
    Kelas ini mengelola cache prediksi edge dengan menggunakan fixed_wave_data_id.
    Data cache di-load dari file: "batch_{fixed_wave_data_id}.pkl".
    Cache hanya dibaca (tidak menulis ulang) untuk menghindari komputasi ulang.
    """
    def __init__(
        self,
        cache_dir: str,
        fixed_wave_data_id: str,
        batch_size: int = 100_000,
        max_memory_cache: int = 20_000_000,
        compression_level: int = 3
    ):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.batch_size = batch_size
        self.max_memory_cache = max_memory_cache
        self.compression_level = compression_level

        # Fixed wave_data_id yang digunakan untuk operasi cache
        self.fixed_wave_data_id = fixed_wave_data_id
        self.current_wave_data_id: Optional[str] = None
        self.memory_cache: Dict[str, dict] = {}
        self._dirty = False

        self.set_current_wave_data_id(self.fixed_wave_data_id)

    def _generate_edge_key(self, edge_data: dict) -> str:
        """
        Menghasilkan key unik berdasarkan source_coords, target_coords, ship_speed, dan condition.
        """
        key_data = {
            "source": edge_data["source_coords"],
            "target": edge_data["target_coords"],
            "ship_speed": edge_data["ship_speed"],
            "condition": edge_data["condition"]
        }
        return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()

    def set_current_wave_data_id(self, _ignored_wave_data_id: str):
        """
        Abaikan parameter, gunakan self.fixed_wave_data_id.
        Muat cache dari file batch_{fixed_wave_data_id}.pkl.
        """
        forced_wid = self.fixed_wave_data_id
        if forced_wid == self.current_wave_data_id:
            return  # Tidak berubah

        if self.current_wave_data_id and self._dirty:
            logger.info("Cache sudah tercemar, melakukan flush cache sebelumnya.")
            self._flush_to_disk(self.current_wave_data_id)

        self.current_wave_data_id = forced_wid
        self.memory_cache.clear()
        self._dirty = False

        pkl_file = os.path.join(self.cache_dir, f"batch_{forced_wid}.pkl")
        if os.path.exists(pkl_file):
            lock_file = f"{pkl_file}.lock"
            try:
                with FileLock(lock_file, timeout=10):
                    loaded_data = joblib.load(pkl_file)
                if isinstance(loaded_data, dict):
                    self.memory_cache.update(loaded_data)
                    logger.info(f"Loaded {len(loaded_data)} edge predictions from {pkl_file}")
                else:
                    logger.error(f"Format data di {pkl_file} tidak sesuai. Diharapkan dict.")
            except Exception as e:
                logger.error(f"Error saat memuat file {pkl_file}: {e}")
        else:
            logger.info(f"Tidak ditemukan cache file untuk fixed_wave_data_id={forced_wid}. Memulai dengan cache kosong.")

    def _flush_to_disk(self, _wave_data_id: str):
        """
        Menyimpan memory_cache ke disk jika terdapat perubahan.
        Dalam scenario ini, operasi flush tidak terlalu penting karena kita tidak menulis ulang.
        """
        if not self.current_wave_data_id or not self._dirty:
            return
        pkl_file = os.path.join(self.cache_dir, f"batch_{self.fixed_wave_data_id}.pkl")
        lock_file = f"{pkl_file}.lock"
        try:
            with FileLock(lock_file, timeout=10):
                joblib.dump(self.memory_cache, pkl_file, compress=self.compression_level)
            logger.info(f"Flushed {len(self.memory_cache)} entries to {pkl_file}")
        except Exception as e:
            logger.error(f"Error saat flush file {pkl_file}: {e}")
        self._dirty = False

    def get_cached_predictions(self, edge_data: dict) -> Optional[dict]:
        """
        Mengambil prediksi cache berdasarkan key yang dihasilkan.
        """
        key = self._generate_edge_key(edge_data)
        return self.memory_cache.get(key)

    def _lru_cleanup(self):
        """
        Membersihkan memory_cache jika ukuran melebihi max_memory_cache.
        """
        if len(self.memory_cache) > self.max_memory_cache:
            keys = list(self.memory_cache.keys())
            for key in keys[:len(self.memory_cache) - self.max_memory_cache]:
                self.memory_cache.pop(key)

    def save_batch(self, predictions: List[dict], edge_data: List[dict]):
        """
        Metode ini tidak menyimpan data baru ke cache.
        """
        logger.debug("save_batch dipanggil, namun penyimpanan cache dinonaktifkan untuk fixed_wave_data_id.")
        pass

    def finalize(self):
        if self.current_wave_data_id and self._dirty:
            self._flush_to_disk(self.current_wave_data_id)
        logger.info("EdgeBatchCache finalize complete.")


class RouteOptimizer:
    """
    RouteOptimizer menggunakan fixed_wave_data_id untuk cache edge.
    Data prediksi (roll, heave, pitch, isBlocked) di-load dari cache, kemudian
    disinkronkan ke dalam graph (igraph) sehingga digunakan oleh metode find_shortest_path
    dan get_blocked_edges_in_view.
    """
    def __init__(
        self,
        graph_file: str,
        wave_data_locator,
        model_path: str,
        input_scaler_pkl: str,
        output_scaler_pkl: str,
        grid_locator: GridLocator
    ):
        setup_tf_for_production()

        self.graph_file = graph_file
        self.wave_data_locator = wave_data_locator  # Untuk mengambil wave data pada vertex (opsional)
        self.model_path = model_path
        self.input_scaler_pkl = input_scaler_pkl
        self.output_scaler_pkl = output_scaler_pkl
        self.grid_locator = grid_locator

        self.saved_graph_file = "region_graph.pkl"
        self.igraph_graph = self._load_or_build_graph()

        # Local Dijkstra cache
        self.cache: Dict[str, Any] = {}

        logger.info(f"Loading ML model from {self.model_path} ...")
        self.model = load_model(self.model_path, compile=False)

        logger.info(f"Loading input scaler from {self.input_scaler_pkl} ...")
        self.input_scaler = joblib.load(self.input_scaler_pkl)

        logger.info(f"Loading output scaler from {self.output_scaler_pkl} ...")
        self.output_scaler = joblib.load(self.output_scaler_pkl)

        self.dijkstra_cache_dir = os.path.join(DATA_DIR, "dijkstra")
        os.makedirs(self.dijkstra_cache_dir, exist_ok=True)

        fixed_wave_data_id = "8f3fd6520880cccae131cfbe23640734acaccb600206c264bf9349ac0dcd9bf9"
        self.edge_cache = EdgeBatchCache(
            os.path.join(DATA_DIR_CACHE, "edge_predictions"),
            fixed_wave_data_id=fixed_wave_data_id,
            batch_size=100_000,
            max_memory_cache=20_000_000,
            compression_level=3
        )

        # Pastikan atribut edge di graph
        for attr in ["roll", "heave", "pitch", "isBlocked"]:
            if attr not in self.igraph_graph.es.attributes():
                if attr == "isBlocked":
                    self.igraph_graph.es[attr] = [False] * self.igraph_graph.ecount()
                else:
                    self.igraph_graph.es[attr] = [0.0] * self.igraph_graph.ecount()

        # Update graph dengan data cache yang telah dimuat
        self._update_graph_with_cache()

        logger.info("RouteOptimizer initialized with fixed_wave_data_id. Using cached edge predictions.")

    def _update_graph_with_cache(self):
        """
        Memperbarui atribut-atribut pada graph (roll, heave, pitch, isBlocked)
        berdasarkan data yang di-load dari cache. Validasi bahwa setiap edge memiliki nilai.
        """
        wave_data_id = self.edge_cache.fixed_wave_data_id
        logger.info("Updating graph edges with cached predictions...")
        missing_count = 0
        for e in self.igraph_graph.es:
            source_coords = (self.igraph_graph.vs[e.source]["lon"], self.igraph_graph.vs[e.source]["lat"])
            target_coords = (self.igraph_graph.vs[e.target]["lon"], self.igraph_graph.vs[e.target]["lat"])
            edge_data = {
                "source_coords": source_coords,
                "target_coords": target_coords,
                "ship_speed": 8,      # Asumsi default
                "condition": 1        # Asumsi default
            }
            cached = self.edge_cache.get_cached_predictions(edge_data)
            if cached is not None:
                e["roll"] = float(cached.get("roll", 0.0))
                e["heave"] = float(cached.get("heave", 0.0))
                e["pitch"] = float(cached.get("pitch", 0.0))
                e["isBlocked"] = bool(cached.get("blocked", False))
            else:
                # Jika tidak ada, set nilai default dan hitung jumlah missing
                e["roll"] = 0.0
                e["heave"] = 0.0
                e["pitch"] = 0.0
                e["isBlocked"] = False
                missing_count += 1
        logger.info(f"Graph update complete. Missing cached predictions for {missing_count} edges.")

    def finalize(self):
        self.edge_cache.finalize()

    def update_wave_data_locator(self, wave_data_locator, ship_speed: float = 8, condition: int = 1):
        logger.warning("update_wave_data_locator ignored (using fixed_wave_data_id).")

    def _get_wave_data_identifier(self) -> str:
        return self.edge_cache.fixed_wave_data_id

    def _load_or_build_graph(self) -> ig.Graph:
        if os.path.exists(self.saved_graph_file):
            logger.info(f"Loading graph from {self.saved_graph_file}...")
            try:
                g = ig.Graph.Read_Pickle(self.saved_graph_file)
                logger.info(f"Graph loaded: {g.vcount()} vertices, {g.ecount()} edges.")
                return g
            except Exception as e:
                logger.error(f"Failed to load {self.saved_graph_file}: {e}.")
        raise FileNotFoundError(f"Graph file {self.saved_graph_file} not found or failed to load.")

    def _compute_bearing(self, start: Tuple[float, float], end: Tuple[float, float]) -> float:
        lon1, lat1 = np.radians(start)
        lon2, lat2 = np.radians(end)
        dlon = lon2 - lon1
        x = np.sin(dlon) * np.cos(lat2)
        y = (np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon))
        ib = np.degrees(np.arctan2(x, y))
        return (ib + 360) % 360

    def _compute_heading(self, bearing: float, dirpwsfc: float) -> float:
        return (bearing - dirpwsfc + 90) % 360

    def _predict_blocked(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        colnames = ["ship_speed", "wave_heading", "wave_height", "wave_period", "condition"]
        df.columns = colnames
        scaled = self.input_scaler.transform(df)
        preds = self.model.predict(scaled, verbose=0)
        unscaled = self.output_scaler.inverse_transform(preds)
        roll = unscaled[:, 0].astype(float)
        heave = unscaled[:, 1].astype(float)
        pitch = unscaled[:, 2].astype(float)
        blocked = (roll >= 6) | (heave >= 0.7) | (pitch >= 3)
        return blocked, roll, heave, pitch

    def _batch_process_edges(self, edges_data: List[dict], _ignored_wave_data_id: str) -> List[dict]:
        logger.debug("Batch processing skipped; using fixed cache values.")
        results = []
        for ed in edges_data:
            cached = self.edge_cache.get_cached_predictions(ed)
            if cached is not None:
                results.append(cached)
            else:
                logger.warning(f"No cached prediction for edge_id={ed.get('edge_id', 'unknown')}. Using default values.")
                results.append({
                    "blocked": False,
                    "roll": 0.0,
                    "heave": 0.0,
                    "pitch": 0.0
                })
        return results

    def _dijkstra_cache_key(
        self, start: Tuple[float, float], end: Tuple[float, float],
        use_model: bool, ship_speed: float, condition: int, wave_data_id: str
    ) -> str:
        data = {
            "start": list(start),
            "end": list(end),
            "use_model": use_model,
            "ship_speed": ship_speed,
            "condition": condition,
            "wave_data_id": wave_data_id
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def save_dijkstra_result(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        use_model: bool,
        ship_speed: float,
        condition: int,
        path_data: List[dict],
        distance: float,
        partial_paths: List[List[dict]]
    ):
        wave_data_id = self._get_wave_data_identifier()
        cache_file = os.path.join(self.dijkstra_cache_dir, f"{wave_data_id}.json")
        lock_file = f"{cache_file}.lock"
        category = "with_model" if use_model else "without_model"
        item = {
            "start": list(start),
            "end": list(end),
            "use_model": use_model,
            "ship_speed": ship_speed,
            "condition": condition,
            "path": path_data,
            "distance": float(distance),
            "partial_paths": partial_paths,
            "all_edges": [],
            "timestamp": f"{datetime.utcnow().isoformat()}Z",
            "fixed": True
        }

        try:
            with FileLock(lock_file, timeout=10):
                if os.path.exists(cache_file):
                    try:
                        with open(cache_file, 'r') as f:
                            cache_json = json.load(f)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in dijkstra cache; resetting {cache_file}.")
                        cache_json = {
                            "wave_data_id": wave_data_id,
                            "dijkstra_results": {"with_model": [], "without_model": []}
                        }
                else:
                    cache_json = {
                        "wave_data_id": wave_data_id,
                        "dijkstra_results": {"with_model": [], "without_model": []}
                    }
                cache_json["dijkstra_results"][category].append(item)
                with open(cache_file, 'w') as f:
                    json.dump(cache_json, f, indent=4)
                logger.info(f"Dijkstra result saved to {cache_file} for category {category}.")
        except Exception as e:
            logger.error(f"Error saving Dijkstra result to {cache_file}: {e}")

    def load_dijkstra_result(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        use_model: bool,
        ship_speed: float,
        condition: int
    ) -> Optional[Dict[str, Any]]:
        wave_data_id = self._get_wave_data_identifier()
        cache_file = os.path.join(self.dijkstra_cache_dir, f"{wave_data_id}.json")
        if not os.path.exists(cache_file):
            logger.info(f"No Dijkstra cache found for wave_data_id={wave_data_id}.")
            return None

        lock_file = f"{cache_file}.lock"
        query_str = json.dumps({
            "start": list(start),
            "end": list(end),
            "use_model": use_model,
            "ship_speed": ship_speed,
            "condition": condition,
            "wave_data_id": wave_data_id
        }, sort_keys=True)
        query_key = hashlib.sha256(query_str.encode()).hexdigest()

        try:
            with FileLock(lock_file, timeout=10):
                with open(cache_file, 'r') as f:
                    try:
                        cache_json = json.load(f)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in Dijkstra cache {cache_file}; resetting.")
                        cache_json = {
                            "wave_data_id": wave_data_id,
                            "dijkstra_results": {"with_model": [], "without_model": []}
                        }
                        with open(cache_file, 'w') as fw:
                            json.dump(cache_json, fw, indent=4)
                        return None
        except Exception as e:
            logger.error(f"Error loading Dijkstra cache from {cache_file}: {e}")
            return None

        category = "with_model" if use_model else "without_model"
        results_list = cache_json["dijkstra_results"].get(category, [])
        for item in results_list:
            if not item.get("fixed", False):
                continue
            item_str = json.dumps({
                "start": item["start"],
                "end": item["end"],
                "use_model": item["use_model"],
                "ship_speed": item["ship_speed"],
                "condition": item["condition"],
                "wave_data_id": wave_data_id
            }, sort_keys=True)
            item_key = hashlib.sha256(item_str.encode()).hexdigest()
            if item_key == query_key:
                logger.info("Found matching fixed Dijkstra result in cache; returning result.")
                return {
                    "path": item["path"],
                    "distance": float(item["distance"]),
                    "partial_paths": item.get("partial_paths", []),
                    "all_edges": []
                }
        logger.info("No matching fixed Dijkstra result found in cache.")
        return None

    def compute_block_status_for_all_edges(self, ship_speed: float, condition: int):
        """
        Dalam scenario fixed, compute_block_status_for_all_edges tidak memanggil prediksi ulang.
        """
        logger.info("Skipping compute_block_status_for_all_edges (using fixed cache).")

    def get_blocked_edges_in_view(
        self,
        view_bounds: Tuple[float, float, float, float],
        max_edges: int = 300_000,
        include_blocked_only: bool = True
    ) -> List[dict]:
        """
        Mengembalikan edge di view_bounds.
        Jika include_blocked_only True, kembalikan hanya edge yang diblokir;
        jika False, kembalikan semua edge di dalam view bounds.
        """
        min_lon, min_lat, max_lon, max_lat = view_bounds
        g = self.igraph_graph
        edges_in_view = []

        for edge in g.es:
            if include_blocked_only and not edge["isBlocked"]:
                continue
            source = g.vs[edge.source]
            target = g.vs[edge.target]
            s_lon, s_lat = source["lon"], source["lat"]
            t_lon, t_lat = target["lon"], target["lat"]
            if ((min_lon <= s_lon <= max_lon and min_lat <= s_lat <= max_lat) or 
                (min_lon <= t_lon <= max_lon and min_lat <= t_lat <= max_lat)):
                edges_in_view.append({
                    "edge_id": edge.index,
                    "source_coords": (s_lon, s_lat),
                    "target_coords": (t_lon, t_lat),
                    "isBlocked": bool(edge["isBlocked"])
                })
                if len(edges_in_view) >= max_edges:
                    break
        logger.info(f"get_blocked_edges_in_view => Returned {len(edges_in_view)} edges " +
                    f"({'blocked only' if include_blocked_only else 'all'}) in view bounds.")
        return edges_in_view

    def find_shortest_path(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        use_model: bool = False,
        ship_speed: float = 8,
        condition: int = 1
    ) -> Tuple[List[dict], float, List[List[dict]], List[dict]]:
        """
        Menjalankan Dijkstra dengan early stopping.
        Data edge (roll/heave/pitch/isBlocked) digunakan dari graph yang telah di-update dengan cache.
        Jika hasil Dijkstra sudah di-cache, langsung digunakan.
        """
        wave_data_id = self._get_wave_data_identifier()
        self.edge_cache.set_current_wave_data_id(wave_data_id)
        self.compute_block_status_for_all_edges(ship_speed=ship_speed, condition=condition)
        
        cached = self.load_dijkstra_result(start, end, use_model, ship_speed, condition)
        if cached:
            logger.info(f"Using cached Dijkstra result for start={start} and end={end}.")
            return cached["path"], cached["distance"], cached["partial_paths"], cached["all_edges"]

        gcopy = self.igraph_graph.copy()
        if use_model:
            blocked_flags = np.array([bool(e["isBlocked"]) for e in gcopy.es])
            gcopy.es["weight"] = np.where(blocked_flags, float('inf'), gcopy.es["weight"])

        start_idx = self.grid_locator.find_nearest_node(*start)
        end_idx = self.grid_locator.find_nearest_node(*end)
        if start_idx < 0 or end_idx < 0:
            logger.error(f"Invalid start or end index for start={start}, end={end}.")
            return [], 0.0, [], []

        logger.info("Starting optimized Dijkstra...")
        dist = [float('inf')] * gcopy.vcount()
        parent = [-1] * gcopy.vcount()
        dist[start_idx] = 0.0
        pq = [(0.0, start_idx)]
        partial_paths: List[List[dict]] = []
        expansions = 0

        while pq:
            current_dist, u = heapq.heappop(pq)
            expansions += 1
            if current_dist > dist[end_idx]:
                continue
            if current_dist > dist[u]:
                continue

            path_u = self._reconstruct_path(gcopy, parent, u)
            expanded_path = self._build_path_data(gcopy, path_u)
            partial_paths.append(expanded_path)

            if u == end_idx:
                logger.debug("Reached end index; halting further expansions.")
                break

            for v in gcopy.neighbors(u, mode="ALL"):
                e_id = gcopy.get_eid(u, v)
                w = gcopy.es[e_id]["weight"]
                if w == float('inf'):
                    continue
                new_dist = current_dist + w
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    parent[v] = u
                    heapq.heappush(pq, (new_dist, v))
                    logger.debug(f"Updated node {v} with new_dist={new_dist} via node {u}.")

        if dist[end_idx] == float('inf'):
            logger.warning("No valid path found.")
            return [], 0.0, partial_paths, []

        distance = dist[end_idx]
        final_nodes = self._reconstruct_path(gcopy, parent, end_idx)
        path_data = self._build_path_data(gcopy, final_nodes)
        logger.info(f"Dijkstra completed: {expansions} expansions, distance={distance}, path nodes={len(final_nodes)}.")

        self.save_dijkstra_result(start, end, use_model, ship_speed, condition, path_data, distance, partial_paths)
        return path_data, distance, partial_paths, []

    def _reconstruct_path(self, g: ig.Graph, parent: List[int], target_idx: int) -> List[int]:
        path = []
        cur = target_idx
        while cur != -1:
            path.append(cur)
            cur = parent[cur]
        return path[::-1]

    def _build_path_data(self, g: ig.Graph, node_list: List[int]) -> List[dict]:
        if not node_list:
            return []
        path_data = []
        for i, node_i in enumerate(node_list):
            vx = g.vs[node_i]
            coords = (vx["lon"], vx["lat"])
            wave_data = {}
            try:
                wave_data = self.wave_data_locator.get_wave_data(coords)
            except Exception as e:
                logger.debug(f"Error retrieving wave data for node {vx['name']}: {e}")
            if i == 0:
                roll, heave, pitch = 0.0, 0.0, 0.0
            else:
                try:
                    edge_id = g.get_eid(node_list[i - 1], node_i)
                    roll = g.es[edge_id]["roll"]
                    heave = g.es[edge_id]["heave"]
                    pitch = g.es[edge_id]["pitch"]
                except Exception as e:
                    logger.debug(f"Error retrieving edge data for {node_list[i-1]}->{node_i}: {e}")
                    roll, heave, pitch = 0.0, 0.0, 0.0
            path_data.append({
                "node_id": vx["name"],
                "coordinates": list(coords),
                "htsgwsfc": float(wave_data.get("htsgwsfc", 0.0)),
                "perpwsfc": float(wave_data.get("perpwsfc", 0.0)),
                "dirpwsfc": float(wave_data.get("dirpwsfc", 0.0)),
                "Roll": float(roll),
                "Heave": float(heave),
                "Pitch": float(pitch),
                "rel_heading": 0.0
            })
        return path_data

    def save_dijkstra_result(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        use_model: bool,
        ship_speed: float,
        condition: int,
        path_data: List[dict],
        distance: float,
        partial_paths: List[List[dict]]
    ):
        wave_data_id = self._get_wave_data_identifier()
        cache_file = os.path.join(self.dijkstra_cache_dir, f"{wave_data_id}.json")
        lock_file = f"{cache_file}.lock"
        category = "with_model" if use_model else "without_model"

        item = {
            "start": list(start),
            "end": list(end),
            "use_model": use_model,
            "ship_speed": ship_speed,
            "condition": condition,
            "path": path_data,
            "distance": float(distance),
            "partial_paths": partial_paths,
            "all_edges": [],
            "timestamp": f"{datetime.utcnow().isoformat()}Z",
            "fixed": True
        }

        try:
            with FileLock(lock_file, timeout=10):
                if os.path.exists(cache_file):
                    try:
                        with open(cache_file, 'r') as f:
                            cache_json = json.load(f)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in dijkstra cache; resetting {cache_file}.")
                        cache_json = {
                            "wave_data_id": wave_data_id,
                            "dijkstra_results": {"with_model": [], "without_model": []}
                        }
                else:
                    cache_json = {
                        "wave_data_id": wave_data_id,
                        "dijkstra_results": {"with_model": [], "without_model": []}
                    }
                cache_json["dijkstra_results"][category].append(item)
                with open(cache_file, 'w') as f:
                    json.dump(cache_json, f, indent=4)
                logger.info(f"Dijkstra result saved to {cache_file} for category {category}.")
        except Exception as e:
            logger.error(f"Error saving Dijkstra result to {cache_file}: {e}")

    def load_dijkstra_result(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        use_model: bool,
        ship_speed: float,
        condition: int
    ) -> Optional[Dict[str, Any]]:
        wave_data_id = self._get_wave_data_identifier()
        cache_file = os.path.join(self.dijkstra_cache_dir, f"{wave_data_id}.json")
        if not os.path.exists(cache_file):
            logger.info(f"No dijkstra cache for wave_data_id={wave_data_id}.")
            return None

        lock_file = f"{cache_file}.lock"
        query_str = json.dumps({
            "start": list(start),
            "end": list(end),
            "use_model": use_model,
            "ship_speed": ship_speed,
            "condition": condition,
            "wave_data_id": wave_data_id
        }, sort_keys=True)
        query_key = hashlib.sha256(query_str.encode()).hexdigest()

        try:
            with FileLock(lock_file, timeout=10):
                with open(cache_file, 'r') as f:
                    try:
                        cache_json = json.load(f)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in dijkstra cache {cache_file}; resetting.")
                        cache_json = {
                            "wave_data_id": wave_data_id,
                            "dijkstra_results": {"with_model": [], "without_model": []}
                        }
                        with open(cache_file, 'w') as fw:
                            json.dump(cache_json, fw, indent=4)
                        return None
        except Exception as e:
            logger.error(f"Error loading dijkstra cache from {cache_file}: {e}")
            return None

        category = "with_model" if use_model else "without_model"
        results_list = cache_json["dijkstra_results"].get(category, [])

        for item in results_list:
            if not item.get("fixed", False):
                continue
            item_str = json.dumps({
                "start": item["start"],
                "end": item["end"],
                "use_model": item["use_model"],
                "ship_speed": item["ship_speed"],
                "condition": item["condition"],
                "wave_data_id": wave_data_id
            }, sort_keys=True)
            item_key = hashlib.sha256(item_str.encode()).hexdigest()
            if item_key == query_key:
                logger.info("Found matching fixed Dijkstra result in cache; returning result.")
                return {
                    "path": item["path"],
                    "distance": float(item["distance"]),
                    "partial_paths": item.get("partial_paths", []),
                    "all_edges": []
                }
        logger.info("No matching fixed Dijkstra result found in cache.")
        return None
