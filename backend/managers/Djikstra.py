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
    Opsional: Konfigurasi TensorFlow agar memanfaatkan GPU / multithread jika ada.
    Jika tidak ada GPU, tetap jalan single-thread CPU.
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
    Manajemen cache edge predictions menggunakan wave_data_id tetap:
      - "batch_{fixed_wave_data_id}.pkl"
      - In-memory dict menampung edge predictions
      - Tidak menulis ulang file .pkl
    """
    def __init__(
        self,
        cache_dir: str,
        fixed_wave_data_id: str,
        batch_size: int = 100_000,
        max_memory_cache: int = 20_000_000,
        compression_level: int = 3  # Menggunakan kompresi moderat
    ):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.batch_size = batch_size
        self.max_memory_cache = max_memory_cache
        self.compression_level = compression_level

        # wave_data_id DIPAKSA
        self.fixed_wave_data_id = fixed_wave_data_id

        self.current_wave_data_id: Optional[str] = None
        self.memory_cache: Dict[str, dict] = {}
        self._dirty = False

        # Set wave_data_id saat inisialisasi
        self.set_current_wave_data_id(self.fixed_wave_data_id)

    def _generate_edge_key(self, edge_data: dict) -> str:
        """
        Menghasilkan key dari edge_data:
          - Sumber, target, ship_speed, condition
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
        Abaikan parameter, pakai self.fixed_wave_data_id.
        Hanya load file batch_<fixed_wave_data_id>.pkl
        """
        forced_wid = self.fixed_wave_data_id
        if forced_wid == self.current_wave_data_id:
            return  # Tidak berubah

        # Flush cache lama jika perlu (tidak akan terjadi karena fixed_wave_data_id tetap)
        if self.current_wave_data_id and self._dirty:
            logger.info("self dirty")
            self._flush_to_disk(self.current_wave_data_id)

        self.current_wave_data_id = forced_wid
        self.memory_cache.clear()
        self._dirty = False

        # Load pkl jika ada
        pkl_file = os.path.join(self.cache_dir, f"batch_{forced_wid}.pkl")
        if os.path.exists(pkl_file):
            lock_file = pkl_file + ".lock"
            try:
                with FileLock(lock_file, timeout=10):  # Timeout untuk mencegah deadlock
                    loaded_data = joblib.load(pkl_file)
                if isinstance(loaded_data, dict):
                    self.memory_cache.update(loaded_data)
                    logger.info(f"Loaded {len(loaded_data)} edge preds from {pkl_file}")
                else:
                    logger.error(f"Unexpected data format in {pkl_file}")
            except Exception as e:
                logger.error(f"Error loading pkl {pkl_file}: {e}")
        else:
            logger.info(f"No pkl cache for fixed_wave_data_id={forced_wid}, starting fresh.")

    def _flush_to_disk(self, _wave_data_id: str):
        """
        Menyimpan memory_cache ke disk jika ada perubahan.
        Dalam scenario ini, ini mungkin tidak perlu karena kita tidak ingin menulis ulang.
        """
        if not self.current_wave_data_id or not self._dirty:
            return
        pkl_file = os.path.join(self.cache_dir, f"batch_{self.fixed_wave_data_id}.pkl")
        lock_file = pkl_file + ".lock"
        try:
            with FileLock(lock_file, timeout=10):
                joblib.dump(self.memory_cache, pkl_file, compress=self.compression_level)
            logger.info(f"Flushed {len(self.memory_cache)} entries to {pkl_file}")
        except Exception as e:
            logger.error(f"Error flushing pkl {pkl_file}: {e}")
        self._dirty = False

    def get_cached_predictions(self, edge_data: dict) -> Optional[dict]:
        """
        Mengambil prediksi yang telah di-cache untuk edge tertentu.
        """
        key = self._generate_edge_key(edge_data)
        return self.memory_cache.get(key)

    def _lru_cleanup(self):
        """
        Membersihkan memory_cache jika melebihi batas max_memory_cache.
        """
        if len(self.memory_cache) > self.max_memory_cache:
            # Menghapus item tertua
            keys = list(self.memory_cache.keys())
            for key in keys[:len(self.memory_cache) - self.max_memory_cache]:
                self.memory_cache.pop(key)

    def save_batch(self, predictions: List[dict], edge_data: List[dict]):
        """
        Di scenario ini, kita **tidak** menyimpan batch baru ke cache.
        Jadi metode ini dibiarkan kosong atau dapat diberi log jika diperlukan.
        """
        logger.debug("save_batch called, but saving to cache is disabled for fixed_wave_data_id.")
        pass  # Tidak melakukan apa-apa

    def finalize(self):
        """
        Menyimpan semua cache yang belum tersimpan ke disk.
        Dalam scenario ini, ini mungkin tidak perlu karena kita tidak menulis ulang.
        """
        if self.current_wave_data_id and self._dirty:
            self._flush_to_disk(self.current_wave_data_id)
        logger.info("EdgeBatchCache finalize complete.")


class RouteOptimizer:
    """
    RouteOptimizer:
     - Single file pkl per wave_data_id => cache edge predictions
     - Simpan roll/heave/pitch/isBlocked di igraph agar tidak memanggil batch process terus
     - Dijkstra caching => load/save JSON
     - Memastikan tidak melakukan prediksi ulang dengan menggunakan fixed_wave_data_id
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
        self.wave_data_locator = wave_data_locator  # Mungkin diabaikan jika menggunakan fixed_wave_data_id
        self.model_path = model_path
        self.input_scaler_pkl = input_scaler_pkl
        self.output_scaler_pkl = output_scaler_pkl
        self.grid_locator = grid_locator

        self.saved_graph_file = "region_graph.pkl"
        self.igraph_graph = self._load_or_build_graph()

        # Local Dijkstra cache => <wave_data_id>.json
        self.cache: Dict[str, Any] = {}

        logger.info(f"Loading ML model from {self.model_path} ...")
        self.model = load_model(self.model_path, compile=False)

        logger.info(f"Loading input scaler from {self.input_scaler_pkl} ...")
        self.input_scaler = joblib.load(self.input_scaler_pkl)

        logger.info(f"Loading output scaler from {self.output_scaler_pkl} ...")
        self.output_scaler = joblib.load(self.output_scaler_pkl)

        self.dijkstra_cache_dir = os.path.join(DATA_DIR, "dijkstra")
        os.makedirs(self.dijkstra_cache_dir, exist_ok=True)

        # EdgeBatchCache dengan fixed_wave_data_id
        fixed_wave_data_id = "8f3fd6520880cccae131cfbe23640734acaccb600206c264bf9349ac0dcd9bf9"
        self.edge_cache = EdgeBatchCache(
            os.path.join(DATA_DIR_CACHE, "edge_predictions"),
            fixed_wave_data_id=fixed_wave_data_id,
            batch_size=100_000,
            max_memory_cache=20_000_000,
            compression_level=3  # Kompresi moderat untuk keseimbangan kecepatan dan ukuran
        )

        # Pastikan atribut di igraph (edges)
        for attr in ["roll", "heave", "pitch", "isBlocked"]:
            if attr not in self.igraph_graph.es.attributes():
                if attr == "isBlocked":
                    self.igraph_graph.es[attr] = [False] * self.igraph_graph.ecount()
                else:
                    self.igraph_graph.es[attr] = [0.0] * self.igraph_graph.ecount()

        # Tidak memanggil compute_block_status_for_all_edges karena kita tidak ingin memprediksi ulang
        logger.info("RouteOptimizer initialized with fixed_wave_data_id. Skipping compute_block_status_for_all_edges.")

    def finalize(self):
        self.edge_cache.finalize()

    def update_wave_data_locator(self, wave_data_locator, ship_speed: float = 8, condition: int = 1):
        """
        Memperbarui wave_data_locator, namun karena kita menggunakan fixed_wave_data_id,
        cache tetap sama dan tidak mempengaruhi data roll/heave/pitch/isBlocked.
        """
        logger.warning("update_wave_data_locator called, but using fixed_wave_data_id. Operation ignored.")
        # Tidak melakukan apa-apa karena menggunakan fixed_wave_data_id

    def _get_wave_data_identifier(self) -> str:
        """
        Menghasilkan identifier unik untuk wave data saat ini.
        Dalam scenario ini, selalu kembali ke fixed_wave_data_id.
        """
        return self.edge_cache.fixed_wave_data_id

    def _load_or_build_graph(self) -> ig.Graph:
        """
        Memuat graph dari file pickle jika ada, atau membangun dari file JSON.
        """
        if os.path.exists(self.saved_graph_file):
            logger.info(f"Loading graph from {self.saved_graph_file}...")
            try:
                g = ig.Graph.Read_Pickle(self.saved_graph_file)
                logger.info(f"Graph loaded: {g.vcount()} vertices, {g.ecount()} edges.")
                return g
            except Exception as e:
                logger.error(f"Failed to load {self.saved_graph_file}: {e}. Rebuilding from JSON...")

        with open(self.graph_file, 'r') as f:
            data = json.load(f)
        logger.info(f"Building graph. nodes={len(data['nodes'])}, edges={len(data['edges'])}")
        g = ig.Graph(n=len(data["nodes"]), directed=False)

        node_ids = list(data["nodes"].keys())
        coords = np.array([data["nodes"][nid] for nid in node_ids])
        g.vs["name"] = node_ids
        g.vs["lon"] = coords[:, 0]
        g.vs["lat"] = coords[:, 1]

        node_map = {nid: i for i, nid in enumerate(node_ids)}
        edge_list = data["edges"]
        tuples = []
        w_list = []
        b_list = []
        for ed in edge_list:
            s = node_map[ed["source"]]
            t = node_map[ed["target"]]
            w = float(ed.get("weight", 1.0))
            b = float(ed.get("bearing", 0.0))
            tuples.append((s, t))
            w_list.append(w)
            b_list.append(b)

        g.add_edges(tuples)
        g.es["weight"] = w_list
        g.es["bearing"] = b_list

        # Inisialisasi roll, heave, pitch, isBlocked di edges
        for attr in ["roll", "heave", "pitch", "isBlocked"]:
            if attr not in g.es.attributes():
                if attr == "isBlocked":
                    g.es[attr] = [False] * g.ecount()
                else:
                    g.es[attr] = [0.0] * g.ecount()

        g.write_pickle(self.saved_graph_file)
        logger.info("Graph built & pickled.")
        return g

    def _compute_bearing(self, start: Tuple[float, float], end: Tuple[float, float]) -> float:
        """
        Menghitung bearing dari titik start ke titik end.
        """
        lon1, lat1 = np.radians(start)
        lon2, lat2 = np.radians(end)
        dlon = lon2 - lon1
        x = np.sin(dlon) * np.cos(lat2)
        y = (np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon))
        ib = np.degrees(np.arctan2(x, y))
        return (ib + 360) % 360

    def _compute_heading(self, bearing: float, dirpwsfc: float) -> float:
        """
        Menghitung heading berdasarkan bearing dan dirpwsfc.
        """
        return (bearing - dirpwsfc + 90) % 360

    def _predict_blocked(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Menggunakan model ML untuk memprediksi apakah suatu edge terblokir berdasarkan fitur.
        """
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

    def _batch_process_edges(self, edges_data: List[dict], _wave_data_id: str) -> List[dict]:
        """
        Memproses batch edge untuk prediksi blokir.
        Dalam scenario ini, proses prediksi diabaikan karena menggunakan fixed_wave_data_id.
        """
        logger.debug("Batch processing skipped because using fixed_wave_data_id.")
        # Mengembalikan semua data dari cache tanpa memprediksi ulang
        results = []
        for ed in edges_data:
            cached = self.edge_cache.get_cached_predictions(ed)
            if cached:
                results.append(cached)
            else:
                # Jika data tidak ada di cache, tetapkan nilai default atau handle sesuai kebutuhan
                logger.warning(f"No cached prediction for edge_id={ed['edge_id']}. Assigning default values.")
                results.append({
                    "blocked": False,
                    "roll": 0.0,
                    "heave": 0.0,
                    "pitch": 0.0
                })
        return results

    # ----------------------------
    #  PART: Dijkstra Cache
    # ----------------------------
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
        """
        Menyimpan hasil Dijkstra baru dengan penanda "fixed": True.
        """
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
            "all_edges": [],  # Kosong
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "fixed": True   # Tandai hasil dijkstra "fixed" (baru)
        }

        try:
            with FileLock(lock_file, timeout=10):
                if os.path.exists(cache_file):
                    try:
                        with open(cache_file, 'r') as f:
                            cache_json = json.load(f)
                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON in dijkstra cache => resetting.")
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
                # Save
                with open(cache_file, 'w') as f:
                    json.dump(cache_json, f, indent=4)
                logger.info(f"Dijkstra result saved => {cache_file} ({category})")
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
        """
        Hanya mengembalikan hasil Dijkstra yang "fixed" == True.
        """
        wave_data_id = self._get_wave_data_identifier()
        cache_file = os.path.join(self.dijkstra_cache_dir, f"{wave_data_id}.json")
        if not os.path.exists(cache_file):
            logger.info(f"No dijkstra cache => wave_data_id={wave_data_id}")
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
                try:
                    with open(cache_file, 'r') as f:
                        cache_json = json.load(f)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON in dijkstra cache => resetting.")
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
            # Pastikan item "fixed" == True
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
                logger.info("Found fixed dijkstra result in local JSON => returning cache.")
                return {
                    "path": item["path"],
                    "distance": float(item["distance"]),
                    "partial_paths": item.get("partial_paths", []),
                    "all_edges": []
                }
        logger.info("No matching fixed dijkstra result in local JSON.")
        return None

    # ----------------------------------------------------------
    #   compute_block_status_for_all_edges => DIABAIKAN
    # ----------------------------------------------------------
    def compute_block_status_for_all_edges(self, ship_speed: float, condition: int):
        """
        Memproses SELURUH edges => simpan roll/heave/pitch/isBlocked di igraph
        Agar get_blocked_edges_in_view lebih cepat.
        Dalam scenario ini, proses ini diabaikan karena menggunakan fixed_wave_data_id.
        """
        logger.info("SKIPPING: compute_block_status_for_all_edges => Not re-predicting.")

    # ----------------------------------------------------------
    #   get_blocked_edges_in_view => tanpa rtree
    # ----------------------------------------------------------
    def get_blocked_edges_in_view(
            self,
            view_bounds: Tuple[float, float, float, float],
            max_blocked_edges: int = 300_000
        ) -> List[dict]:
        """
        Filter edges within view bounds that are blocked.
        Eliminates all redundant operations for maximum efficiency.
        """
        min_lon, min_lat, max_lon, max_lat = view_bounds
        g = self.igraph_graph
        edges_in_view = []

        # Directly filter and process edges
        for edge in g.es:
            # Skip immediately if not blocked
            if not edge["isBlocked"]:
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
                    "isBlocked": True
                })

                if len(edges_in_view) >= max_blocked_edges:
                    break

        logger.info(f"get_blocked_edges_in_view => total {len(edges_in_view)} blocked edges returned.")
        return edges_in_view

    # ----------------------------------------------------------
    #   find_shortest_path => optimized Dijkstra
    # ----------------------------------------------------------
    def find_shortest_path(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        use_model: bool = False,
        ship_speed: float = 8,
        condition: int = 1
    ) -> Tuple[List[dict], float, List[List[dict]], List[dict]]:
        """
        1. Coba loadDijkstra.
        2. Jika ketemu => return.
        3. Jika tidak => set weight=inf pada edges yang diblokir (jika use_model)
           dan jalankan Dijkstra.
        4. Save hasil Dijkstra ke cache.
        5. Return hasil.
        """
        # Dapatkan wave_data_id saat ini (fixed_wave_data_id)
        wave_data_id = self._get_wave_data_identifier()

        # Pastikan edge_cache diatur dengan benar (tidak berubah)
        self.edge_cache.set_current_wave_data_id(wave_data_id)

        # Proses ulang block status dengan ship_speed dan condition yang diberikan
        # Namun, dalam scenario ini, proses diabaikan
        self.compute_block_status_for_all_edges(ship_speed=ship_speed, condition=condition)

        # Coba load dari cache
        cached = self.load_dijkstra_result(start, end, use_model, ship_speed, condition)
        if cached:
            logger.info("Using cached dijkstra from local JSON => returning.")
            return cached["path"], cached["distance"], cached["partial_paths"], cached["all_edges"]

        gcopy = self.igraph_graph.copy()

        # Gunakan isBlocked => set weight=inf jika use_model
        if use_model:
            blocked_array = np.array([bool(e["isBlocked"]) for e in gcopy.es])
            gcopy.es["weight"] = np.where(blocked_array, float('inf'), gcopy.es["weight"])

        start_idx = self.grid_locator.find_nearest_node(*start)
        end_idx = self.grid_locator.find_nearest_node(*end)
        if start_idx < 0 or end_idx < 0:
            logger.error("Invalid start or end index => no path.")
            return [], 0.0, [], []

        # ----------------------
        #  Optimized Dijkstra
        # ----------------------
        dist = [float('inf')] * gcopy.vcount()
        parent = [-1] * gcopy.vcount()
        dist[start_idx] = 0.0

        pq = [(0.0, start_idx)]

        partial_paths: List[List[dict]] = []

        while pq:
            current_dist, u = heapq.heappop(pq)
            if current_dist > dist[end_idx]:
                continue
            if current_dist > dist[u]:
                continue

            path_u = self._reconstruct_path(gcopy, parent, u)
            expanded_path = self._build_path_data(gcopy, path_u)
            partial_paths.append(expanded_path)

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

        if dist[end_idx] == float('inf'):
            logger.warning("No path found => returning empty.")
            return [], 0.0, partial_paths, []

        distance = dist[end_idx]
        final_nodes = self._reconstruct_path(gcopy, parent, end_idx)
        path_data = self._build_path_data(gcopy, final_nodes)

        self.save_dijkstra_result(
            start, end, use_model, ship_speed, condition, path_data, distance, partial_paths
        )
        return path_data, distance, partial_paths, []

    # ----------------------------------------------------------
    #   Helper: Reconstruct Path
    # ----------------------------------------------------------
    def _reconstruct_path(self, g: ig.Graph, parent: List[int], target_idx: int) -> List[int]:
        """
        Rekonstruksi path (list of node indices) dengan parent[] ke target_idx.
        """
        path = []
        cur = target_idx
        while cur != -1:
            path.append(cur)
            cur = parent[cur]
        return path[::-1]

    # ----------------------------------------------------------
    #   Helper: Build Path Data
    # ----------------------------------------------------------
    def _build_path_data(self, g: ig.Graph, node_list: List[int]) -> List[dict]:
        """
        Membangun array of dict untuk jalur tertentu.
        Memasukkan wave_data (dari wave_data_locator) dan
        roll/heave/pitch (dari edge di jalur).
        """
        if not node_list:
            return []

        path_data = []
        for i, node_i in enumerate(node_list):
            vx = g.vs[node_i]
            coords = (vx["lon"], vx["lat"])

            # Wave data di vertex
            wave_data = {}
            try:
                wave_data = self.wave_data_locator.get_wave_data(coords)
            except Exception as e:
                logger.error(f"Error retrieving wave data for node {vx['name']}: {e}")

            # roll/heave/pitch => ambil dari edge (kecuali node pertama, set 0)
            if i == 0:
                roll = 0.0
                heave = 0.0
                pitch = 0.0
            else:
                try:
                    edge_id = g.get_eid(node_list[i - 1], node_i)
                    roll = g.es[edge_id]["roll"]
                    heave = g.es[edge_id]["heave"]
                    pitch = g.es[edge_id]["pitch"]
                except Exception as e:
                    logger.error(f"Error retrieving edge data between nodes {node_list[i - 1]} and {node_i}: {e}")
                    roll = heave = pitch = 0.0

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
