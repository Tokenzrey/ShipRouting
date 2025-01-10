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
    Manajemen cache edge predictions satu file .pkl per wave_data_id:
      - "batch_{wave_data_id}.pkl"
      - In-memory dict menampung edge predictions
      - Tulis (overwrite) file .pkl saat wave_data_id berubah, finalize(),
        atau melebihi limit (opsional LRU).
    """
    def __init__(
        self,
        cache_dir: str,
        batch_size: int = 100_000,
        max_memory_cache: int = 20_000_000,
        compression_level: int = 0
    ):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.batch_size = batch_size
        self.max_memory_cache = max_memory_cache
        self.compression_level = compression_level

        self.current_wave_data_id: Optional[str] = None
        self.memory_cache: Dict[str, dict] = {}
        self._dirty = False

    def _generate_edge_key(self, edge_data: dict, wave_data_id: str) -> str:
        key_data = {
            "source": edge_data["source_coords"],
            "target": edge_data["target_coords"],
            "wave_data_id": wave_data_id,
            "ship_speed": edge_data["ship_speed"],
            "condition": edge_data["condition"]
        }
        return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()

    def set_current_wave_data_id(self, wave_data_id: str):
        if wave_data_id == self.current_wave_data_id:
            return  # Tidak berubah
        if self.current_wave_data_id and self._dirty:
            self._flush_to_disk(self.current_wave_data_id)

        self.current_wave_data_id = wave_data_id
        self.memory_cache.clear()
        self._dirty = False

        # Load pkl jika ada
        pkl_file = os.path.join(self.cache_dir, f"batch_{wave_data_id}.pkl")
        if os.path.exists(pkl_file):
            lock_file = pkl_file + ".lock"
            with FileLock(lock_file):
                try:
                    with open(pkl_file, 'rb') as f:
                        loaded_data = joblib.load(f)
                    if isinstance(loaded_data, dict):
                        self.memory_cache.update(loaded_data)
                        logger.info(f"Loaded {len(loaded_data)} edge preds from {pkl_file}")
                    else:
                        logger.error(f"Unexpected data format in {pkl_file}")
                except Exception as e:
                    logger.error(f"Error loading pkl {pkl_file}: {e}")
        else:
            logger.info(f"No pkl cache for wave_data_id={wave_data_id}, starting fresh.")

    def _flush_to_disk(self, wave_data_id: str):
        if not wave_data_id or not self._dirty:
            return
        pkl_file = os.path.join(self.cache_dir, f"batch_{wave_data_id}.pkl")
        lock_file = pkl_file + ".lock"
        with FileLock(lock_file):
            try:
                with open(pkl_file, 'wb') as f:
                    joblib.dump(self.memory_cache, f, compress=self.compression_level)
                logger.info(f"Flushed {len(self.memory_cache)} entries to {pkl_file}")
            except Exception as e:
                logger.error(f"Error flushing pkl {pkl_file}: {e}")
        self._dirty = False

    def get_cached_predictions(self, edge_data: dict, wave_data_id: str) -> Optional[dict]:
        if wave_data_id != self.current_wave_data_id:
            return None
        key = self._generate_edge_key(edge_data, wave_data_id)
        return self.memory_cache.get(key)

    def _lru_cleanup(self):
        if len(self.memory_cache) > self.max_memory_cache:
            while len(self.memory_cache) > self.max_memory_cache:
                self.memory_cache.popitem()

    def save_batch(self, predictions: List[dict], edge_data: List[dict], wave_data_id: str):
        if wave_data_id != self.current_wave_data_id:
            self.set_current_wave_data_id(wave_data_id)
        for pred, ed in zip(predictions, edge_data):
            key = self._generate_edge_key(ed, wave_data_id)
            self.memory_cache[key] = pred
        self._dirty = True
        self._lru_cleanup()

    def finalize(self):
        if self.current_wave_data_id and self._dirty:
            self._flush_to_disk(self.current_wave_data_id)
        logger.info("EdgeBatchCache finalize complete.")


class RouteOptimizer:
    """
    RouteOptimizer:
     - Single file pkl per wave_data_id => cache edge predictions
     - Simpan roll/heave/pitch/isBlocked di igraph agar tidak memanggil batch process terus
     - Dijkstra caching => load/save JSON
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
        self.wave_data_locator = wave_data_locator
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

        self.edge_cache = EdgeBatchCache(
            os.path.join(DATA_DIR_CACHE, "edge_predictions"),
            batch_size=100_000,
            max_memory_cache=20_000_000,
            compression_level=0
        )

        # Pastikan atribut di igraph (edges)
        for attr in ["roll", "heave", "pitch", "isBlocked"]:
            if attr not in self.igraph_graph.es.attributes():
                if attr == "isBlocked":
                    self.igraph_graph.es[attr] = [False] * self.igraph_graph.ecount()
                else:
                    self.igraph_graph.es[attr] = [0.0] * self.igraph_graph.ecount()

    def finalize(self):
        self.edge_cache.finalize()

    def update_wave_data_locator(self, wave_data_locator):
        # Flush cache & wave data
        self.finalize()
        self.wave_data_locator = wave_data_locator
        self.cache.clear()
        logger.info("WaveDataLocator updated => caches cleared.")

    def _get_wave_data_identifier(self) -> str:
        wf = self.wave_data_locator.wave_file
        path = os.path.join(DATA_DIR_CACHE, wf)
        with open(path, 'rb') as f:
            cont = f.read()
        return hashlib.sha256(cont).hexdigest()

    def _load_or_build_graph(self) -> ig.Graph:
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
        if "roll" not in g.es.attributes():
            g.es["roll"] = [0.0] * g.ecount()
        if "heave" not in g.es.attributes():
            g.es["heave"] = [0.0] * g.ecount()
        if "pitch" not in g.es.attributes():
            g.es["pitch"] = [0.0] * g.ecount()
        if "isBlocked" not in g.es.attributes():
            g.es["isBlocked"] = [False] * g.ecount()

        g.write_pickle(self.saved_graph_file)
        logger.info("Graph built & pickled.")
        return g

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

    def _batch_process_edges(self, edges_data: List[dict], wave_data_id: str) -> List[dict]:
        self.edge_cache.set_current_wave_data_id(wave_data_id)
        chunk_size = self.edge_cache.batch_size
        results: List[dict] = []
        buffer_inputs = []
        buffer_edges = []

        def flush_chunk():
            nonlocal buffer_inputs, buffer_edges
            if not buffer_inputs:
                return []
            df = pd.DataFrame(buffer_inputs)
            blocked, roll, heave, pitch = self._predict_blocked(df)
            out = [
                {
                    "blocked": bool(b),
                    "roll": float(r),
                    "heave": float(h),
                    "pitch": float(p)
                }
                for b, r, h, p in zip(blocked, roll, heave, pitch)
            ]
            self.edge_cache.save_batch(out, buffer_edges, wave_data_id)
            buffer_inputs = []
            buffer_edges = []
            return out

        for ed in edges_data:
            cached = self.edge_cache.get_cached_predictions(ed, wave_data_id)
            if cached:
                results.append(cached)
            else:
                wave_data = self.wave_data_locator.get_wave_data(ed["target_coords"])
                bearing = self._compute_bearing(ed["source_coords"], ed["target_coords"])
                heading = self._compute_heading(bearing, float(wave_data.get("dirpwsfc", 0.0)))

                row = [
                    ed["ship_speed"],
                    heading,
                    float(wave_data.get("htsgwsfc", 0.0)),
                    float(wave_data.get("perpwsfc", 0.0)),
                    ed["condition"]
                ]
                buffer_inputs.append(row)
                buffer_edges.append(ed)
                if len(buffer_inputs) >= chunk_size:
                    chunk_results = flush_chunk()
                    results.extend(chunk_results)
        # flush sisa
        if buffer_inputs:
            chunk_results = flush_chunk()
            results.extend(chunk_results)

        self.edge_cache._flush_to_disk(wave_data_id)
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

        with FileLock(lock_file):
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

        with FileLock(lock_file):
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
    #   compute_block_status_for_all_edges => 1 kali
    # ----------------------------------------------------------
    def compute_block_status_for_all_edges(self, ship_speed: float, condition: int):
        """
        Memproses SELURUH edges => simpan roll/heave/pitch/isBlocked di igraph
        agar get_blocked_edges_in_view lebih cepat.
        """
        wave_data_id = self._get_wave_data_identifier()
        g = self.igraph_graph
        logger.info(f"Compute block status => edges={g.ecount()}, wave_data_id={wave_data_id}")

        # Build edges_data
        edges_data = []
        for e in g.es:
            s = g.vs[e.source]
            t = g.vs[e.target]
            edges_data.append({
                "edge_id": e.index,
                "source_coords": (s["lon"], s["lat"]),
                "target_coords": (t["lon"], t["lat"]),
                "ship_speed": ship_speed,
                "condition": condition
            })
        logger.info("Batch process all edges...")

        # Jalankan batch
        edge_res = self._batch_process_edges(edges_data, wave_data_id)
        if len(edge_res) != len(edges_data):
            logger.warning(f"Results mismatch: {len(edge_res)} vs {len(edges_data)}")

        # Tulis roll, heave, pitch, isBlocked
        for ed, er in zip(edges_data, edge_res):
            eid = ed["edge_id"]
            g.es[eid]["roll"] = er["roll"]
            g.es[eid]["heave"] = er["heave"]
            g.es[eid]["pitch"] = er["pitch"]
            g.es[eid]["isBlocked"] = bool(
                er["blocked"] or (er["roll"] >= 6) or (er["heave"] >= 0.7) or (er["pitch"] >= 3)
            )
        logger.info("Done storing roll/heave/pitch/isBlocked in igraph edges.")

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

            # Single lookup for vertices
            source = g.vs[edge.source]
            target = g.vs[edge.target]

            # Single lookup for coordinates
            s_lon, s_lat = source["lon"], source["lat"]
            t_lon, t_lat = target["lon"], target["lat"]

            # Single bounds check
            if ((min_lon <= s_lon <= max_lon and min_lat <= s_lat <= max_lat) or 
                (min_lon <= t_lon <= max_lon and min_lat <= t_lat <= max_lat)):
                
                edges_in_view.append({
                    "edge_id": edge.index,
                    "source_coords": (s_lon, s_lat),
                    "target_coords": (t_lon, t_lat),
                    "isBlocked": True
                })
                
                # Early exit if max reached
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
        3. Jika tidak => check isBlocked => set weight=inf => Dijkstra (optimized)
           - tetap melebarkan ke node lain (meski end_idx tercapai)
           - hentikan ekspansi jika current_dist > dist[end_idx]
           - catat partial paths (ekspansi) hanya jika ada perbaikan jarak
        4. save => return

        Return: (path_data, distance, partial_paths, [])
        """
        # Coba load dari cache
        cached = self.load_dijkstra_result(start, end, use_model, ship_speed, condition)
        if cached:
            logger.info("Using cached dijkstra from local JSON => returning.")
            return cached["path"], cached["distance"], cached["partial_paths"], cached["all_edges"]

        wave_data_id = self._get_wave_data_identifier()
        gcopy = self.igraph_graph.copy()

        # ============ Gunakan isBlocked => set weight=inf jika use_model
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

        # Priority queue => (distance, node)
        pq = [(0.0, start_idx)]

        # partial_paths => menampung jalur START -> v
        partial_paths: List[List[dict]] = []

        while pq:
            current_dist, u = heapq.heappop(pq)
            # Jika jarak ini sudah melebihi jarak end_idx, hentikan ekspansi
            if current_dist > dist[end_idx]:
                continue
            # Jika jarak ini bukan jarak teraktual => skip
            if current_dist > dist[u]:
                continue

            # Simpan partial path => path START->u
            path_u = self._reconstruct_path(gcopy, parent, u)
            expanded_path = self._build_path_data(gcopy, path_u)
            partial_paths.append(expanded_path)

            # Ekspansi neighbors
            for v in gcopy.neighbors(u, mode="ALL"):
                e_id = gcopy.get_eid(u, v)
                w = gcopy.es[e_id]["weight"]
                if w == float('inf'):
                    continue  # blocked, skip

                new_dist = current_dist + w
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    parent[v] = u
                    heapq.heappush(pq, (new_dist, v))

        # Reconstruct final path from start_idx to end_idx
        if dist[end_idx] == float('inf'):
            logger.warning("No path found => returning empty.")
            return [], 0.0, partial_paths, []

        distance = dist[end_idx]
        final_nodes = self._reconstruct_path(gcopy, parent, end_idx)
        path_data = self._build_path_data(gcopy, final_nodes)

        # Simpan ke cache (fixed = True)
        self.save_dijkstra_result(
            start, end, use_model, ship_speed, condition, path_data, distance, partial_paths
        )
        return path_data, distance, partial_paths, []

    # ----------------------------------------------------------
    #   Helper: Reconstruct Path (array of node indices)
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
        return path[::-1]  # dibalik

    # ----------------------------------------------------------
    #   Helper: Build Path Data => [ {node_id, coords, wave, roll, etc}, ...]
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
            except:
                pass

            # roll/heave/pitch => ambil dari edge (kecuali node pertama, set 0)
            if i == 0:
                roll = 0.0
                heave = 0.0
                pitch = 0.0
            else:
                edge_id = g.get_eid(node_list[i - 1], node_i)
                roll = g.es[edge_id]["roll"]
                heave = g.es[edge_id]["heave"]
                pitch = g.es[edge_id]["pitch"]

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
