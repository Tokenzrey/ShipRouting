import os
import json
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import igraph as ig
from filelock import FileLock
from keras.api.models import load_model
import joblib
import tensorflow as tf
from rtree import index

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
        batch_size: int = 1000000,
        max_memory_cache: int = 20_000_000,
        compression_level: int = 0
    ):
        """
        :param cache_dir: Folder untuk file .pkl
        :param batch_size: Ukuran batch pemrosesan edge
        :param max_memory_cache: Batas jumlah entri in-memory
        :param compression_level: Level kompresi joblib (0 => tanpa kompresi => cepat)
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.batch_size = batch_size
        self.max_memory_cache = max_memory_cache
        self.compression_level = compression_level

        # In-memory untuk wave_data_id aktif
        self.current_wave_data_id: Optional[str] = None
        self.memory_cache: Dict[str, dict] = {}
        self._dirty = False

    def _generate_edge_key(self, edge_data: dict, wave_data_id: str) -> str:
        """
        Buat key unik (sha256) dari source, target, wave_data_id, ship_speed, condition.
        """
        key_data = {
            "source": edge_data["source_coords"],
            "target": edge_data["target_coords"],
            "wave_data_id": wave_data_id,
            "ship_speed": edge_data["ship_speed"],
            "condition": edge_data["condition"]
        }
        return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()

    def set_current_wave_data_id(self, wave_data_id: str):
        """
        Ganti wave_data_id aktif. 
        - Flush yang lama jika _dirty
        - Load file pkl wave_data_id baru kalau ada
        """
        if wave_data_id == self.current_wave_data_id:
            return  # Tidak berubah

        # Flush wave_data_id lama jika masih dirty
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
        """
        Overwrite file pkl "batch_{wave_data_id}.pkl" dengan memory_cache.
        """
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
        """
        Return prediksi edge in-memory jika wave_data_id cocok dan ada.
        """
        if wave_data_id != self.current_wave_data_id:
            return None
        key = self._generate_edge_key(edge_data, wave_data_id)
        return self.memory_cache.get(key)

    def _lru_cleanup(self):
        """
        Placeholder LRU. 
        Sederhana: pop item random jika melebihi max_memory_cache.
        """
        if len(self.memory_cache) > self.max_memory_cache:
            while len(self.memory_cache) > self.max_memory_cache:
                self.memory_cache.popitem()

    def save_batch(self, predictions: List[dict], edge_data: List[dict], wave_data_id: str):
        """
        Simpan batch hasil prediksi ke in-memory. 
        Tidak langsung flush agar cepat, 
        flush dilakukan di akhir atau saat wave_data_id ganti.
        """
        if wave_data_id != self.current_wave_data_id:
            self.set_current_wave_data_id(wave_data_id)

        for pred, ed in zip(predictions, edge_data):
            key = self._generate_edge_key(ed, wave_data_id)
            self.memory_cache[key] = pred
        self._dirty = True

        self._lru_cleanup()

    def finalize(self):
        """
        Flush ke disk jika dirty.
        """
        if self.current_wave_data_id and self._dirty:
            self._flush_to_disk(self.current_wave_data_id)
        logger.info("EdgeBatchCache finalize complete.")

class RouteOptimizer:
    """
    RouteOptimizer siap produksi:
    - Single file pkl per wave_data_id untuk edge caching
    - Optional GPU/multithread via TensorFlow kalau tersedia
    - Memiliki load/save dijkstra result
    - Menyimpan partial paths
    - Menyediakan fungsi untuk mendapatkan edges yang diblokir dalam view
    """
    def __init__(
        self,
        graph_file: str,
        wave_data_locator,
        model_path: str,
        input_scaler_pkl: str,
        output_scaler_pkl: str,
        grid_locator: GridLocator = None
    ):
        """
        :param graph_file: Path ke JSON graph
        :param wave_data_locator: WaveDataLocator
        :param model_path: Keras model path
        :param input_scaler_pkl: scaler input path
        :param output_scaler_pkl: scaler output path
        :param grid_locator: GridLocator
        """
        setup_tf_for_production()

        self.graph_file = graph_file
        self.wave_data_locator = wave_data_locator
        self.model_path = model_path
        self.input_scaler_pkl = input_scaler_pkl
        self.output_scaler_pkl = output_scaler_pkl
        self.grid_locator = grid_locator

        self.saved_graph_file = "region_graph.pkl"
        self.igraph_graph = self._load_or_build_graph()

        # Dijkstra in-memory cache
        self.cache: Dict[str, Tuple[List[dict], float, List[List[dict]], List[dict]]] = {}

        logger.info(f"Loading ML model from {self.model_path} ...")
        self.model = load_model(self.model_path, compile=False)

        logger.info(f"Loading input scaler from {self.input_scaler_pkl} ...")
        self.input_scaler = joblib.load(self.input_scaler_pkl)

        logger.info(f"Loading output scaler from {self.output_scaler_pkl} ...")
        self.output_scaler = joblib.load(self.output_scaler_pkl)

        # Dijkstra cache folder
        self.dijkstra_cache_dir = os.path.join(DATA_DIR, "dijkstra")
        os.makedirs(self.dijkstra_cache_dir, exist_ok=True)

        # EdgeBatchCache single-file pkl
        self.edge_cache = EdgeBatchCache(
            os.path.join(DATA_DIR_CACHE, "edge_predictions"),
            batch_size=100000,
            max_memory_cache=20_000_000,
            compression_level=0
        )

    def finalize(self):
        """
        Dipanggil di shutdown => flush edge_cache
        """
        self.edge_cache.finalize()
        # Tidak ada resource khusus untuk Rtree yang perlu di-finalize

    def update_wave_data_locator(self, wave_data_locator):
        """
        Bila wave data berganti => flush edge cache, clear dijkstra cache
        """
        self.finalize()
        self.wave_data_locator = wave_data_locator
        self.cache.clear()
        logger.info("WaveDataLocator updated, caches cleared, spatial index rebuilt.")

    def _get_wave_data_identifier(self) -> str:
        wf = self.wave_data_locator.wave_file
        path = os.path.join(DATA_DIR_CACHE, wf)
        with open(path, 'rb') as f:
            cont = f.read()
        return hashlib.sha256(cont).hexdigest()

    def _load_or_build_graph(self) -> ig.Graph:
        """
        Load the graph from a saved pickle if available,
        otherwise build from JSON, then save it.
        """
        if os.path.exists(self.saved_graph_file):
            logger.info(f"Loading graph from {self.saved_graph_file}...")
            try:
                g = ig.Graph.Read_Pickle(self.saved_graph_file)
                logger.info(f"Graph loaded: {g.vcount()} vertices, {g.ecount()} edges.")
                return g
            except Exception as e:
                logger.error(f"Failed to load {self.saved_graph_file}: {e}. Rebuilding from JSON...")

        # Rebuild from JSON
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

        # Pastikan semua edge memiliki atribut 'roll', 'heave', dan 'pitch'
        # Inisialisasi dengan nilai default 0.0 jika tidak ada
        for attr in ['roll', 'heave', 'pitch']:
            if attr not in g.es.attributes():
                g.es[attr] = [0.0] * g.ecount()
            else:
                # Ganti None atau nilai yang tidak valid dengan 0.0
                g.es[attr] = [
                    float(val) if val is not None else 0.0
                    for val in g.es[attr]
                ]

        g.write_pickle(self.saved_graph_file)
        logger.info("Graph built & pickled.")
        return g

    def _compute_bearing(self, start: Tuple[float, float], end: Tuple[float, float]) -> float:
        """
        Bearing start->end [0..360).
        """
        lon1, lat1 = np.radians(start)
        lon2, lat2 = np.radians(end)
        dlon = lon2 - lon1

        x = np.sin(dlon) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(dlon))

        ib = np.degrees(np.arctan2(x, y))
        return (ib + 360) % 360

    def _compute_heading(self, bearing: float, dirpwsfc: float) -> float:
        """
        Menambahkan 90 derajat sesuai requirement (silakan sesuaikan).
        """
        return (bearing - dirpwsfc + 90) % 360

    def _predict_blocked(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Memprediksi blocked, roll, heave, pitch. 
        Pastikan semua dikonversi ke float agar JSON-serializable.
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

    def save_dijkstra_result(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        use_model: bool,
        ship_speed: float,
        condition: int,
        path: List[dict],
        distance: float,
        partial_paths: List[List[dict]],
        all_edges: List[dict]  # Akan diabaikan, tapi parameter masih ada untuk konsistensi
    ):
        """
        Menyimpan hasil Dijkstra ke JSON <wave_data_id>.json,
        dengan struktur with_model / without_model,
        termasuk partial_paths.
        all_edges diabaikan dan diisi kosong.
        """
        wave_data_id = self._get_wave_data_identifier()
        cache_file = os.path.join(self.dijkstra_cache_dir, f"{wave_data_id}.json")
        lock_file = f"{cache_file}.lock"
        category = "with_model" if use_model else "without_model"

        data = {
            "start": list(start),
            "end": list(end),
            "use_model": use_model,
            "ship_speed": ship_speed,
            "condition": condition,
            "path": path,
            "distance": float(distance),
            "partial_paths": partial_paths,
            "all_edges": [],  # Mengembalikan array kosong
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        with FileLock(lock_file):
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        cache_json = json.load(f)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to load cache file {cache_file}: {e}. Resetting cache.")
                    cache_json = {
                        "wave_data_id": wave_data_id,
                        "dijkstra_results": {
                            "with_model": [],
                            "without_model": []
                        }
                    }
            else:
                cache_json = {
                    "wave_data_id": wave_data_id,
                    "dijkstra_results": {
                        "with_model": [],
                        "without_model": []
                    }
                }

            cache_json["dijkstra_results"][category].append(data)
            try:
                with open(cache_file, 'w') as f:
                    json.dump(cache_json, f, indent=4)
                logger.info(f"Dijkstra result saved to {cache_file} ({category}).")
            except Exception as e:
                logger.error(f"Failed to save dijkstra result to {cache_file}: {e}")

    def load_dijkstra_result(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        use_model: bool,
        ship_speed: float,
        condition: int
    ) -> Optional[Dict[str, Any]]:
        """
        Memuat hasil Dijkstra dari <wave_data_id>.json
        Termasuk path, distance, partial_paths
        all_edges diabaikan dan dikembalikan sebagai array kosong
        """
        wave_data_id = self._get_wave_data_identifier()
        cache_file = os.path.join(self.dijkstra_cache_dir, f"{wave_data_id}.json")
        if not os.path.exists(cache_file):
            logger.info(f"No dijkstra cache file for wave_data_id={wave_data_id}.")
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
            with FileLock(lock_file):
                with open(cache_file, 'r') as f:
                    cache_json = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to load cache file {cache_file}: {e}. Resetting cache.")
            cache_json = {
                "wave_data_id": wave_data_id,
                "dijkstra_results": {
                    "with_model": [],
                    "without_model": []
                }
            }
            with FileLock(lock_file):
                with open(cache_file, 'w') as f:
                    json.dump(cache_json, f, indent=4)
            return None
        except Exception as e:
            logger.error(f"Error loading dijkstra cache file {cache_file}: {e}")
            return None

        category = "with_model" if use_model else "without_model"

        for item in cache_json.get("dijkstra_results", {}).get(category, []):
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
                logger.info("Found dijkstra result in local JSON cache.")
                return {
                    "path": item["path"],
                    "distance": float(item["distance"]),
                    "partial_paths": item.get("partial_paths", []),
                    "all_edges": []  # Mengembalikan array kosong
                }

        logger.info("No matching dijkstra result in local JSON.")
        return None

    def _batch_process_edges(self, edges_data: List[dict], wave_data_id: str) -> List[dict]:
        """
        Memproses edges secara batch => blocked, roll, heave, pitch
        Menggunakan chunk & EdgeBatchCache single-file pkl.
        Setelah seluruh batch selesai, kita flush ke disk.
        Optimasi:
          - Memproses sebanyak mungkin edge dalam satu batch
          - Mengurangi overhead dengan meminimalisir operasi loop
        """
        self.edge_cache.set_current_wave_data_id(wave_data_id)
        chunk_size = self.edge_cache.batch_size
        results = []

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
            # Simpan batch ke in-memory
            self.edge_cache.save_batch(out, buffer_edges, wave_data_id)
            buffer_inputs = []
            buffer_edges = []
            return out

        # Optimasi: Menggunakan iterasi lebih cepat dan mengurangi fungsi panggilan
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

        # Flush sisa edges
        if buffer_inputs:
            chunk_results = flush_chunk()
            results.extend(chunk_results)

        # Setelah semua batch selesai, flush ke disk 
        self.edge_cache._flush_to_disk(wave_data_id)

        return results

    def find_shortest_path(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        use_model: bool = False,
        ship_speed: float = 8,
        condition: int = 1
    ) -> Tuple[List[dict], float, List[List[dict]], List[dict]]:
        """
        Cari path terpendek. 
        - use_model=True => block edge via wave_data + ML 
        - jika no-model => tetap gunakan roll, heave, pitch dari edge cache
        - Mengembalikan path_data, distance, partial_paths, dan all_edges (kosong)
        Optimasi:
          - Menggunakan Dijkstra bawaan igraph jika memungkinkan
          - Mengoptimalkan batch processing edge
        """
        if not self.wave_data_locator:
            raise ValueError("WaveDataLocator belum diinisialisasi.")

        # Coba load dari cache
        cached = self.load_dijkstra_result(start, end, use_model, ship_speed, condition)
        if cached:
            logger.info("Using cached dijkstra result from local JSON.")
            return (
                cached["path"],
                cached["distance"],
                cached.get("partial_paths", []),
                cached.get("all_edges", [])
            )

        wave_data_id = self._get_wave_data_identifier()
        start_idx = self.grid_locator.find_nearest_node(*start)
        end_idx = self.grid_locator.find_nearest_node(*end)

        if start_idx < 0 or end_idx < 0:
            logger.error("Invalid start or end index.")
            return [], 0.0, [], []

        if use_model:
            # ========== Dijkstra dengan model (blocking edges) ==========
            gcopy = self.igraph_graph.copy()
            edges_data = []

            for e in gcopy.es:
                s = gcopy.vs[e.source]
                t = gcopy.vs[e.target]
                edge_info = {
                    "edge_id": e.index,
                    "source_coords": (s["lon"], s["lat"]),
                    "target_coords": (t["lon"], t["lat"]),
                    "ship_speed": ship_speed,
                    "condition": condition
                }
                edges_data.append(edge_info)

            logger.info(f"Batch processing edges => wave_data_id={wave_data_id} ...")
            edge_res = self._batch_process_edges(edges_data, wave_data_id)

            # Update graph edges: if blocked or roll/heave/pitch melebihi ambang => weight=inf
            # Optimasi: Gunakan numpy untuk operasi batch
            is_blocked = np.array([
                er["blocked"] or er["roll"] >= 6 or er["heave"] >= 0.7 or er["pitch"] >= 3
                for er in edge_res
            ])
            gcopy.es["weight"] = np.where(is_blocked, float('inf'), gcopy.es["weight"])

            # Simpan informasi untuk all_edges (akan diabaikan, mengembalikan array kosong)
            all_edges = []  # Diubah menjadi array kosong

            # Menggunakan Dijkstra Bawaan igraph
            try:
                shortest_paths = gcopy.get_shortest_paths(
                    start_idx,
                    to=end_idx,
                    weights="weight",
                    output="vpath"
                )[0]
                if not shortest_paths:
                    logger.warning("No path found with model using igraph's Dijkstra.")
                    return [], 0.0, [], all_edges
                distance = gcopy.shortest_paths(source=start_idx, target=end_idx, weights="weight")[0][0]
            except Exception as e:
                logger.error(f"Error during Dijkstra with model: {e}")
                return [], 0.0, [], all_edges

            # Convert path_ids to path_data
            path_data = []
            for i, node_i in enumerate(shortest_paths):
                vx = gcopy.vs[node_i]
                coords = (vx["lon"], vx["lat"])

                wave_data = {}
                try:
                    wave_data = self.wave_data_locator.get_wave_data(coords)
                except Exception as e:
                    logger.warning(f"WaveData error: {e}")

                roll = heave = pitch = 0.0

                if i < len(shortest_paths) - 1:
                    nxt = gcopy.vs[shortest_paths[i + 1]]
                    eid = gcopy.get_eid(node_i, nxt.index, directed=False, error=False)
                    if eid != -1:
                        # Pastikan atribut ada sebelum mengakses
                        roll = float(gcopy.es[eid]["roll"]) if "roll" in gcopy.es[eid].attributes() else 0.0
                        heave = float(gcopy.es[eid]["heave"]) if "heave" in gcopy.es[eid].attributes() else 0.0
                        pitch = float(gcopy.es[eid]["pitch"]) if "pitch" in gcopy.es[eid].attributes() else 0.0

                    bearing = self._compute_bearing(coords, (nxt["lon"], nxt["lat"]))
                    heading = self._compute_heading(bearing, float(wave_data.get("dirpwsfc", 0.0)))
                else:
                    heading = 0.0  # No heading for the last node

                path_data.append({
                    "node_id": vx["name"],
                    "coordinates": list(coords),
                    "htsgwsfc": float(wave_data.get("htsgwsfc", 0.0)),
                    "perpwsfc": float(wave_data.get("perpwsfc", 0.0)),
                    "dirpwsfc": float(wave_data.get("dirpwsfc", 0.0)),
                    "Roll": roll,
                    "Heave": heave,
                    "Pitch": pitch,
                    "rel_heading": heading
                })

            # Partial paths dapat diambil dari path langkah demi langkah
            partial_paths = []
            for i in range(1, len(shortest_paths)):
                partial_path = []
                for node_id in shortest_paths[:i+1]:
                    vx = gcopy.vs[node_id]
                    coords = (vx["lon"], vx["lat"])
                    wave_data = {}
                    try:
                        wave_data = self.wave_data_locator.get_wave_data(coords)
                    except Exception as e:
                        logger.warning(f"WaveData error: {e}")

                    partial_path.append({
                        "node_id": vx["name"],
                        "coordinates": list(coords),
                        "htsgwsfc": float(wave_data.get("htsgwsfc", 0.0)),
                        "perpwsfc": float(wave_data.get("perpwsfc", 0.0)),
                        "dirpwsfc": float(wave_data.get("dirpwsfc", 0.0))
                    })
                partial_paths.append(partial_path)

            self.save_dijkstra_result(
                start, end, use_model, ship_speed, condition, path_data, distance, partial_paths, all_edges
            )
            return path_data, distance, partial_paths, all_edges

        else:
            # ========== Dijkstra standar (tanpa blocking), tapi roll/heave/pitch tetap dari cache ==========
            # Menggunakan Dijkstra Bawaan igraph
            try:
                shortest_paths = self.igraph_graph.get_shortest_paths(
                    start_idx,
                    to=end_idx,
                    weights="weight",
                    output="vpath"
                )[0]
                if not shortest_paths:
                    logger.warning("No path found (no-model) using igraph's Dijkstra.")
                    return [], 0.0, [], []
                distance = self.igraph_graph.shortest_paths(source=start_idx, target=end_idx, weights="weight")[0][0]
            except Exception as e:
                logger.error(f"Error during Dijkstra without model: {e}")
                return [], 0.0, [], []

            # Convert path_ids to path_data
            path_data = []
            for i, node_i in enumerate(shortest_paths):
                vx = self.igraph_graph.vs[node_i]
                coords = (vx["lon"], vx["lat"])

                wave_data = {}
                try:
                    wave_data = self.wave_data_locator.get_wave_data(coords)
                except Exception as e:
                    logger.warning(f"WaveData error: {e}")

                roll = heave = pitch = 0.0
                heading = 0.0

                if i < len(shortest_paths) - 1:
                    nxt = self.igraph_graph.vs[shortest_paths[i + 1]]
                    eid = self.igraph_graph.get_eid(node_i, nxt.index, directed=False, error=False)
                    if eid != -1:
                        cached = self.edge_cache.get_cached_predictions({
                            "source_coords": (vx["lon"], vx["lat"]),
                            "target_coords": (nxt["lon"], nxt["lat"]),
                            "ship_speed": ship_speed,
                            "condition": condition
                        }, wave_data_id)
                        if cached:
                            roll = float(cached.get("roll", 0.0))
                            heave = float(cached.get("heave", 0.0))
                            pitch = float(cached.get("pitch", 0.0))
                        else:
                            # Pastikan atribut ada sebelum mengakses
                            roll = float(self.igraph_graph.es[eid]["roll"]) if "roll" in self.igraph_graph.es[eid].attributes() else 0.0
                            heave = float(self.igraph_graph.es[eid]["heave"]) if "heave" in self.igraph_graph.es[eid].attributes() else 0.0
                            pitch = float(self.igraph_graph.es[eid]["pitch"]) if "pitch" in self.igraph_graph.es[eid].attributes() else 0.0

                        bearing = self._compute_bearing(coords, (nxt["lon"], nxt["lat"]))
                        heading = self._compute_heading(bearing, float(wave_data.get("dirpwsfc", 0.0)))

                path_data.append({
                    "node_id": vx["name"],
                    "coordinates": list(coords),
                    "htsgwsfc": float(wave_data.get("htsgwsfc", 0.0)),
                    "perpwsfc": float(wave_data.get("perpwsfc", 0.0)),
                    "dirpwsfc": float(wave_data.get("dirpwsfc", 0.0)),
                    "Roll": roll,
                    "Heave": heave,
                    "Pitch": pitch,
                    "rel_heading": heading
                })

            # Partial paths dapat diambil dari path langkah demi langkah
            partial_paths = []
            for i in range(1, len(shortest_paths)):
                partial_path = []
                for node_id in shortest_paths[:i+1]:
                    vx = self.igraph_graph.vs[node_id]
                    coords = (vx["lon"], vx["lat"])
                    wave_data = {}
                    try:
                        wave_data = self.wave_data_locator.get_wave_data(coords)
                    except Exception as e:
                        logger.warning(f"WaveData error: {e}")

                    partial_path.append({
                        "node_id": vx["name"],
                        "coordinates": list(coords),
                        "htsgwsfc": float(wave_data.get("htsgwsfc", 0.0)),
                        "perpwsfc": float(wave_data.get("perpwsfc", 0.0)),
                        "dirpwsfc": float(wave_data.get("dirpwsfc", 0.0))
                    })
                partial_paths.append(partial_path)

            # **Mengumpulkan semua edges dalam graph untuk visualisasi (Diabaikan)**
            all_edges = []  # Diubah menjadi array kosong

            self.save_dijkstra_result(
                start, end, use_model, ship_speed, condition, path_data, distance, partial_paths, all_edges
            )
            return path_data, distance, partial_paths, all_edges
