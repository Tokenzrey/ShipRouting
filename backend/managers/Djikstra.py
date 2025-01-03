import os
import json
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import igraph as ig
import joblib
from filelock import FileLock
from keras.api.models import load_model
# Jika Anda memakai TensorFlow:
import tensorflow as tf

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
            # Menggunakan GPU pertama jika tersedia
            tf.config.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)  # Opsional: set memory growth
            logger.info(f"GPU mode aktif: {gpus[0]}")
        else:
            # Cek jumlah CPU
            cpus = tf.config.list_physical_devices('CPU')
            if len(cpus) > 1:
                logger.info("Multiple CPU detected. TensorFlow might use multi-thread automatically.")
            else:
                logger.info("Single CPU mode.")
    except Exception as e:
        logger.warning(f"TensorFlow GPU/threads config error: {e}. Using default single-thread CPU.")

class EdgeBatchCache:
    """
    Manajemen cache edge predictions satu file .pkl per wave_data_id:
      - "batch_{wave_data_id}.pkl"
      - In-memory dict menampung edge predictions
      - Tulis (overwrite) file .pkl saat wave_data_id berubah, finalize(), atau melebihi limit
    """

    def __init__(
        self,
        cache_dir: str,
        batch_size: int = 50000,
        max_memory_cache: int = 2_000_000,
        compression_level: int = 0
    ):
        """
        :param cache_dir: Folder untuk file .pkl
        :param batch_size: Ukuran batch pemrosesan edge
        :param max_memory_cache: Batas jumlah entri in-memory (placeholder LRU)
        :param compression_level: Level kompresi joblib (0 => tanpa kompresi => ngebut)
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
            return  # tidak berubah

        # flush lama jika dirty
        if self.current_wave_data_id and self._dirty:
            self._flush_to_disk(self.current_wave_data_id)

        self.current_wave_data_id = wave_data_id
        self.memory_cache.clear()
        self._dirty = False

        # load pkl jika ada
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
                    logger.error(f"Error load pkl {pkl_file}: {e}")
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
                logger.error(f"Error flush pkl {pkl_file}: {e}")
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
        Simpan batch ke in-memory, tunda flush ke disk.
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
        # Opsional: setup TF GPU/threads
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
        self.cache: Dict[str, Tuple[List[dict], float]] = {}

        logger.info(f"Load model from {self.model_path} ...")
        self.model = load_model(self.model_path, compile=False)

        logger.info(f"Load input scaler from {self.input_scaler_pkl} ...")
        self.input_scaler = joblib.load(self.input_scaler_pkl)

        logger.info(f"Load output scaler from {self.output_scaler_pkl} ...")
        self.output_scaler = joblib.load(self.output_scaler_pkl)

        # Dijkstra cache folder
        self.dijkstra_cache_dir = os.path.join(DATA_DIR, "dijkstra")
        os.makedirs(self.dijkstra_cache_dir, exist_ok=True)

        # EdgeBatchCache single-file pkl
        self.edge_cache = EdgeBatchCache(
            os.path.join(DATA_DIR_CACHE, "edge_predictions"),
            batch_size=500000,
            max_memory_cache=20_000_000,  # Batas in-memory
            compression_level=0          # No compression => fastest
        )

    def finalize(self):
        """
        Dipanggil di shutdown => flush edge_cache
        """
        self.edge_cache.finalize()

    def update_wave_data_locator(self, wave_data_locator):
        """
        Bila wave data berganti => flush edge cache, clear dijkstra cache
        """
        self.finalize()
        self.wave_data_locator = wave_data_locator
        self.cache.clear()
        logger.info("WaveDataLocator updated, caches cleared.")

    def _get_wave_data_identifier(self) -> str:
        wf = self.wave_data_locator.current_wave_file
        path = os.path.join(DATA_DIR_CACHE, wf)
        with open(path, 'rb') as f:
            cont = f.read()
        return hashlib.sha256(cont).hexdigest()

    def _load_or_build_graph(self)->ig.Graph:
        """
        Load .pkl jika ada, kalau gagal, build dari JSON => simpan .pkl => return.
        """
        if os.path.exists(self.saved_graph_file):
            logger.info(f"Loading graph from {self.saved_graph_file}...")
            try:
                g = ig.Graph.Read_Pickle(self.saved_graph_file)
                logger.info(f"Graph loaded: {g.vcount()} vertices, {g.ecount()} edges.")
                return g
            except Exception as e:
                logger.error(f"Failed to load {self.saved_graph_file}: {e}. Rebuild from JSON...")

        # Build from JSON
        with open(self.graph_file, 'r') as f:
            data = json.load(f)
        logger.info(f"Building graph. nodes={len(data['nodes'])}, edges={len(data['edges'])}")
        g = ig.Graph(n=len(data["nodes"]), directed=False)

        node_ids = list(data["nodes"].keys())
        coords = np.array([data["nodes"][nid] for nid in node_ids])
        g.vs["name"] = node_ids
        g.vs["lon"]  = coords[:,0]
        g.vs["lat"]  = coords[:,1]

        node_map = {nid:i for i,nid in enumerate(node_ids)}

        edge_list = data["edges"]
        tuples = []
        w_list = []
        b_list = []
        for ed in edge_list:
            s = node_map[ed["source"]]
            t = node_map[ed["target"]]
            w = float(ed.get("weight",1.0))
            b = float(ed.get("bearing",0.0))
            tuples.append((s,t))
            w_list.append(w)
            b_list.append(b)

        g.add_edges(tuples)
        g.es["weight"]  = w_list
        g.es["bearing"] = b_list

        g.write_pickle(self.saved_graph_file)
        logger.info("Graph built & pickled.")
        return g

    def _compute_bearing(self, start:Tuple[float,float], end:Tuple[float,float])->float:
        """
        Bearing start->end [0..360)
        """
        lon1, lat1 = np.radians(start)
        lon2, lat2 = np.radians(end)
        dlon = lon2 - lon1
        x = np.sin(dlon)*np.cos(lat2)
        y = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
        ib = np.degrees(np.arctan2(x,y))
        return (ib+360)%360

    def _compute_heading(self, bearing: float, dirpwsfc: float)->float:
        return (bearing - dirpwsfc)%360

    def _predict_blocked(self, df: pd.DataFrame):
        colnames = ["ship_speed","wave_heading","wave_height","wave_period","condition"]
        df.columns = colnames
        scaled = self.input_scaler.transform(df)
        preds  = self.model.predict(scaled, verbose=0)
        unscaled = self.output_scaler.inverse_transform(preds)

        roll  = unscaled[:,0].astype(float)
        heave = unscaled[:,1].astype(float)
        pitch = unscaled[:,2].astype(float)
        blocked = (roll>=6)|(heave>=0.7)|(pitch>=3)
        return blocked, roll, heave, pitch

    def save_dijkstra_result(
        self,
        start: Tuple[float,float],
        end:   Tuple[float,float],
        use_model: bool,
        ship_speed: float,
        condition: int,
        path: List[dict],
        distance: float
    ):
        """
        Menyimpan hasil Dijkstra ke JSON <wave_data_id>.json,
        dengan struktur 'with_model' vs 'without_model'.
        """
        wave_data_id = self._get_wave_data_identifier()
        cache_file = os.path.join(self.dijkstra_cache_dir, f"{wave_data_id}.json")
        lock_file  = cache_file + ".lock"
        category   = "with_model" if use_model else "without_model"

        data = {
            "start": list(start),
            "end":   list(end),
            "use_model": use_model,
            "ship_speed": ship_speed,
            "condition": condition,
            "path": path,
            "distance": float(distance),
            "timestamp": datetime.utcnow().isoformat()+"Z"
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
                        "dijkstra_results":{
                            "with_model":[],
                            "without_model":[]
                        }
                    }
            else:
                cache_json = {
                    "wave_data_id": wave_data_id,
                    "dijkstra_results":{
                        "with_model":[],
                        "without_model":[]
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
        start: Tuple[float,float],
        end:   Tuple[float,float],
        use_model: bool,
        ship_speed: float,
        condition: int
    ) -> Optional[Tuple[List[dict], float]]:
        """
        Load dijkstra result from <wave_data_id>.json
        """
        wave_data_id = self._get_wave_data_identifier()
        cache_file = os.path.join(self.dijkstra_cache_dir, f"{wave_data_id}.json")
        if not os.path.exists(cache_file):
            logger.info(f"No dijkstra cache file for wave_data_id={wave_data_id}.")
            return None

        lock_file = cache_file + ".lock"
        query_str = json.dumps({
            "start": list(start),
            "end":   list(end),
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
            # Reset cache file
            cache_json = {
                "wave_data_id": wave_data_id,
                "dijkstra_results":{
                    "with_model":[],
                    "without_model":[]
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
        for item in cache_json.get("dijkstra_results",{}).get(category,[]):
            # Buat string item serupa
            item_str = json.dumps({
                "start": item["start"],
                "end":   item["end"],
                "use_model": item["use_model"],
                "ship_speed": item["ship_speed"],
                "condition": item["condition"],
                "wave_data_id": wave_data_id
            }, sort_keys=True)
            item_key = hashlib.sha256(item_str.encode()).hexdigest()

            if item_key == query_key:
                logger.info("Found dijkstra result in local JSON cache.")
                return (item["path"], float(item["distance"]))

        logger.info("No matching dijkstra result in local JSON.")
        return None

    def _batch_process_edges(self, edges_data: List[dict], wave_data_id: str)->List[dict]:
        """
        Batch process edges => blocked, roll, heave, pitch
        Memakai chunk & EdgeBatchCache single-file pkl
        """
        self.edge_cache.set_current_wave_data_id(wave_data_id)
        chunk_size = 100000  # misal chunk 10k
        results = []

        buffer_inputs = []
        buffer_edges  = []

        def flush_chunk():
            if not buffer_inputs:
                return []
            df = pd.DataFrame(buffer_inputs)
            blocked, roll, heave, pitch = self._predict_blocked(df)
            out = []
            for i in range(len(buffer_inputs)):
                out.append({
                    "blocked": bool(blocked[i]),
                    "roll": float(roll[i]),
                    "heave": float(heave[i]),
                    "pitch": float(pitch[i])
                })
            self.edge_cache.save_batch(out, buffer_edges, wave_data_id)
            return out

        for ed in edges_data:
            cached = self.edge_cache.get_cached_predictions(ed, wave_data_id)
            if cached:
                results.append(cached)
            else:
                # build input row
                wave_data = self.wave_data_locator.get_wave_data(ed["target_coords"])
                bearing = self._compute_bearing(ed["source_coords"], ed["target_coords"])
                heading = self._compute_heading(bearing, wave_data.get("dirpwsfc",0.0))

                row = [
                    ed["ship_speed"],
                    heading,
                    wave_data.get("htsgwsfc",0.0),
                    wave_data.get("perpwsfc",0.0),
                    ed["condition"]
                ]
                buffer_inputs.append(row)
                buffer_edges.append(ed)

                if len(buffer_inputs)>=chunk_size:
                    out = flush_chunk()
                    results.extend(out)
                    buffer_inputs.clear()
                    buffer_edges.clear()

        if buffer_inputs:
            out = flush_chunk()
            results.extend(out)
            buffer_inputs.clear()
            buffer_edges.clear()

        return results

    def _check_edge_blocked(
        self,
        source_node_id: str,
        target_node_id: str,
        ship_speed: float,
        condition: int
    )->Tuple[bool, float, float, float]:
        """
        Hanya dipakai di mode tanpa use_model (untuk sekadar prediksi).
        """
        try:
            source_vertex = self.igraph_graph.vs.find(name=source_node_id)
            target_vertex = self.igraph_graph.vs.find(name=target_node_id)
            source_coords = (source_vertex["lon"], source_vertex["lat"])
            target_coords = (target_vertex["lon"], target_vertex["lat"])

            wave_data = self.wave_data_locator.get_wave_data(target_coords)
            bearing   = self._compute_bearing(source_coords, target_coords)
            heading   = self._compute_heading(bearing, wave_data.get("dirpwsfc",0.0))

            df = pd.DataFrame([[
                ship_speed,
                heading,
                wave_data.get("htsgwsfc",0.0),
                wave_data.get("perpwsfc",0.0),
                condition
            ]])
            blocked, roll, heave, pitch = self._predict_blocked(df)
            return bool(blocked[0]), float(roll[0]), float(heave[0]), float(pitch[0])
        except Exception as e:
            logger.error(f"check_edge_blocked error: {e}")
            return True, 0.0, 0.0, 0.0

    def find_shortest_path(
        self,
        start: Tuple[float,float],
        end:   Tuple[float,float],
        use_model: bool=False,
        ship_speed: float=8,
        condition: int=1
    )->Tuple[List[dict], float]:
        """
        Cari path terpendek. 
        - use_model=True => block edge berdasarkan wave_data + ML 
        - GPU/multithread dipakai jika TF mendukung
        - load/save dijkstra result
        """
        if not self.wave_data_locator:
            raise ValueError("WaveDataLocator belum diinisialisasi.")

        # Coba load dijkstra result
        cached = self.load_dijkstra_result(start, end, use_model, ship_speed, condition)
        if cached:
            logger.info("Menggunakan dijkstra cache result from local JSON.")
            return cached

        # Proses path
        wave_data_id = self._get_wave_data_identifier()
        start_idx = self.grid_locator.find_nearest_node(*start)
        end_idx   = self.grid_locator.find_nearest_node(*end)

        if start_idx < 0 or end_idx < 0:
            logger.error("Invalid start or end index.")
            return [], 0.0

        if use_model:
            # copy graph
            gcopy = self.igraph_graph.copy()
            edges_data = []
            for e in gcopy.es:
                s = gcopy.vs[e.source]
                t = gcopy.vs[e.target]
                edges_data.append({
                    "edge_id": e.index,
                    "source_coords": (s["lon"], s["lat"]),
                    "target_coords": (t["lon"], t["lat"]),
                    "ship_speed": ship_speed,
                    "condition": condition
                })

            logger.info(f"Batch process edges => wave_data_id={wave_data_id} ...")
            edge_res = self._batch_process_edges(edges_data, wave_data_id)
            # update weight
            for ed, er in zip(edges_data, edge_res):
                if er["blocked"]:
                    gcopy.es[ed["edge_id"]]["weight"] = float('inf')
                gcopy.es[ed["edge_id"]]["roll"]  = float(er["roll"])
                gcopy.es[ed["edge_id"]]["heave"] = float(er["heave"])
                gcopy.es[ed["edge_id"]]["pitch"] = float(er["pitch"])

            path_ids = gcopy.get_shortest_paths(v=start_idx, to=end_idx, weights="weight")[0]
            if not path_ids:
                logger.warning("No path found with model.")
                return [], 0.0
            dist = float(gcopy.distances(source=start_idx, target=end_idx, weights="weight")[0][0])

            # Bangun path
            path_data = []
            for i,node_i in enumerate(path_ids):
                vx = gcopy.vs[node_i]
                coords = (vx["lon"], vx["lat"])
                wave_data = {}
                try:
                    wave_data = self.wave_data_locator.get_wave_data(coords)
                except Exception as e:
                    logger.warning(f"WaveData error: {e}")

                roll=heave=pitch=0.0
                heading=0.0
                if i < len(path_ids)-1:
                    nxt= gcopy.vs[path_ids[i+1]]
                    eid= gcopy.get_eid(node_i, nxt.index, error=False)
                    if eid != -1:
                        roll = float(gcopy.es[eid].get("roll",0.0))
                        heave= float(gcopy.es[eid].get("heave",0.0))
                        pitch= float(gcopy.es[eid].get("pitch",0.0))
                    bearing= self._compute_bearing(coords,(nxt["lon"],nxt["lat"]))
                    heading= self._compute_heading(bearing, float(wave_data.get("dirpwsfc",0.0)))

                path_data.append({
                    "node_id": vx["name"],
                    "coordinates": [coords[0], coords[1]],
                    "htsgwsfc": float(wave_data.get("htsgwsfc",0.0)),
                    "perpwsfc": float(wave_data.get("perpwsfc",0.0)),
                    "dirpwsfc": float(wave_data.get("dirpwsfc",0.0)),
                    "Roll":  roll,
                    "Heave": heave,
                    "Pitch": pitch,
                    "rel_heading": heading
                })
            self.save_dijkstra_result(start, end, use_model, ship_speed, condition, path_data, dist)
            return path_data, dist
        else:
            # Dijkstra standar
            path_ids = self.igraph_graph.get_shortest_paths(
                v=start_idx, to=end_idx, weights="weight"
            )[0]
            if not path_ids:
                logger.warning("No path found (no-model).")
                return [], 0.0
            dist = float(self.igraph_graph.distances(source=start_idx, target=end_idx, weights="weight")[0][0])

            path_data=[]
            for i,node_i in enumerate(path_ids):
                vx = self.igraph_graph.vs[node_i]
                coords = (vx["lon"], vx["lat"])
                wave_data={}
                try:
                    wave_data = self.wave_data_locator.get_wave_data(coords)
                except Exception as e:
                    logger.warning(f"WaveData error: {e}")

                roll=heave=pitch=0.0
                heading=0.0
                if i < len(path_ids)-1:
                    nxt = self.igraph_graph.vs[path_ids[i+1]]
                    bearing= self._compute_bearing(coords,(nxt["lon"],nxt["lat"]))
                    heading= self._compute_heading(bearing, float(wave_data.get("dirpwsfc",0.0)))
                    # check blocked => tetap panggil _check_edge_blocked tapi tdk mengubah weight
                    blocked, r, h, p = self._check_edge_blocked(
                        vx["name"], nxt["name"], ship_speed, condition
                    )
                    roll, heave, pitch = float(r), float(h), float(p)

                path_data.append({
                    "node_id": vx["name"],
                    "coordinates": [coords[0], coords[1]],
                    "htsgwsfc": float(wave_data.get("htsgwsfc",0.0)),
                    "perpwsfc": float(wave_data.get("perpwsfc",0.0)),
                    "dirpwsfc": float(wave_data.get("dirpwsfc",0.0)),
                    "Roll":  roll,
                    "Heave": heave,
                    "Pitch": pitch,
                    "rel_heading": heading
                })
            self.save_dijkstra_result(start, end, use_model, ship_speed, condition, path_data, dist)
            return path_data, dist
