import os
import json
import hashlib
import logging
import time
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
from concurrent.futures import ThreadPoolExecutor

from utils import GridLocator  # Pastikan ini diimplementasikan
from constants import DATA_DIR, DATA_DIR_CACHE  # Sesuaikan path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_tf_for_production():
    """
    Opsional: Konfigurasi TensorFlow agar memanfaatkan GPU/multithread jika ada.
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
        logger.warning(f"TensorFlow config error: {e}. Using default single-thread CPU.")

class EdgeBatchCache:
    """
    Cache prediksi edge dalam 1 file .pkl per wave_data_id.
    Menyimpan blocked, roll, heave, pitch.
    """
    def __init__(
        self,
        cache_dir: str,
        batch_size: int = 100_000,
        max_memory_cache: int = 2_000_000,
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
            return
        # Flush wave_data_id lama
        if self.current_wave_data_id and self._dirty:
            self._flush_to_disk(self.current_wave_data_id)
        self.current_wave_data_id = wave_data_id
        self.memory_cache.clear()
        self._dirty = False

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
                except Exception as e:
                    logger.error(f"Error loading {pkl_file}: {e}")
        else:
            logger.info(f"No pkl cache for wave_data_id={wave_data_id}, start fresh.")

    def _flush_to_disk(self, wave_data_id: str):
        if not wave_data_id or not self._dirty:
            return
        pkl_file = os.path.join(self.cache_dir, f"batch_{wave_data_id}.pkl")
        lock_file = pkl_file + ".lock"
        with FileLock(lock_file):
            try:
                with open(pkl_file, 'wb') as f:
                    joblib.dump(self.memory_cache, f, compress=self.compression_level)
                logger.info(f"Flushed {len(self.memory_cache)} entries => {pkl_file}")
            except Exception as e:
                logger.error(f"Error flushing {pkl_file}: {e}")
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
    1) Membangun graph igraph
    2) Precompute blocked status => g.es["blocked"], roll,heave,pitch
    3) R-tree untuk bounding box
    4) get_blocked_edges_in_view => intersection R-tree
    5) Dijkstra caching load/save
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

        logger.info(f"Loading ML model from {self.model_path} ...")
        self.model = load_model(self.model_path, compile=False)
        logger.info(f"Loading input scaler from {self.input_scaler_pkl}...")
        self.input_scaler = joblib.load(self.input_scaler_pkl)
        logger.info(f"Loading output scaler from {self.output_scaler_pkl}...")
        self.output_scaler = joblib.load(self.output_scaler_pkl)

        self.dijkstra_cache_dir = os.path.join(DATA_DIR, "dijkstra")
        os.makedirs(self.dijkstra_cache_dir, exist_ok=True)

        self.edge_cache = EdgeBatchCache(
            os.path.join(DATA_DIR_CACHE, "edge_predictions"),
            batch_size=100_000,
            max_memory_cache=2_000_000,
            compression_level=0
        )

        # Pastikan atribut "blocked" ada
        if "blocked" not in self.igraph_graph.es.attributes():
            self.igraph_graph.es["blocked"] = [False]*self.igraph_graph.ecount()

        self.cache: Dict[str, Tuple[List[dict], float, List[List[dict]], List[dict]]] = {}
        # Bangun R-tree
        self.edge_spatial_index = index.Index()
        self._build_spatial_index_fast()

    def finalize(self):
        self.edge_cache.finalize()

    def update_wave_data_locator(self, wave_data_locator):
        self.finalize()
        self.wave_data_locator = wave_data_locator
        self.cache.clear()
        logger.info("WaveDataLocator updated, caches cleared.")

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
        logger.info(f"Building graph from JSON. nodes={len(data['nodes'])}, edges={len(data['edges'])}")

        g = ig.Graph(n=len(data["nodes"]), directed=False)

        node_ids = list(data["nodes"].keys())
        coords = np.array([data["nodes"][nid] for nid in node_ids])
        g.vs["name"] = node_ids
        g.vs["lon"] = coords[:, 0]
        g.vs["lat"] = coords[:, 1]

        node_map = {nid: i for i, nid in enumerate(node_ids)}
        edge_list = data["edges"]
        tuples, w_list, b_list = [],[],[]
        for ed in edge_list:
            s = node_map[ed["source"]]
            t = node_map[ed["target"]]
            w = float(ed.get("weight",1.0))
            b = float(ed.get("bearing",0.0))
            tuples.append((s,t))
            w_list.append(w)
            b_list.append(b)

        g.add_edges(tuples)
        g.es["weight"] = w_list
        g.es["bearing"] = b_list

        for attr in ["roll","heave","pitch"]:
            if attr not in g.es.attributes():
                g.es[attr] = [0.0]*g.ecount()
            else:
                g.es[attr] = [float(x) if x else 0.0 for x in g.es[attr]]

        g.write_pickle(self.saved_graph_file)
        logger.info("Graph built & pickled.")
        return g

    def _compute_bearing(self, start: Tuple[float,float], end: Tuple[float,float]) -> float:
        lon1, lat1 = np.radians(start)
        lon2, lat2 = np.radians(end)
        dlon = lon2 - lon1
        x = np.sin(dlon)*np.cos(lat2)
        y = np.cos(lat1)*np.sin(lat2) - (np.sin(lat1)*np.cos(lat2)*np.cos(dlon))
        ib = np.degrees(np.arctan2(x,y))
        return (ib+360)%360

    def _compute_heading(self, bearing: float, dirpwsfc: float) -> float:
        return (bearing - dirpwsfc + 90)%360

    def _predict_blocked(self, df: pd.DataFrame) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        colnames = ["ship_speed","wave_heading","wave_height","wave_period","condition"]
        df.columns = colnames
        scaled = self.input_scaler.transform(df)
        preds = self.model.predict(scaled, verbose=0)
        unscaled = self.output_scaler.inverse_transform(preds)

        roll = unscaled[:,0].astype(float)
        heave= unscaled[:,1].astype(float)
        pitch= unscaled[:,2].astype(float)
        blocked = (roll>=6)|(heave>=0.7)|(pitch>=3)
        return blocked, roll, heave, pitch

    def _batch_process_edges(self, edges_data: List[dict], wave_data_id: str) -> List[dict]:
        # Memanggil model batch -> blocked, roll, heave, pitch
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
            out=[]
            for b,r,h,p in zip(blocked,roll,heave,pitch):
                out.append({
                    "blocked": bool(b),
                    "roll": float(r),
                    "heave": float(h),
                    "pitch": float(p)
                })
            self.edge_cache.save_batch(out, buffer_edges, wave_data_id)
            buffer_inputs, buffer_edges = [], []
            return out

        for ed in edges_data:
            cached = self.edge_cache.get_cached_predictions(ed, wave_data_id)
            if cached:
                results.append(cached)
            else:
                wave_data = self.wave_data_locator.get_wave_data(ed["target_coords"])
                bearing = self._compute_bearing(ed["source_coords"], ed["target_coords"])
                heading = self._compute_heading(bearing, float(wave_data.get("dirpwsfc",0.0)))
                row = [
                    ed["ship_speed"],
                    heading,
                    float(wave_data.get("htsgwsfc",0.0)),
                    float(wave_data.get("perpwsfc",0.0)),
                    ed["condition"]
                ]
                buffer_inputs.append(row)
                buffer_edges.append(ed)

                if len(buffer_inputs)>=chunk_size:
                    chunk_out = flush_chunk()
                    results.extend(chunk_out)

        if buffer_inputs:
            chunk_out = flush_chunk()
            results.extend(chunk_out)

        self.edge_cache._flush_to_disk(wave_data_id)
        return results

    def _build_spatial_index_fast(self):
        """
        Membangun R-tree bounding box edge memakai multithread.
        Supaya cepat untuk 14 juta edge.
        """
        start_time = time.time()
        logger.info("🚀 Building R-tree index (multithread) ...")

        p = index.Property()
        p.dimension=2
        p.leaf_capacity=50_000
        p.fill_factor=0.9
        self.edge_spatial_index = index.Index(properties=p)

        g = self.igraph_graph
        edges = list(range(g.ecount()))
        total_e = len(edges)
        logger.info(f"Total edges = {total_e}")

        BATCH_SIZE = 200_000
        batches = [edges[i:i+BATCH_SIZE] for i in range(0,total_e,BATCH_SIZE)]
        logger.info(f"Split into {len(batches)} batches, size~{BATCH_SIZE}")

        def process_batch(batch_idx):
            # Return (indices, boxes)
            idxes=[]
            boxes=[]
            for eid in batch_idx:
                e = g.es[eid]
                s = g.vs[e.source]
                t = g.vs[e.target]
                min_lon = min(s["lon"], t["lon"])
                min_lat = min(s["lat"], t["lat"])
                max_lon = max(s["lon"], t["lon"])
                max_lat = max(s["lat"], t["lat"])
                idxes.append(eid)
                boxes.append((min_lon, min_lat, max_lon, max_lat))
            return idxes, boxes

        # Multithread
        prep_start = time.time()
        all_indices = []
        all_boxes = []
        with ThreadPoolExecutor() as exe:
            results = list(exe.map(process_batch, batches))
        for (i2,b2) in results:
            all_indices.extend(i2)
            all_boxes.extend(b2)
        logger.info(f"✅ Prepared bounding boxes in {time.time()-prep_start:.2f}s, total={len(all_indices)}")

        logger.info("Bulk inserting R-tree ...")
        insert_start = time.time()
        for idx, box in zip(all_indices, all_boxes):
            self.edge_spatial_index.insert(idx, box)
        logger.info(f"✅ Insert done in {time.time()-insert_start:.2f}s")

        logger.info(f"Total build R-tree => {time.time()-start_time:.2f}s")

    # ---------- Precompute blocked status: dipanggil saat wave_data berganti
    def update_blocked_status_for_all_edges(self, ship_speed: float, condition: int):
        """
        Satu kali untuk precompute blocked => g.es["blocked"], g.es["roll"], etc.
        """
        wave_data_id = self._get_wave_data_identifier()
        g = self.igraph_graph
        total = g.ecount()
        logger.info(f"Precomputing blocked for {total} edges ...")

        BATCH_SZ = 200_000
        e_idx = list(range(total))

        # simpan di array
        arr_blocked = [False]*total
        arr_roll    = [0.0]*total
        arr_heave   = [0.0]*total
        arr_pitch   = [0.0]*total

        def do_batch(batch_ids):
            # build edges_data
            edges_data = []
            for eid in batch_ids:
                e = g.es[eid]
                s = g.vs[e.source]
                t = g.vs[e.target]
                edges_data.append({
                    "edge_id": eid,
                    "source_coords": (s["lon"], s["lat"]),
                    "target_coords": (t["lon"], t["lat"]),
                    "ship_speed": ship_speed,
                    "condition": condition
                })
            batch_res = self._batch_process_edges(edges_data, wave_data_id)
            return batch_res

        start = time.time()
        subbatches = [ e_idx[i:i+BATCH_SZ] for i in range(0,total,BATCH_SZ)]
        offset=0
        for i,bids in enumerate(subbatches):
            logger.info(f"Batch {i+1}/{len(subbatches)} => {len(bids)} edges")
            subres = do_batch(bids)
            for local_i, eid in enumerate(bids):
                rdict = subres[local_i]
                # blocked
                b = rdict["blocked"] or rdict["roll"]>=6 or rdict["heave"]>=0.7 or rdict["pitch"]>=3
                arr_blocked[eid]=b
                arr_roll[eid]= rdict["roll"]
                arr_heave[eid]=rdict["heave"]
                arr_pitch[eid]=rdict["pitch"]

        g.es["blocked"] = arr_blocked
        g.es["roll"]    = arr_roll
        g.es["heave"]   = arr_heave
        g.es["pitch"]   = arr_pitch

        logger.info(f"Done precompute block => {time.time()-start:.2f}s for {total} edges")

    # -------------- get_blocked_edges_in_view --------------
    def get_blocked_edges_in_view(
        self,
        view_bounds: Tuple[float,float,float,float],
        MAX_EDGES: int=100_000
    ) -> List[Dict[str,Any]]:
        """
        Query bounding box from R-tree => read g.es[eid]["blocked"]
        """
        min_lon, min_lat, max_lon, max_lat = view_bounds
        cands = list(self.edge_spatial_index.intersection(view_bounds))
        logger.info(f"Candidate edges => {len(cands)}")

        g = self.igraph_graph
        results=[]
        for eid in cands[:MAX_EDGES]:
            e = g.es[eid]
            s = g.vs[e.source]
            t = g.vs[e.target]
            results.append({
                "edge_id": eid,
                "source_coords": (s["lon"], s["lat"]),
                "target_coords": (t["lon"], t["lat"]),
                "isBlocked": bool(e["blocked"])
            })
        logger.info(f"Return edges => {len(results)}")
        return results

    # -------------- Caching Dijkstra --------------
    def save_dijkstra_result(
        self,
        start: Tuple[float,float],
        end: Tuple[float,float],
        use_model: bool,
        ship_speed: float,
        condition: int,
        path: List[dict],
        distance: float,
        partial_paths: List[List[dict]],
        all_edges: List[dict]
    ):
        wave_data_id = self._get_wave_data_identifier()
        cache_file = os.path.join(self.dijkstra_cache_dir, f"{wave_data_id}.json")
        lock_file = cache_file + ".lock"
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
            "all_edges": [],
            "timestamp": datetime.utcnow().isoformat()+"Z"
        }

        with FileLock(lock_file):
            if os.path.exists(cache_file):
                try:
                    with open(cache_file,'r') as f:
                        cache_json = json.load(f)
                except json.JSONDecodeError as e:
                    logger.error(f"Error load {cache_file}: {e}, reset.")
                    cache_json = {
                        "wave_data_id": wave_data_id,
                        "dijkstra_results": {
                            "with_model":[],
                            "without_model":[]
                        }
                    }
            else:
                cache_json = {
                    "wave_data_id": wave_data_id,
                    "dijkstra_results": {
                        "with_model":[],
                        "without_model":[]
                    }
                }

            cache_json["dijkstra_results"][category].append(data)
            with open(cache_file,'w') as f:
                json.dump(cache_json, f, indent=4)
            logger.info(f"Saved dijkstra => {cache_file} ({category})")

    def load_dijkstra_result(
        self,
        start: Tuple[float,float],
        end: Tuple[float,float],
        use_model: bool,
        ship_speed: float,
        condition: int
    ) -> Optional[Dict[str,Any]]:
        wave_data_id = self._get_wave_data_identifier()
        cache_file = os.path.join(self.dijkstra_cache_dir, f"{wave_data_id}.json")
        if not os.path.exists(cache_file):
            logger.info(f"No dijkstra cache => {cache_file}")
            return None
        lock_file = cache_file+".lock"
        query_str = json.dumps({
            "start": list(start),
            "end": list(end),
            "use_model": use_model,
            "ship_speed": ship_speed,
            "condition": condition,
            "wave_data_id": wave_data_id
        }, sort_keys=True)
        query_key=hashlib.sha256(query_str.encode()).hexdigest()

        with FileLock(lock_file):
            try:
                with open(cache_file,'r') as f:
                    cache_json = json.load(f)
            except:
                logger.error(f"Error reading {cache_file}, reset.")
                return None

        category = "with_model" if use_model else "without_model"
        items = cache_json.get("dijkstra_results",{}).get(category,[])
        for it in items:
            it_str = json.dumps({
                "start": it["start"],
                "end": it["end"],
                "use_model": it["use_model"],
                "ship_speed": it["ship_speed"],
                "condition": it["condition"],
                "wave_data_id": wave_data_id
            }, sort_keys=True)
            it_key=hashlib.sha256(it_str.encode()).hexdigest()
            if it_key==query_key:
                logger.info("Found dijkstra in local JSON.")
                return {
                    "path": it["path"],
                    "distance": it["distance"],
                    "partial_paths": it["partial_paths"],
                    "all_edges": it["all_edges"]
                }
        return None

    # -------------- find_shortest_path --------------
    def find_shortest_path(
        self,
        start: Tuple[float,float],
        end: Tuple[float,float],
        use_model: bool=False,
        ship_speed: float=8.0,
        condition: int=1
    ) -> Tuple[List[dict], float, List[List[dict]], List[dict]]:
        """
        if use_model => blocked => weight=inf
        else => normal
        """
        cached = self.load_dijkstra_result(start,end,use_model,ship_speed,condition)
        if cached:
            logger.info("Use cached dijkstra from local JSON.")
            return (
                cached["path"],
                cached["distance"],
                cached["partial_paths"],
                cached["all_edges"]
            )

        # Mesti jalankan dijkstra
        s_idx = self.grid_locator.find_nearest_node(*start)
        e_idx = self.grid_locator.find_nearest_node(*end)
        if s_idx<0 or e_idx<0:
            logger.error("Invalid start/end idx.")
            return [],0.0,[],[]

        gcopy = self.igraph_graph.copy()
        # if use_model => blocked => weight=inf
        if use_model:
            arr_blocked = np.array([
                (gcopy.es[i]["blocked"] or
                 gcopy.es[i]["roll"]>=6 or
                 gcopy.es[i]["heave"]>=0.7 or
                 gcopy.es[i]["pitch"]>=3)
                for i in range(gcopy.ecount())
            ], dtype=bool)
            arr_weight = np.array(gcopy.es["weight"], dtype=float)
            arr_weight[arr_blocked] = float('inf')
            gcopy.es["weight"] = arr_weight
        # else no change

        try:
            sp = gcopy.get_shortest_paths(s_idx,to=e_idx,weights="weight",output="vpath")[0]
            if not sp:
                logger.warning("No path found.")
                return [],0.0,[],[]
            dist = gcopy.shortest_paths(s_idx,e_idx,weights="weight")[0][0]
        except Exception as e:
            logger.error(f"Dijkstra error: {e}")
            return [],0.0,[],[]

        path_data=[]
        for i,node_i in enumerate(sp):
            vx = gcopy.vs[node_i]
            coords=(vx["lon"], vx["lat"])
            wave_data={}
            try:
                wave_data=self.wave_data_locator.get_wave_data(coords)
            except:
                pass

            roll=0.0;heave=0.0;pitch=0.0;rel_heading=0.0
            if i<len(sp)-1:
                eid = gcopy.get_eid(node_i, sp[i+1], directed=False, error=False)
                if eid!=-1:
                    roll= gcopy.es[eid]["roll"]
                    heave=gcopy.es[eid]["heave"]
                    pitch=gcopy.es[eid]["pitch"]
                    bearing= self._compute_bearing(coords,(gcopy.vs[sp[i+1]]["lon"],gcopy.vs[sp[i+1]]["lat"]))
                    rel_heading= self._compute_heading(bearing,float(wave_data.get("dirpwsfc",0.0)))

            path_data.append({
                "node_id": vx["name"],
                "coordinates": list(coords),
                "htsgwsfc": float(wave_data.get("htsgwsfc",0.0)),
                "perpwsfc": float(wave_data.get("perpwsfc",0.0)),
                "dirpwsfc": float(wave_data.get("dirpwsfc",0.0)),
                "Roll": roll,
                "Heave": heave,
                "Pitch": pitch,
                "rel_heading": rel_heading
            })

        # partial paths
        partial_paths=[]
        for i in range(1,len(sp)):
            sub=[]
            for node_id in sp[:i+1]:
                vx= gcopy.vs[node_id]
                coords= (vx["lon"],vx["lat"])
                wave={}
                try:
                    wave=self.wave_data_locator.get_wave_data(coords)
                except:
                    pass
                sub.append({
                    "node_id": vx["name"],
                    "coordinates": list(coords),
                    "htsgwsfc": float(wave.get("htsgwsfc",0.0)),
                    "perpwsfc": float(wave.get("perpwsfc",0.0)),
                    "dirpwsfc": float(wave.get("dirpwsfc",0.0))
                })
            partial_paths.append(sub)

        self.save_dijkstra_result(
            start, end, use_model, ship_speed, condition, path_data, dist, partial_paths, []
        )
        return path_data, dist, partial_paths, []
