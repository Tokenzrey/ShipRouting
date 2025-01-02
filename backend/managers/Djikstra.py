import glob
import os
import json
import hashlib
import logging
from datetime import datetime
from typing import List, Optional, Tuple, Dict
import multiprocessing
from multiprocessing import Manager, Pool
from array import array
import numpy as np
import pandas as pd
import igraph as ig
from filelock import FileLock
from keras.api.models import load_model
import joblib
from scipy.spatial import cKDTree

from utils import GridLocator  # Pastikan modul ini diimplementasikan dengan benar
from constants import DATA_DIR, DATA_DIR_CACHE  # Pastikan konstanta ini didefinisikan dengan benar

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EdgeBatchCache:
    """
    Class untuk mengelola cache batch edge predictions.
    """
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.batch_size = 1000
        self.memory_cache = {}
        self._load_existing_cache()

    def _generate_batch_key(self, wave_data_id: str, batch_idx: int) -> str:
        """Generate unique key untuk batch cache."""
        return f"batch_{wave_data_id}_{batch_idx}"

    def _load_existing_cache(self):
        """Load existing cache files into memory."""
        try:
            cache_files = glob.glob(os.path.join(self.cache_dir, "batch_*.json"))
            for cache_file in cache_files:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    self.memory_cache.update(cache_data)
            logger.info(f"Loaded {len(self.memory_cache)} cached edge predictions.")
        except Exception as e:
            logger.error(f"Error loading existing cache: {e}")

    def get_cached_predictions(self, edge_data: dict, wave_data_id: str) -> Optional[dict]:
        """Get cached predictions for an edge if available."""
        cache_key = self._generate_edge_key(edge_data, wave_data_id)
        return self.memory_cache.get(cache_key)

    def _generate_edge_key(self, edge_data: dict, wave_data_id: str) -> str:
        """Generate unique key untuk single edge prediction."""
        key_data = {
            "source": edge_data["source_coords"],
            "target": edge_data["target_coords"],
            "wave_data_id": wave_data_id,
            "ship_speed": edge_data["ship_speed"],
            "condition": edge_data["condition"]
        }
        return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()

    def save_batch(self, predictions: List[dict], edge_data: List[dict], wave_data_id: str):
        """Save batch predictions to cache."""
        batch_data = {}
        for pred, edge in zip(predictions, edge_data):
            edge_key = self._generate_edge_key(edge, wave_data_id)
            batch_data[edge_key] = pred

        # Save to memory cache
        self.memory_cache.update(batch_data)

        # Save to disk
        try:
            cache_file = os.path.join(self.cache_dir, f"batch_{wave_data_id}.json")
            with FileLock(f"{cache_file}.lock"):
                if os.path.exists(cache_file):
                    with open(cache_file, 'r') as f:
                        existing_data = json.load(f)
                    existing_data.update(batch_data)
                    batch_data = existing_data

                with open(cache_file, 'w') as f:
                    json.dump(batch_data, f)
            logger.info(f"Saved batch predictions to {cache_file}.")
        except Exception as e:
            logger.error(f"Error saving batch cache: {e}")

class RouteOptimizer:
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
        Inisialisasi RouteOptimizer dengan konfigurasi yang diperlukan.

        :param graph_file: Path ke file graph JSON
        :param wave_data_locator: Instance dari WaveDataLocator
        :param model_path: Path ke model machine learning (.h5)
        :param input_scaler_pkl: Path ke file pickle MinMaxScaler untuk input
        :param output_scaler_pkl: Path ke file pickle MinMaxScaler untuk output
        :param grid_locator: Instance dari GridLocator
        """
        self.graph_file = graph_file
        self.wave_data_locator = wave_data_locator
        self.model_path = model_path
        self.input_scaler_pkl = input_scaler_pkl
        self.output_scaler_pkl = output_scaler_pkl
        self.grid_locator = grid_locator

        self.saved_graph_file = "region_graph.pkl"

        # Load or build the igraph.Graph
        self.igraph_graph = self._load_or_build_graph()

        # Inisialisasi cache
        self.cache: Dict[str, Tuple[List[dict], float]] = {}

        # Memuat model ML dan scaler
        logger.info(f"Memuat model ML dari {self.model_path}...")
        self.model = load_model(self.model_path, compile=False)
        logger.info(f"Memuat input scaler dari {self.input_scaler_pkl}...")
        self.input_scaler = joblib.load(self.input_scaler_pkl)
        logger.info(f"Memuat output scaler dari {self.output_scaler_pkl}...")
        self.output_scaler = joblib.load(self.output_scaler_pkl)

        # Inisialisasi direktori cache Dijkstra
        self.dijkstra_cache_dir = os.path.join(DATA_DIR, "dijkstra")
        os.makedirs(self.dijkstra_cache_dir, exist_ok=True)

    def _get_wave_data_identifier(self) -> str:
        """
        Menghasilkan identifier unik untuk data gelombang saat ini berdasarkan isinya.
        """
        wave_data_file = self.wave_data_locator.current_wave_file
        wave_data_path = os.path.join(DATA_DIR_CACHE, wave_data_file)
        try:
            with open(wave_data_path, 'rb') as f:
                wave_data_content = f.read()
            wave_data_hash = hashlib.sha256(wave_data_content).hexdigest()
            return wave_data_hash
        except Exception as e:
            logger.error(f"Gagal menghasilkan identifier data gelombang: {e}")
            raise

    def _query_key(self, start, end, use_model, ship_speed, condition) -> str:
        """
        Menghasilkan hash unik untuk parameter query yang diberikan.

        :param start: Koordinat awal
        :param end: Koordinat akhir
        :param use_model: Apakah menggunakan model ML
        :param ship_speed: Kecepatan kapal
        :param condition: Nilai kondisi untuk prediksi model
        :return: Hash unik sebagai string
        """
        try:
            wave_data_hash = self._get_wave_data_identifier()
        except Exception as e:
            logger.error(f"Tidak dapat menghasilkan kunci query karena kegagalan identifier data gelombang: {e}")
            raise

        # Membuat representasi string unik dari parameter query
        query_string = json.dumps({
            "start": start,
            "end": end,
            "use_model": use_model,
            "ship_speed": ship_speed,
            "condition": condition,
            "wave_data_hash": wave_data_hash
        }, sort_keys=True)

        return hashlib.sha256(query_string.encode()).hexdigest()

    def save_dijkstra_result(self, start, end, use_model, ship_speed, condition, path, distance):
        """
        Menyimpan hasil Dijkstra ke cache berdasarkan data gelombang.

        :param start: Koordinat awal
        :param end: Koordinat akhir
        :param use_model: Apakah menggunakan model ML
        :param ship_speed: Kecepatan kapal
        :param condition: Nilai kondisi untuk prediksi model
        :param path: Path yang ditemukan
        :param distance: Total jarak/path weight
        """
        try:
            wave_data_id = self._get_wave_data_identifier()
        except Exception as e:
            logger.error(f"Tidak dapat menyimpan hasil Dijkstra karena kegagalan identifier data gelombang: {e}")
            return

        cache_filename = f"{wave_data_id}.json"
        cache_filepath = os.path.join(self.dijkstra_cache_dir, cache_filename)
        lock_filepath = f"{cache_filepath}.lock"

        # Menyiapkan data hasil
        result_data = {
            "start": list(start),
            "end": list(end),
            "use_model": use_model,
            "ship_speed": ship_speed,
            "condition": condition,
            "path": path,
            "distance": distance,
            "timestamp": datetime.utcnow().isoformat() + 'Z'
        }

        # Menggunakan file lock untuk mencegah penulisan simultan
        lock = FileLock(lock_filepath, timeout=10)
        try:
            with lock:
                if os.path.exists(cache_filepath):
                    # Memuat cache yang sudah ada
                    with open(cache_filepath, 'r') as f:
                        cache_content = json.load(f)
                else:
                    # Menginisialisasi struktur cache baru
                    cache_content = {
                        "wave_data_id": wave_data_id,
                        "dijkstra_results": []
                    }

                # Menambahkan hasil baru
                cache_content["dijkstra_results"].append(result_data)

                # Menyimpan kembali ke file JSON
                with open(cache_filepath, 'w') as f:
                    json.dump(cache_content, f, indent=4)
            logger.info(f"Hasil Dijkstra disimpan ke {cache_filepath}.")
        except Exception as e:
            logger.error(f"Gagal menyimpan hasil Dijkstra: {e}")

    def update_wave_data_locator(self, wave_data_locator):
        """
        Memperbarui instance WaveDataLocator dalam RouteOptimizer.

        :param wave_data_locator: Instance baru dari WaveDataLocator
        """
        self.wave_data_locator = wave_data_locator
        self.cache.clear()
        logger.info("WaveDataLocator telah diperbarui dalam RouteOptimizer dan cache dibersihkan.")

    def _load_or_build_graph(self) -> ig.Graph:
        """
        Load the graph from a saved pickle file if available.
        Otherwise, load from JSON, build the igraph.Graph, and save it.

        :return: An igraph.Graph object
        """
        if os.path.exists(self.saved_graph_file):
            logger.info(f"Loading graph from saved file: {self.saved_graph_file}...")
            try:
                g = ig.Graph.Read_Pickle(self.saved_graph_file)
                logger.info(f"Graph loaded from {self.saved_graph_file} with {g.vcount()} vertices and {g.ecount()} edges.")
                return g
            except Exception as e:
                logger.error(f"Failed to load graph from {self.saved_graph_file}: {e}")
                logger.info("Attempting to load graph from JSON and rebuild...")
        
        # If saved graph doesn't exist or failed to load, build from JSON
        logger.info(f"Loading graph from JSON file: {self.graph_file}...")
        try:
            with open(self.graph_file, "r") as f:
                graph_data = json.load(f)
            logger.info(f"Graph successfully loaded with {len(graph_data['nodes'])} nodes and {len(graph_data['edges'])} edges.\n")

            # **Debugging: Display sample nodes and edges**
            logger.debug("Sample Nodes:")
            sample_nodes = list(graph_data["nodes"].items())[:5]  # First 5 nodes
            for node_id, data in sample_nodes:
                logger.debug(f"Node ID: {node_id}")
                logger.debug(f"Data: {data}")
                logger.debug(f"Data Type: {type(data)}\n")

            logger.debug("Sample Edges:")
            sample_edges = graph_data["edges"][:5]  # First 5 edges
            for edge in sample_edges:
                logger.debug(edge)
                logger.debug(f"Edge Type: {type(edge)}\n")

            # Build igraph.Graph
            g = self._build_igraph(graph_data)

            # Save the built graph for future use
            logger.info(f"Saving built graph to {self.saved_graph_file}...")
            g.write_pickle(self.saved_graph_file)
            logger.info("Graph successfully saved.")

            return g
        except Exception as e:
            logger.error(f"Failed to load or build graph: {e}")
            raise

    def _build_igraph(self) -> ig.Graph:
        """
        Builds an optimized igraph Graph from the loaded graph data.
        """
        logger.info("Building igraph graph...")
        
        # Pre-allocate vertex array
        nodes = self.graph["nodes"]
        vertex_count = len(nodes)
        g = ig.Graph(n=vertex_count, directed=False)
        
        # Process vertices in bulk
        node_ids = list(nodes.keys())
        coords = np.array([nodes[nid] for nid in node_ids])
        
        # Set vertex attributes in bulk
        g.vs["name"] = node_ids
        g.vs["lon"] = coords[:, 0]
        g.vs["lat"] = coords[:, 1]
        
        # Create node ID mapping using dictionary comprehension
        node_id_map = {nid: idx for idx, nid in enumerate(node_ids)}
        
        # Process edges in bulk using numpy arrays
        edges = self.graph["edges"]
        edge_count = len(edges)
        
        # Pre-allocate arrays
        edge_tuples = np.empty((edge_count, 2), dtype=np.int32)
        edge_weights = np.empty(edge_count, dtype=np.float32)
        edge_bearings = np.empty(edge_count, dtype=np.float32)
        
        # Fill arrays in single pass
        for i, edge in enumerate(edges):
            edge_tuples[i] = [node_id_map[edge["source"]], node_id_map[edge["target"]]]
            edge_weights[i] = float(edge.get("weight", 1.0))
            edge_bearings[i] = float(edge.get("bearing", 0.0))
        
        # Add edges and attributes in bulk
        g.add_edges(edge_tuples)
        g.es["weight"] = edge_weights
        g.es["bearing"] = edge_bearings
        
        print(f"Graph built with {g.vcount()} vertices and {g.ecount()} edges.")
        return g

    def _compute_bearing(self, start: Tuple[float, float], end: Tuple[float, float]) -> float:
        """
        Menghitung bearing dari koordinat start ke koordinat end.

        :param start: (lon, lat)
        :param end: (lon, lat)
        :return: Bearing dalam derajat
        """
        lon1, lat1 = map(np.radians, start)
        lon2, lat2 = map(np.radians, end)

        dlon = lon2 - lon1

        x = np.sin(dlon) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(dlon))

        initial_bearing = np.degrees(np.arctan2(x, y))
        bearing = (initial_bearing + 360) % 360

        return bearing

    def _compute_heading(self, adjusted_bearing: float, dirpwfsc: float) -> float:
        """
        Menghitung heading relatif berdasarkan bearing yang disesuaikan dan arah gelombang.

        :param adjusted_bearing: Bearing dari source ke target node dalam derajat.
        :param dirpwfsc: Arah gelombang dalam derajat
        :return: Heading relatif dalam derajat
        """
        heading = (adjusted_bearing - dirpwfsc) % 360
        return heading

    def _predict_blocked(self, inputs: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Memprediksi apakah edges diblokir berdasarkan model ML dan mengembalikan metrik gelombang terkait.

        :param inputs: DataFrame yang berisi fitur input batch untuk model
        :return: Tuple yang berisi:
                 - Array boolean yang menunjukkan apakah setiap input diblokir
                 - Array nilai Roll
                 - Array nilai Heave
                 - Array nilai Pitch
        """
        # Memastikan DataFrame memiliki nama kolom yang benar
        feature_names = ['ship_speed', 'wave_heading', 'wave_height', 'wave_period', 'condition']
        inputs.columns = feature_names

        try:
            # Menyaring fitur input
            scaled_inputs = self.input_scaler.transform(inputs)

            # Melakukan prediksi menggunakan model
            predictions = self.model.predict(scaled_inputs, verbose=0)

            # Membalikkan skala prediksi untuk mendapatkan metrik gelombang sebenarnya
            unscaled_predictions = self.output_scaler.inverse_transform(predictions)

            # Mengekstrak Roll, Heave, dan Pitch dari prediksi
            roll = unscaled_predictions[:, 0]
            heave = unscaled_predictions[:, 1]
            pitch = unscaled_predictions[:, 2]

            # Mendefinisikan kondisi pemblokiran
            blocked_mask = (roll >= 6) | (pitch >= 3) | (heave >= 0.7)

            return blocked_mask, roll, heave, pitch
        except Exception as e:
            logger.error(f"Gagal memprediksi status blokir: {e}")
            raise

    def _check_edge_blocked(self, source_node_id: str, target_node_id: str, ship_speed: float, condition: int) -> Tuple[bool, float, float, float]:
        """
        Memeriksa apakah sebuah edge diblokir berdasarkan kondisi gelombang dan heading relatif.

        Args:
            source_node_id (str): ID node sumber
            target_node_id (str): ID node target
            ship_speed (float): Kecepatan kapal dalam knot
            condition (int): Nilai kondisi untuk prediksi model

        Returns:
            tuple: (is_blocked, roll, heave, pitch)
        """
        try:
            # Mendapatkan koordinat untuk node sumber dan target
            source_vertex = self.igraph_graph.vs.find(name=source_node_id)
            target_vertex = self.igraph_graph.vs.find(name=target_node_id)
            source_coords = (source_vertex["lon"], source_vertex["lat"])
            target_coords = (target_vertex["lon"], target_vertex["lat"])

            # Mendapatkan data gelombang untuk node target
            wave_data = self.wave_data_locator.get_wave_data(target_coords)

            # Menghitung bearing dan heading relatif
            adjusted_bearing = self._compute_bearing(source_coords, target_coords)
            rel_heading = self._compute_heading(
                adjusted_bearing,
                wave_data.get("dirpwsfc", 0.0)
            )

            # Menyiapkan fitur untuk prediksi model
            features = pd.DataFrame([[
                ship_speed,
                rel_heading,
                wave_data.get("htsgwsfc", 0.0),
                wave_data.get("perpwsfc", 0.0),
                condition
            ]], columns=["ship_speed", "wave_heading", "wave_height", "wave_period", "condition"])

            # Mendapatkan prediksi
            blocked_mask, roll, heave, pitch = self._predict_blocked(features)

            return blocked_mask[0], roll[0], heave[0], pitch[0]

        except Exception as e:
            logger.error(f"Gagal memeriksa status blokir edge dari {source_node_id} ke {target_node_id}: {e}")
            return True, 0.0, 0.0, 0.0  # Mengasumsikan edge diblokir jika terjadi error

    def _batch_process_edges(self, edges_data: List[dict], wave_data_id: str) -> List[dict]:
        """
        Process edges in batches with caching.
        """
        results = []
        batch_to_process = []
        batch_edge_data = []

        for edge_data in edges_data:
            # Check cache first
            cached_result = self.edge_cache.get_cached_predictions(edge_data, wave_data_id)
            if cached_result:
                results.append(cached_result)
            else:
                batch_to_process.append(edge_data)
                batch_edge_data.append(edge_data)

            # Process batch when it reaches batch size
            if len(batch_to_process) >= self.edge_cache.batch_size:
                batch_results = self._process_edge_batch(batch_to_process, wave_data_id)
                self.edge_cache.save_batch(batch_results, batch_edge_data, wave_data_id)
                results.extend(batch_results)
                batch_to_process = []
                batch_edge_data = []

        # Process remaining edges
        if batch_to_process:
            batch_results = self._process_edge_batch(batch_to_process, wave_data_id)
            self.edge_cache.save_batch(batch_results, batch_edge_data, wave_data_id)
            results.extend(batch_results)

        return results

    def _process_edge_batch(self, batch: List[dict], wave_data_id: str) -> List[dict]:
        """
        Process a single batch of edges.
        """
        features_list = []
        for edge_data in batch:
            source_coords = edge_data["source_coords"]
            target_coords = edge_data["target_coords"]

            wave_data = self.wave_data_locator.get_wave_data(target_coords)
            adjusted_bearing = self._compute_bearing(source_coords, target_coords)
            rel_heading = self._compute_heading(
                adjusted_bearing,
                wave_data.get("dirpwsfc", 0.0)
            )

            features_list.append([
                edge_data["ship_speed"],
                rel_heading,
                wave_data.get("htsgwsfc", 0.0),
                wave_data.get("perpwsfc", 0.0),
                edge_data["condition"]
            ])

        # Predict using model
        features_df = pd.DataFrame(
            features_list,
            columns=["ship_speed", "wave_heading", "wave_height", "wave_period", "condition"]
        )
        blocked_mask, roll, heave, pitch = self._predict_blocked(features_df)

        # Format results
        results = []
        for i in range(len(batch)):
            results.append({
                "blocked": bool(blocked_mask[i]),
                "roll": float(roll[i]),
                "heave": float(heave[i]),
                "pitch": float(pitch[i])
            })
        return results

    def find_shortest_path(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        use_model: bool = False,
        ship_speed: float = 8,
        condition: int = 1
    ) -> Tuple[List[dict], float]:
        """
        Menemukan path terpendek antara koordinat start dan end, dengan opsi menggunakan model ML
        untuk validasi path berdasarkan kondisi gelombang.

        Args:
            start (Tuple[float, float]): Koordinat awal (longitude, latitude)
            end (Tuple[float, float]): Koordinat akhir (longitude, latitude)
            use_model (bool): Apakah menggunakan model ML untuk validasi path
            ship_speed (float): Kecepatan kapal dalam knot
            condition (int): Nilai kondisi untuk prediksi model

        Returns:
            Tuple[List[dict], float]: Tuple yang berisi:
                - List dictionary yang berisi informasi node dalam path
                - Total jarak/path weight

        Raises:
            ValueError: Jika WaveDataLocator belum diinisialisasi
            RuntimeError: Jika pencarian path gagal
        """
        # Memvalidasi inisialisasi
        if self.wave_data_locator is None:
            raise ValueError("WaveDataLocator belum diinisialisasi")

        # Menghasilkan kunci cache untuk query ini
        try:
            query_key = self._query_key(start, end, use_model, ship_speed, condition)
        except Exception as e:
            logger.error(f"Gagal menghasilkan kunci cache: {e}")
            raise RuntimeError(f"Gagal menghasilkan kunci cache: {str(e)}")

        # Memeriksa cache in-memory
        if query_key in self.cache:
            logger.info("Menggunakan hasil cache untuk query ini")
            return self.cache[query_key]

        try:
            # Menemukan node terdekat untuk start dan end
            start_node = self.grid_locator.find_nearest_node(*start)
            end_node = self.grid_locator.find_nearest_node(*end)

            start_node_id = self.igraph_graph.vs[start_node]["name"]
            end_node_id = self.igraph_graph.vs[end_node]["name"]

            logger.info(f"Node Start: {start_node_id}, Node End: {end_node_id}")

            if use_model:
                # Membuat salinan graph untuk dimodifikasi
                g = self.igraph_graph.copy()

                # Prepare edge data for batch processing
                edges_data = []
                for edge in g.es:
                    source_vertex = g.vs[edge.source]
                    target_vertex = g.vs[edge.target]
                    edges_data.append({
                        "edge_id": edge.index,
                        "source_coords": (source_vertex["lon"], source_vertex["lat"]),
                        "target_coords": (target_vertex["lon"], target_vertex["lat"]),
                        "ship_speed": ship_speed,
                        "condition": condition
                    })

                # Get wave data identifier
                wave_data_id = self._get_wave_data_identifier()

                # Process edges in batches with caching
                logger.info("Memproses edge dalam batch dengan caching...")
                edge_results = self._batch_process_edges(edges_data, wave_data_id)

                # Update graph weights based on results
                for edge_data, result in zip(edges_data, edge_results):
                    edge_id = edge_data["edge_id"]
                    if result["blocked"]:
                        g.es[edge_id]["weight"] = float("inf")  # Mengatur weight menjadi inf jika diblokir
                    g.es[edge_id]["roll"] = result["roll"]
                    g.es[edge_id]["heave"] = result["heave"]
                    g.es[edge_id]["pitch"] = result["pitch"]

                # Menjalankan algoritma Dijkstra menggunakan weights yang telah dimodifikasi
                logger.info("Menjalankan algoritma Dijkstra...")
                path_indices = g.get_shortest_paths(
                    v=start_node,
                    to=end_node,
                    weights="weight",
                    mode=ig.OUT,
                    algorithm="dijkstra"
                )[0]

                if not path_indices:
                    logger.warning("Tidak ditemukan path yang valid")
                    result = ([], 0.0)
                else:
                    # Menghitung total weight/path distance
                    total_weight = g.distances(
                        source=start_node,
                        target=end_node,
                        weights="weight"
                    )[0][0]

                    # Membangun informasi path dengan metrik yang tepat
                    path = []
                    for i in range(len(path_indices)):
                        vertex = g.vs[path_indices[i]]
                        coords = (vertex["lon"], vertex["lat"])
                        node_id = vertex["name"]
                        try:
                            wave_data = self.wave_data_locator.get_wave_data(coords)
                        except Exception as e:
                            logger.warning(f"Gagal mendapatkan data gelombang untuk node {node_id}: {e}")
                            wave_data = {}

                        if i < len(path_indices) - 1:
                            next_vertex = g.vs[path_indices[i + 1]]
                            next_coords = (next_vertex["lon"], next_vertex["lat"])

                            adjusted_bearing = self._compute_bearing(coords, next_coords)
                            rel_heading = self._compute_heading(
                                adjusted_bearing,
                                wave_data.get("dirpwsfc", 0.0)
                            )

                            edge = g.get_eid(vertex.index, next_vertex.index, error=False)
                            if edge != -1:
                                roll = g.es[edge].get("roll", 0.0)
                                heave = g.es[edge].get("heave", 0.0)
                                pitch = g.es[edge].get("pitch", 0.0)
                            else:
                                roll, heave, pitch = 0.0, 0.0, 0.0  # Nilai default
                        else:
                            # Node terakhir dalam path
                            rel_heading = 0.0  # Atau nilai default lainnya
                            roll, heave, pitch = 0.0, 0.0, 0.0  # Nilai default

                        node_data = {
                            "node_id": node_id,
                            "coordinates": list(coords),
                            "htsgwsfc": float(wave_data.get("htsgwsfc", 0.0)),
                            "perpwsfc": float(wave_data.get("perpwsfc", 0.0)),
                            "dirpwfsfc": float(wave_data.get("dirpwsfc", 0.0)),
                            "rel_heading": float(rel_heading),
                            "Roll": float(roll),
                            "Heave": float(heave),
                            "Pitch": float(pitch)
                        }
                        path.append(node_data)

                    result = (path, total_weight)
            else:
                # Dijkstra standar tanpa model
                logger.info("Menjalankan algoritma Dijkstra standar...")
                try:
                    path_indices = self.igraph_graph.get_shortest_paths(
                        v=start_node,
                        to=end_node,
                        weights="weight",
                        mode=ig.OUT,
                        algorithm="dijkstra"
                    )[0]
                except Exception as e:
                    logger.error(f"Gagal menjalankan algoritma Dijkstra: {e}")
                    raise RuntimeError(f"Gagal menjalankan algoritma Dijkstra: {str(e)}")

                if not path_indices:
                    logger.warning("Tidak ditemukan path")
                    result = ([], 0.0)
                else:
                    try:
                        total_weight = self.igraph_graph.distances(
                            source=start_node,
                            target=end_node,
                            weights="weight"
                        )[0][0]
                    except Exception as e:
                        logger.error(f"Gagal menghitung total jarak: {e}")
                        raise RuntimeError(f"Gagal menghitung total jarak: {str(e)}")

                    # Membangun informasi path dengan metrik yang tepat
                    path = []
                    for i in range(len(path_indices)):
                        vertex = self.igraph_graph.vs[path_indices[i]]
                        coords = (vertex["lon"], vertex["lat"])
                        node_id = vertex["name"]
                        try:
                            wave_data = self.wave_data_locator.get_wave_data(coords)
                        except Exception as e:
                            logger.warning(f"Gagal mendapatkan data gelombang untuk node {node_id}: {e}")
                            wave_data = {}

                        if i < len(path_indices) - 1:
                            next_vertex = self.igraph_graph.vs[path_indices[i + 1]]
                            next_coords = (next_vertex["lon"], next_vertex["lat"])

                            adjusted_bearing = self._compute_bearing(coords, next_coords)
                            rel_heading = self._compute_heading(
                                adjusted_bearing,
                                wave_data.get("dirpwsfc", 0.0)
                            )

                            # Mendapatkan metrik gelombang dari edge menggunakan metode _check_edge_blocked
                            is_blocked, roll, heave, pitch = self._check_edge_blocked(
                                source_node_id=node_id,
                                target_node_id=next_vertex["name"],
                                ship_speed=ship_speed,
                                condition=condition
                            )

                            # Jika edge diblokir, kita masih ingin path yang ditemukan sebelumnya tetap valid,
                            # karena use_model=False. Namun, Roll, Heave, Pitch tetap diprediksi.
                            # Bisa juga diputuskan untuk logika lain tergantung kebutuhan.
                        else:
                            # Node terakhir dalam path
                            rel_heading = 0.0  # Atau nilai default lainnya
                            is_blocked, roll, heave, pitch = False, 0.0, 0.0, 0.0  # Nilai default

                        node_data = {
                            "node_id": node_id,
                            "coordinates": list(coords),
                            "htsgwsfc": float(wave_data.get("htsgwsfc", 0.0)),
                            "perpwsfc": float(wave_data.get("perpwsfc", 0.0)),
                            "dirpwfsfc": float(wave_data.get("dirpwsfc", 0.0)),
                            "rel_heading": float(rel_heading),
                            "Roll": float(roll),
                            "Heave": float(heave),
                            "Pitch": float(pitch)
                        }
                        path.append(node_data)

                    result = (path, total_weight)

            # Memperbarui cache dengan hasil
            self.cache[query_key] = result

            # Menyimpan ke cache persistensi jika path ditemukan dan menggunakan model
            if use_model and path_indices:
                try:
                    self.save_dijkstra_result(
                        start, end, use_model, ship_speed, condition,
                        result[0], result[1]
                    )
                except Exception as e:
                    logger.error(f"Gagal menyimpan ke cache persistensi: {e}")

            return result, 200
        except Exception as e:
            logger.error(f"Gagal: {e}")
            raise RuntimeError(f"Gagal: {str(e)}")