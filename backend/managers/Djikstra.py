import os
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from keras.api.models import load_model
import joblib
from typing import List, Tuple, Dict
import igraph as ig
import logging
import hashlib

logger = logging.getLogger(__name__)

class GridLocator:
    def __init__(self, graph_coords: np.ndarray):
        """
        Initializes a GridLocator using cKDTree for efficient nearest neighbor search.
        """
        self.graph_coords = graph_coords
        self.kdtree = cKDTree(self.graph_coords)
        logger.info("GridLocator diinisialisasi dengan KDTree.")

    def find_nearest_node(self, lon: float, lat: float) -> int:
        """
        Finds the nearest node index in the graph to the given longitude and latitude.
        """
        distance, idx = self.kdtree.query([lon, lat])
        logger.info(f"Nearest node untuk koordinat ({lon}, {lat}) adalah index {idx} dengan jarak {distance}.")
        return idx

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
        Initializes the RouteOptimizer with required configurations.

        :param graph_file: Path to the graph JSON file
        :param wave_data_locator: Instance of WaveDataLocator
        :param model_path: Path to the .h5 machine learning model
        :param input_scaler_pkl: Path to the MinMaxScaler for inputs
        :param output_scaler_pkl: Path to the MinMaxScaler for outputs
        :param grid_locator: Instance of GridLocator
        """
        self.graph_file = graph_file
        self.wave_data_locator = wave_data_locator
        self.model_path = model_path
        self.input_scaler_pkl = input_scaler_pkl
        self.output_scaler_pkl = output_scaler_pkl
        self.grid_locator = grid_locator

        # Load graph
        self.graph = self._load_graph()

        # Initialize igraph Graph
        self.igraph_graph = self._build_igraph()

        # Existing initialization code...
        self.last_query = None  # Cache for the last query parameters
        self.last_result = None  # Cache for the last result

        # Load ML model and scalers
        print(f"Loading ML model from {self.model_path}...")
        self.model = load_model(self.model_path, compile=False)
        print(f"Loading input scaler from {self.input_scaler_pkl}...")
        self.input_scaler = joblib.load(self.input_scaler_pkl)
        print(f"Loading output scaler from {self.output_scaler_pkl}...")
        self.output_scaler = joblib.load(self.output_scaler_pkl)

    def _query_key(self, start, end, use_model, ship_speed, condition) -> str:
        """
        Generates a unique hash for the given query parameters.
        """
        wave_data_hash = hashlib.sha256(
            str([
                (key, tuple(map(float, value)))  # Convert numpy array to tuple
                for key, value in self.wave_data_locator.aggregated_wave_data.items()
            ]).encode()
        ).hexdigest()

        return f"{start}-{end}-{use_model}-{ship_speed}-{condition}-{wave_data_hash}"
    
    def update_wave_data_locator(self, wave_data_locator):
        """
        Updates the WaveDataLocator instance in RouteOptimizer.
        """
        self.wave_data_locator = wave_data_locator
        self.last_query = None

    def _load_graph(self) -> Dict:
        """
        Loads the graph JSON file.

        :return: Dictionary containing graph data
        """
        print(f"Loading graph from {self.graph_file}...")
        with open(self.graph_file, "r") as f:
            graph = json.load(f)
        print(f"Graph loaded successfully with {len(graph['nodes'])} nodes and {len(graph['edges'])} edges.\n")

        # **Tambahkan kode berikut untuk menampilkan sample nodes dan edges**
        print("Sample Nodes:")
        sample_nodes = list(graph["nodes"].items())[:5]  # Ambil 5 sample pertama
        for node_id, data in sample_nodes:
            print(f"Node ID: {node_id}")
            print(f"Data: {data}")
            print(f"Type of data: {type(data)}\n")

        print("Sample Edges:")
        sample_edges = graph["edges"][:5]  # Ambil 5 sample pertama
        for edge in sample_edges:
            print(edge)
            print(f"Type of edge: {type(edge)}\n")

        return graph

    def _build_igraph(self) -> ig.Graph:
        """
        Builds an igraph Graph from the loaded graph data.

        :return: igraph.Graph object
        """
        print("Building igraph graph...")
        g = ig.Graph(directed=False)
        node_id_map = {}  # Map node_id to igraph vertex index

        # Add vertices with coordinates as attributes
        for node_id, data in self.graph["nodes"].items():
            try:
                # Pastikan data adalah list dengan dua elemen [lon, lat]
                lon, lat = data
                g.add_vertex(name=node_id, lon=lon, lat=lat)
                node_id_map[node_id] = g.vcount() - 1
            except (IndexError, TypeError) as e:
                print(f"Error adding vertex for node {node_id}: {e}")

        # Add edges dengan weights dan bearings
        edge_tuples = []
        edge_weights = []
        edge_bearings = []
        for edge in self.graph["edges"]:
            source = edge.get("source")
            target = edge.get("target")
            weight = edge.get("weight")
            bearing = edge.get("bearing")  # Pastikan 'bearing' ada dalam edge data
            if source in node_id_map and target in node_id_map:
                edge_tuples.append((node_id_map[source], node_id_map[target]))
                edge_weights.append(weight)
                edge_bearings.append(bearing if bearing is not None else 0.0)  # Atur default jika tidak ada
            else:
                print(f"Skipping edge from {source} to {target} due to missing nodes.")

        g.add_edges(edge_tuples)
        g.es["weight"] = edge_weights
        g.es["bearing"] = edge_bearings
        print("igraph graph built successfully.\n")
        return g

    def _compute_bearing(self, start: Tuple[float, float], end: Tuple[float, float]) -> float:
        """
        Computes the bearing from start to end coordinates.

        :param start: (lon, lat)
        :param end: (lon, lat)
        :return: Bearing in degrees
        """
        lon1, lat1 = map(np.radians, start)
        lon2, lat2 = map(np.radians, end)

        dlon = lon2 - lon1

        x = np.sin(dlon) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(dlon))

        initial_bearing = np.degrees(np.arctan2(x, y))
        bearing = (initial_bearing + 360) % 360

        return bearing

    def _compute_heading(self, rel_heading: float, dirpwfsc: float) -> float:
        """
        Computes the adjusted bearing based on relative heading and dirpwfsc.

        :param rel_heading: Relative heading in degrees
        :param dirpwfsc: Direction of wave in degrees
        :return: Adjusted bearing in degrees
        """
        heading = (rel_heading - dirpwfsc) % 360
        return heading

    def _predict_blocked(self, inputs: pd.DataFrame) -> np.ndarray:
        """
        Predicts whether nodes are blocked based on the ML model.

        :param inputs: DataFrame containing batch input features for the model
        :return: Array of booleans indicating whether each input is blocked
        """
        # Pastikan DataFrame memiliki nama kolom yang sesuai
        feature_names = ['ship_speed', 'wave_heading', 'wave_height', 'wave_period', 'condition']
        inputs.columns = feature_names

        # Skala input
        scaled_inputs = self.input_scaler.transform(inputs)

        # Prediksi menggunakan model
        predictions = self.model.predict(scaled_inputs, verbose=0)

        # Balikkan skala output
        unscaled_predictions = self.output_scaler.inverse_transform(predictions)

        # Ambil nilai roll, heave, dan pitch
        roll = unscaled_predictions[:, 0]
        heave = unscaled_predictions[:, 1]
        pitch = unscaled_predictions[:, 2]

        # Kondisi untuk blokir (True jika melebihi threshold)
        blocked_mask = (roll >= 6) | (pitch >= 3) | (heave >= 0.7)

        return blocked_mask

    def find_shortest_path(self, start: Tuple[float, float], end: Tuple[float, float], use_model: bool = False, ship_speed = 8, condition = 1) -> Tuple[List[str], float]:
        """
        Finds the shortest path between start and end coordinates.
        
        Args:
            start: Starting coordinates (longitude, latitude)
            end: Ending coordinates (longitude, latitude)
            use_model: Whether to use ML model for path finding
            ship_speed: Speed of the ship
            condition: Condition value for the model
            
        Returns:
            Tuple containing path (list of node IDs) and total distance
        """
        query_key = self._query_key(start, end, use_model, ship_speed, condition)

        # Check cache - moved after all variable declarations
        if query_key == self.last_query and self.last_result is not None:
            print("Menggunakan hasil cache untuk query ini.")
            return self.last_result
        
        start_node = self.grid_locator.find_nearest_node(*start)
        end_node = self.grid_locator.find_nearest_node(*end)

        print(f"Start node: {self.igraph_graph.vs[start_node]['name']}, End node: {self.igraph_graph.vs[end_node]['name']}")

        if use_model:
            g = self.igraph_graph.copy()
            wave_data_cache = {}
            for v in g.vs:
                node_id = v["name"]
                coords = self.graph["nodes"][node_id]
                try:
                    wave_data_cache[node_id] = self.wave_data_locator.get_wave_data(tuple(coords))
                except ValueError:
                    wave_data_cache[node_id] = None

            edge_batches = []
            batch_edge_indices = []
            current_batch = []

            for edge in g.es:
                source = edge.source
                target = edge.target
                target_node_id = g.vs[target]["name"]

                if wave_data_cache[target_node_id] is not None:
                    wave_data = wave_data_cache[target_node_id]
                    dirpwfsc = wave_data["dirpwsfc"]

                    source_node_id = g.vs[source]["name"]
                    source_coords = self.graph["nodes"][source_node_id]
                    target_coords = self.graph["nodes"][target_node_id]
                    rel_heading = self._compute_bearing(tuple(source_coords), tuple(target_coords))
                    adjusted_bearing = self._compute_heading(rel_heading, dirpwfsc)

                    current_batch.append([ship_speed, adjusted_bearing, wave_data["htsgwsfc"], wave_data["perpwsfc"], condition])
                    batch_edge_indices.append(edge.index)

                    if len(current_batch) >= 1000:
                        edge_batches.append((current_batch.copy(), batch_edge_indices.copy()))
                        current_batch = []
                        batch_edge_indices = []

            if current_batch:
                edge_batches.append((current_batch, batch_edge_indices))

            for batch, indices in edge_batches:
                inputs_df = pd.DataFrame(batch, columns=["ship_speed", "wave_heading", "wave_height", "wave_period", "condition"])
                blocked_mask = self._predict_blocked(inputs_df)
                for idx, is_blocked in zip(indices, blocked_mask):
                    if is_blocked:
                        g.es[idx]["weight"] = float("inf")

            path_indices = g.get_shortest_paths(v=start_node, to=end_node, weights="weight", mode=ig.OUT, algorithm="dijkstra")[0]
            if not path_indices:
                result = ([], 0.0)
            else:
                total_weight = g.distances(source=start_node, target=end_node, weights="weight")[0][0]
                path = [self.igraph_graph.vs[idx]["name"] for idx in path_indices]
                result = (path, total_weight)
        else:
            path_indices = self.igraph_graph.get_shortest_paths(v=start_node, to=end_node, weights="weight", mode=ig.OUT, algorithm="dijkstra")[0]
            if not path_indices:
                result = ([], 0.0)
            else:
                total_weight = self.igraph_graph.distances(source=start_node, target=end_node, weights="weight")[0][0]
                path = [self.igraph_graph.vs[idx]["name"] for idx in path_indices]
                result = (path, total_weight)

        # Update cache with the new result
        self.last_query = query_key
        self.last_result = result

        return result
