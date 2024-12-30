import json
import math
import os
import gzip
import logging
import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon, shape
from shapely.prepared import prep
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque, defaultdict
from scipy.spatial import KDTree

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class UnionFind:
    """Union-Find data structure to manage connected components."""
    def __init__(self):
        self.parent = {}

    def find(self, x):
        """Find the root of x with path compression."""
        if x not in self.parent:
            self.add(x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """Union the sets containing x and y."""
        if x not in self.parent:
            self.add(x)
        if y not in self.parent:
            self.add(y)
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_y] = root_x

    def add(self, x):
        """Add element x to the structure."""
        if x not in self.parent:
            self.parent[x] = x

class RegionGraphGenerator:
    def __init__(self, config):
        self.config = config
        self.prepared_geometry = None
        self.grid = None  # 2D grid (numpy array)
        self.edges = []
        self.node_indices = {}  # Map valid nodes to their index_key
        self.max_nodes = config.get("MAX_NODES", 100000)
        self.adjacency = defaultdict(set)  # Adjacency list for connectivity

    def validate_config(self):
        required_keys = ["GEOJSON_FILE", "GRID_SPACING", "INDONESIA_EXTENT", "OUTPUT_FILE"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        if not os.path.exists(self.config["GEOJSON_FILE"]):
            raise FileNotFoundError(f"GeoJSON file not found: {self.config['GEOJSON_FILE']}")
        logger.info("Configuration validated successfully.")

    def load_geojson(self):
        logger.info("Loading GeoJSON file...")
        with open(self.config["GEOJSON_FILE"], "r") as f:
            geojson = json.load(f)

        polygons = []
        for feature in geojson["features"]:
            geometry = shape(feature["geometry"])
            if geometry.is_valid:
                if isinstance(geometry, Polygon):
                    polygons.append(geometry)
                elif isinstance(geometry, MultiPolygon):
                    polygons.extend(geometry.geoms)

        if not polygons:
            raise ValueError("No valid polygons found in the GeoJSON.")

        self.prepared_geometry = prep(MultiPolygon(polygons))
        logger.info(f"Loaded GeoJSON with {len(polygons)} valid polygons.")

    def generate_grid(self):
        logger.info("Generating 2D grid of nodes (30% from the right of the map)...")
        grid_spacing = self.config["GRID_SPACING"]
        lon_min, lat_min, lon_max, lat_max = self.config["INDONESIA_EXTENT"]

        # Calculate the starting longitude for the 30% right portion
        right_start_lon = lon_min + 0.5 * (lon_max - lon_min)

        lon_range = np.round(np.arange(right_start_lon, lon_max, grid_spacing), 6)
        lat_range = np.round(np.arange(lat_min, lat_max, grid_spacing), 6)

        self.grid = np.full((len(lat_range), len(lon_range)), None, dtype=object)
        node_count = 0

        for i, lon in enumerate(lon_range):
            for j, lat in enumerate(lat_range):
                if node_count >= self.max_nodes:
                    logger.info(f"Reached maximum node limit: {self.max_nodes}")
                    return
                point = Point(lon, lat)
                if self.prepared_geometry.intersects(point):
                    self.grid[j, i] = (round(lon, 6), round(lat, 6))  # Round coordinates
                    self.node_indices[f"{round(lon, 6)}_{round(lat, 6)}"] = (j, i)
                    node_count += 1

        logger.info(f"Generated grid with {node_count} valid nodes from the right 30% of the map.")

    def process_edges(self):
        logger.info("Processing edges using direct array indexing...")
        directions = [
            (-1, -1), (0, -1), (1, -1),
            (-1, 0),         (1, 0),
            (-1, 1),  (0, 1),  (1, 1)
        ]

        rows, cols = self.grid.shape
        for j in tqdm(range(rows), desc="🔗 Processing Edges", dynamic_ncols=True):
            for i in range(cols):
                if self.grid[j, i] is None:
                    continue
                lon, lat = self.grid[j, i]
                index_key = f"{lon}_{lat}"

                for dx, dy in directions:
                    for step in range(1, 10):
                        nj, ni = j + dy * step, i + dx * step
                        if 0 <= nj < rows and 0 <= ni < cols and self.grid[nj, ni] is not None:
                            n_lon, n_lat = self.grid[nj, ni]
                            neighbor_key = f"{n_lon}_{n_lat}"
                            distance = self.haversine_distance((lon, lat), (n_lon, n_lat))
                            bearing = self.calculate_bearing((lon, lat), (n_lon, n_lat))

                            # Round values to reduce floating point precision issues
                            distance = round(distance, 6)
                            bearing = round(bearing, 6)

                            self.edges.append({
                                "source": index_key,
                                "target": neighbor_key,
                                "weight": distance,
                                "bearing": bearing
                            })
                            self.adjacency[index_key].add(neighbor_key)
                            self.adjacency[neighbor_key].add(index_key)
                            break

        logger.info(f"Generated {len(self.edges)} edges.")

    def connect_disconnected_nodes(self):
        logger.info("Connecting disconnected nodes...")
        all_nodes = set(self.node_indices.keys())
        connected_nodes = set(self.adjacency.keys())
        disconnected_nodes = all_nodes - connected_nodes

        if not disconnected_nodes:
            logger.info("No disconnected nodes found.")
            return

        logger.info(f"Found {len(disconnected_nodes)} disconnected nodes.")

        # Create KD-Tree from connected nodes
        if connected_nodes:
            connected_coords = [self.grid[self.node_indices[node]] for node in connected_nodes]
            connected_coords_array = np.array(connected_coords)
            kd_tree = KDTree(connected_coords_array)
            connected_node_list = list(connected_nodes)
        else:
            logger.warning("No connected nodes available to connect to.")
            return

        for node_key in tqdm(disconnected_nodes, desc="🔗 Connecting Disconnected Nodes"):
            j, i = self.node_indices[node_key]
            current_coords = self.grid[j, i]

            # Find nearest connected node
            distance, index = kd_tree.query([current_coords], k=1)
            nearest_node_key = connected_node_list[index[0]]
            distance = distance[0]

            # Calculate bearing
            nearest_coords = self.grid[self.node_indices[nearest_node_key]]
            bearing = self.calculate_bearing(current_coords, nearest_coords)

            # Add edge
            self.edges.append({
                "source": node_key,
                "target": nearest_node_key,
                "weight": round(distance, 6),
                "bearing": round(bearing, 6)
            })

            # Update adjacency
            self.adjacency[node_key].add(nearest_node_key)
            self.adjacency[nearest_node_key].add(node_key)

    def check_full_connectivity(self):
        logger.info("Checking if the graph is fully connected...")
        if not self.adjacency:
            logger.warning("Adjacency list is empty. No edges to check connectivity.")
            return False

        all_nodes = set(self.node_indices.keys())
        start_node = next(iter(all_nodes))
        visited = set()
        queue = deque([start_node])

        while queue:
            current = queue.popleft()
            if current not in visited:
                visited.add(current)
                queue.extend(self.adjacency[current] - visited)

        is_connected = len(visited) == len(all_nodes)
        logger.info(f"Graph connectivity check: {'Connected' if is_connected else 'NOT connected'}")
        return is_connected

    def ensure_full_connectivity(self):
        logger.info("Ensuring the graph is fully connected...")

        # First check current connectivity
        if self.check_full_connectivity():
            logger.info("Graph is already fully connected.")
            return

        # Initialize UnionFind with all nodes
        uf = UnionFind()
        for node in self.node_indices.keys():
            uf.add(node)

        # Add existing connections to UnionFind
        for edge in self.edges:
            uf.union(edge["source"], edge["target"])

        # Identify components
        components = defaultdict(set)
        for node in self.node_indices.keys():
            root = uf.find(node)
            components[root].add(node)

        # Convert to list of components
        component_list = list(components.values())

        if len(component_list) <= 1:
            logger.info("Only one component found after analysis.")
            return

        logger.info(f"Found {len(component_list)} components to connect.")

        # Connect components
        main_component = component_list[0]
        for other_component in component_list[1:]:
            # Find closest pair between components
            min_distance = float('inf')
            best_pair = None

            # Use node coordinates directly from grid
            main_coords = [self.grid[self.node_indices[n]] for n in main_component]
            other_coords = [self.grid[self.node_indices[n]] for n in other_component]

            main_array = np.array(main_coords)
            other_array = np.array(other_coords)

            # Use KDTree for efficient nearest neighbor search
            tree = KDTree(main_array)
            distances, indices = tree.query(other_array, k=1)

            # Find the minimum distance pair
            min_idx = np.argmin(distances)
            min_distance = distances[min_idx]

            node_a = list(other_component)[min_idx]
            node_b = list(main_component)[indices[min_idx]]

            # Add new edge
            bearing = self.calculate_bearing(
                self.grid[self.node_indices[node_a]],
                self.grid[self.node_indices[node_b]]
            )

            self.edges.append({
                "source": node_a,
                "target": node_b,
                "weight": round(min_distance, 6),
                "bearing": round(bearing, 6)
            })

            # Update adjacency
            self.adjacency[node_a].add(node_b)
            self.adjacency[node_b].add(node_a)

            # Merge component into main component
            main_component.update(other_component)

        # Verify final connectivity
        if not self.check_full_connectivity():
            logger.error("Failed to achieve full connectivity after component connection.")
        else:
            logger.info("Successfully connected all components.")

    @staticmethod
    def haversine_distance(coords1, coords2):
        lon1, lat1 = coords1
        lon2, lat2 = coords2
        R = 6371  # Earth's radius in kilometers

        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        # Haversine formula
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))

        return round(R * c, 6)  # Round to 6 decimal places

    @staticmethod
    def calculate_bearing(coords1, coords2):
        lon1, lat1 = map(math.radians, coords1)
        lon2, lat2 = map(math.radians, coords2)

        dlon = lon2 - lon1

        y = math.sin(dlon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)

        initial_bearing = math.atan2(y, x)
        initial_bearing = math.degrees(initial_bearing)
        bearing = (initial_bearing + 360) % 360

        return round(bearing, 6)

    def visualize_graph(self):
        logger.info("Visualizing the graph...")
        plt.figure(figsize=(12, 12))
        plt.title("Region Graph Visualization")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

        # Plot edges
        for edge in self.edges:
            source = self.grid[self.node_indices[edge["source"]]]
            target = self.grid[self.node_indices[edge["target"]]]
            plt.plot([source[0], target[0]], [source[1], target[1]],
                    color="blue", alpha=0.5, linewidth=0.5)

        # Plot nodes
        node_coords = [self.grid[j, i] for j, i in self.node_indices.values()]
        node_lons, node_lats = zip(*node_coords)
        plt.scatter(node_lons, node_lats, color="red", s=10, alpha=0.7)

        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()

    def save_graph(self):
        logger.info("Saving the generated graph...")
        output_file = self.config["OUTPUT_FILE"]

        # Prepare nodes dictionary with rounded coordinates
        valid_nodes = {
            key: [round(coord[0], 6), round(coord[1], 6)]
            for key, (j, i) in self.node_indices.items()
            for coord in [self.grid[j, i]]
        }

        # Round all numeric values in edges
        rounded_edges = []
        for edge in self.edges:
            rounded_edge = {
                "source": edge["source"],
                "target": edge["target"],
                "weight": round(edge["weight"], 6),
                "bearing": round(edge["bearing"], 6)
            }
            rounded_edges.append(rounded_edge)

        graph_data = {
            "nodes": valid_nodes,
            "edges": rounded_edges
        }

        # Determine if output should be gzipped based on file extension
        if output_file.endswith(".gz"):
            with gzip.open(output_file, 'wt', encoding='utf-8') as f:
                json.dump(graph_data, f)
            logger.info(f"Graph saved successfully to {output_file} (gzipped).")
        else:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f)
            logger.info(f"Graph saved successfully to {output_file}.")

    def run(self):
        """Execute the graph generation process."""
        self.validate_config()
        self.load_geojson()
        self.generate_grid()
        self.process_edges()
        self.connect_disconnected_nodes()
        self.ensure_full_connectivity()
        self.save_graph()
        # Optionally visualize the graph
        if self.config.get("VISUALIZE", False):
            self.visualize_graph()

# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        "GEOJSON_FILE": "./drive/MyDrive/eez.json",  # Path to your GeoJSON file
        "GRID_SPACING": 1/111.32,  # Degrees
        "INDONESIA_EXTENT": [92.0, -15.0, 141.0, 10.0],  # [lon_min, lat_min, lon_max, lat_max]
        "OUTPUT_FILE": "./drive/MyDrive/region_graph_50.json",  # Output file path
        "MAX_NODES": 5000000,  # Optional: maximum number of nodes
        "VISUALIZE": False  # Set to True to visualize the graph
    }

    generator = RegionGraphGenerator(config)
    generator.run()
