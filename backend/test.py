import numpy as np
import igraph as ig
from keras.api.models import load_model

from sklearn.preprocessing import MinMaxScaler
from managers import get_wave_data_response_interpolate
import os
import json
from scipy.spatial import cKDTree

# Load the model and scalers
model_path = "model.h5"
model = load_model(model_path)

# Normalize the data
scaler_in = MinMaxScaler()
scaler_out = MinMaxScaler()

# Path to cache folder
CACHE_FOLDER = "./data/cache"

# Find the latest JSON file in the cache folder
def find_latest_wave_data():
    files = [f for f in os.listdir(CACHE_FOLDER) if f.endswith(".json")]
    if not files:
        raise FileNotFoundError("No JSON wave data files found in the cache folder.")
    latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(CACHE_FOLDER, x)))
    return os.path.join(CACHE_FOLDER, latest_file)

# Load wave data from the latest JSON file
def load_latest_wave_data():
    latest_file = find_latest_wave_data()
    print(f"Loading wave data from: {latest_file}")
    with open(latest_file, "r") as f:
        return json.load(f)

# Fetch wave data response and ensure cache is updated
def fetch_and_update_wave_data(args):
    response, status_code = get_wave_data_response_interpolate(args)
    if status_code == 200:
        print("Wave data updated successfully.")
    else:
        raise Exception(f"Failed to update wave data: {response.get('error', 'Unknown error')}")

# Load structured graph files
def load_graph_from_files(base_path="./GridNode", file_prefix="graph_part_", file_suffix="_structured.json"):
    edges = []
    weights = []
    nodes = set()
    bearings = []  # Store bearings for each edge

    for i in range(62):  # 62 files (graph_part_{0} to graph_part_{61})
        file_path = os.path.join(base_path, f"{file_prefix}{i}{file_suffix}")
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            graph_data = json.load(f)
            for node, data in graph_data["nodes"].items():
                lon, lat = data["lon"], data["lat"]
                nodes.add((lon, lat))
                for edge in data["edges"]:
                    target_node = tuple(map(float, edge["to"].split("_")))
                    edges.append(((lon, lat), target_node))
                    weights.append(edge["weight"])
                    bearings.append(edge["bearing"])  # Add bearing information

    return nodes, edges, weights, bearings

# Build igraph graph from nodes and edges
def build_igraph_from_data(nodes, edges, weights, bearings):
    g = ig.Graph(directed=True)

    # Add nodes
    node_list = list(nodes)
    g.add_vertices(len(node_list))
    g.vs["name"] = [f"{lon}_{lat}" for lon, lat in node_list]
    g.vs["coords"] = node_list  # Store coordinates for lookup

    # Add edges
    edge_list = [(node_list.index(source), node_list.index(target)) for source, target in edges]
    g.add_edges(edge_list)
    g.es["weight"] = weights
    g.es["bearing"] = bearings  # Add bearing as an edge attribute

    return g

# Find nearest node in the graph for a given coordinate
def find_nearest_node(graph, target_coords):
    node_coords = np.array(graph.vs["coords"])
    tree = cKDTree(node_coords)
    dist, idx = tree.query(target_coords)
    return idx

# Extract wave data for a specific node
def get_wave_data(node, wave_data):
    latitude, longitude = node
    for var, details in wave_data["variables"].items():
        latitudes = np.array(details["latitude"])
        longitudes = np.array(details["longitude"])
        if latitude in latitudes and longitude in longitudes:
            lat_idx = np.where(latitudes == latitude)[0][0]
            lon_idx = np.where(longitudes == longitude)[0][0]
            if var == "dirpwsfc":
                wave_direction = details["data"][lat_idx][lon_idx]
            elif var == "htsgwsfc":
                wave_height = details["data"][lat_idx][lon_idx]
            elif var == "perpwsfc":
                wave_period = details["data"][lat_idx][lon_idx]
    return wave_direction, wave_height, wave_period

# Predict node status
def predict_node_status(model, scaler_in, scaler_out, ship_speed, wave_heading, wave_height, wave_period, condition):
    new_input = np.array([[ship_speed, wave_heading, wave_height, wave_period, condition]])
    new_input_scaled = scaler_in.transform(new_input)
    predicted_output_scaled = model.predict(new_input_scaled)
    predicted_output = scaler_out.inverse_transform(predicted_output_scaled)
    roll, heave, pitch = predicted_output[0]
    return not (roll < 6 and pitch < 3 and heave < 0.7)

# Custom Dijkstra with wave conditions
def custom_dijkstra_with_wave_conditions(graph, start_coords, end_coords, wave_data, model, scaler_in, scaler_out, condition):
    start_node = find_nearest_node(graph, start_coords)
    end_node = find_nearest_node(graph, end_coords)
    blocked_nodes = set()

    def get_weight(edge):
        source_node = graph.vs[edge.source]["coords"]
        target_node = graph.vs[edge.target]["coords"]
        ship_heading = edge["bearing"]  # Use bearing from edge

        # Fetch wave conditions for the target node
        wave_direction, wave_height, wave_period = get_wave_data(target_node, wave_data)

        # Calculate relative heading
        relative_heading = (wave_direction - ship_heading) % 360
        if relative_heading > 180:
            relative_heading = 360 - relative_heading

        # Predict node status
        is_blocked = predict_node_status(model, scaler_in, scaler_out, 8, relative_heading, wave_height, wave_period, condition)
        if is_blocked:
            blocked_nodes.add(target_node)
            return float("inf")

        return edge["weight"]

    # Update weights dynamically
    graph.es["weight"] = [get_weight(edge) for edge in graph.es]

    # Run Dijkstra
    path = graph.get_shortest_paths(start_node, to=end_node, weights="weight", output="vpath")
    return path[0], blocked_nodes

# Main function
def main():
    # Fetch and update wave data
    args = {"min_lat": -10.0, "max_lat": 10.0, "min_lon": 100.0, "max_lon": 120.0, "interpolate": "true", "use_kdtree": "false"}
    print("Fetching and updating wave data...")
    fetch_and_update_wave_data(args)

    # Load graph data
    print("Loading graph data...")
    nodes, edges, weights, bearings = load_graph_from_files()

    # Build igraph graph
    print("Building graph...")
    graph = build_igraph_from_data(nodes, edges, weights, bearings)

    # Load wave data from the latest JSON file
    print("Loading wave data...")
    wave_data = load_latest_wave_data()

    # Example input coordinates
    start_coords = (100.0, -5.0)  # Replace with actual starting coordinates
    end_coords = (110.0, -6.0)    # Replace with actual ending coordinates
    condition = 1

    # Run custom Dijkstra
    print("Running Dijkstra...")
    path, blocked_nodes = custom_dijkstra_with_wave_conditions(graph, start_coords, end_coords, wave_data, model, scaler_in, scaler_out, condition)

    if path:
        print("Shortest Path:")
        print(" -> ".join(graph.vs[node]["name"] for node in path))
    else:
        print("No path found.")

    print("Blocked Nodes:", blocked_nodes)

if __name__ == "__main__":
    main()
