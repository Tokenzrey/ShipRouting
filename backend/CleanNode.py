import json
import os
from shapely.geometry import shape, MultiPolygon, Point, Polygon
from shapely.prepared import prep
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import logging
import math
from functools import lru_cache

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variable for main geometry
global_main_geometry = None

def init_geometry(main_geometry):
    """Initialize global geometry in worker processes."""
    global global_main_geometry
    global_main_geometry = main_geometry

class GraphCleaner:
    def __init__(self):
        pass

    def check_point(self, lon, lat):
        """Check if a point is within the main geometry."""
        prepared_geometry = prep(global_main_geometry)  # Prepare geometry in this process
        return prepared_geometry.intersects(Point(lon, lat))

    @staticmethod
    def haversine(lon1, lat1, lon2, lat2):
        """Calculate great-circle distance between two points."""
        R = 6371  # Earth's radius in kilometers
        dlon = math.radians(lon2 - lon1)
        dlat = math.radians(lat2 - lat1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def clean_and_restructure(self, graph_file):
        """Clean and restructure the graph into adjacency list format with bearings."""
        logging.info(f"Processing file: {graph_file}")
        try:
            # Load graph file
            with open(graph_file, "r", encoding="utf-8") as f:
                graph = json.load(f)

            nodes = graph["nodes"]
            structured_graph = {"nodes": {}, "metadata": {"total_nodes": 0}}

            for node_key, node_data in nodes.items():
                lon, lat = map(float, node_key.split("_"))
                if not self.check_point(lon, lat):
                    continue

                structured_graph["nodes"][node_key] = {
                    "lon": lon,
                    "lat": lat,
                    "edges": []
                }

                for edge in node_data.get("from_bearings", []):
                    target_key = edge["from"]
                    target_lon, target_lat = map(float, target_key.split("_"))
                    if self.check_point(target_lon, target_lat):
                        distance = self.haversine(lon, lat, target_lon, target_lat)
                        structured_graph["nodes"][node_key]["edges"].append({
                            "to": target_key,
                            "weight": distance,
                            "bearing": edge.get("bearing", None)  # Add bearing if available
                        })

            structured_graph["metadata"]["total_nodes"] = len(structured_graph["nodes"])

            # Save restructured graph
            output_file = graph_file.replace(".json", "_structured.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(structured_graph, f, ensure_ascii=False, indent=2)
            return output_file

        except Exception as e:
            logging.error(f"Error processing {graph_file}: {e}")
            return None


def process_graph_file(args):
    """Wrapper for multiprocessing."""
    cleaner, graph_file = args
    return cleaner.clean_and_restructure(graph_file)


def save_checkpoint(processed_files):
    """Save checkpoint to resume processing."""
    with open("checkpoint.json", "w", encoding="utf-8") as f:
        json.dump(processed_files, f)


def load_checkpoint():
    """Load checkpoint to resume processing."""
    try:
        with open("checkpoint.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def process_feature(feature):
    """Process individual feature to ensure valid geometry."""
    try:
        geom = shape(feature.get("geometry", {}))
        if isinstance(geom, MultiPolygon):
            return list(geom.geoms)  # Split MultiPolygon into Polygons
        elif isinstance(geom, Polygon):
            return geom
    except Exception as e:
        logging.warning(f"Skipping invalid feature: {e}")
    return None

def load_geojson(file_path):
    """Load and process GeoJSON file to ensure valid geometries."""
    logging.info("Loading GeoJSON file...")
    with open(file_path, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    if not isinstance(geojson, dict) or "features" not in geojson:
        raise ValueError("Invalid GeoJSON format")

    # Process geometries in parallel
    with Pool(processes=cpu_count()) as pool:
        geometries = pool.map(process_feature, geojson.get("features", []))

    # Flatten list and filter out invalid geometries
    geometries = [geom for sublist in geometries if sublist for geom in (sublist if isinstance(sublist, list) else [sublist])]

    if not geometries:
        raise ValueError("No valid geometries found in GeoJSON")

    logging.info(f"Loaded {len(geometries)} valid geometries from GeoJSON.")
    return MultiPolygon(geometries)

def main():
    # Load the main geometry from GeoJSON
    logging.info("Loading main geometry from GeoJSON...")
    main_geometry = load_geojson("eez.json")

    # Initialize GraphCleaner
    cleaner = GraphCleaner()

    # Base folder tempat file JSON disimpan
    base_folder = "./GridNode/"

    # Periksa file yang ada
    graph_files = [f"region_graph_part_{i}.json" for i in range(61)]
    existing_files = [os.path.join(base_folder, file) for file in graph_files if os.path.exists(os.path.join(base_folder, file))]

    if not existing_files:
        logging.error("No graph files found to process in folder GridNode.")
        return

    logging.info(f"Found {len(existing_files)} graph files to process.")

    # Load checkpoint to resume
    processed_files = set(load_checkpoint())
    files_to_process = [f for f in existing_files if f not in processed_files]

    logging.info(f"Resuming from checkpoint. {len(files_to_process)} files left to process.")

    # Initialize multiprocessing with geometry
    with Pool(processes=cpu_count(), initializer=init_geometry, initargs=(main_geometry,)) as pool:
        with tqdm(total=len(files_to_process), desc="Processing files") as pbar:
            for result in pool.imap_unordered(process_graph_file, [(cleaner, file) for file in files_to_process]):
                if result:
                    processed_files.add(result)
                    save_checkpoint(list(processed_files))
                pbar.update(1)

    logging.info("Graph restructuring complete.")

if __name__ == "__main__":
    main()
