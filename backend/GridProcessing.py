import json
import math
import os
import gzip
import time
import logging
import multiprocessing as mp
from typing import Dict, List, Tuple
from shapely.geometry import Point, Polygon, MultiPolygon, shape
from shapely.prepared import prep
from tqdm import tqdm
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class RegionGraphGenerator:
    def __init__(self, config):
        self.config = config
        self.geometries = []
        self.main_geometry = None
        self.prepared_geometry = None
        self.graph = {
            "metadata": {
                "total_nodes": 0,
                "total_edges": 0,
                "grid_spacing": config["GRID_SPACING"],
                "start_node": config["START_NODE"]
            },
            "nodes": {},
            "edges": []
        }
        self.output_files = []
        self.current_file_index = 0
        self.max_file_size = config.get("MAX_FILE_SIZE_MB", 100) * 1024 * 1024

    def validate_config(self):
        """Validate required configuration parameters."""
        required_keys = ["START_NODE", "GEOJSON_FILE", "GRID_SPACING", "OUTPUT_FILE"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        if not os.path.exists(self.config["GEOJSON_FILE"]):
            raise FileNotFoundError(f"GeoJSON file not found: {self.config['GEOJSON_FILE']}")
        logger.info("Configuration validated successfully.")

    def load_geojson(self):
        """Load and process GeoJSON file with improved performance."""
        logger.info("Loading GeoJSON file...")
        with open(self.config["GEOJSON_FILE"], "r") as f:
            geojson = json.load(f)

        # Validate GeoJSON
        if not isinstance(geojson, dict) or 'features' not in geojson:
            raise ValueError("Invalid GeoJSON format")

        # Parallel processing of geometries
        with mp.Pool(processes=mp.cpu_count()) as pool:
            valid_geometries = pool.starmap(
                self._process_feature, 
                [(feature,) for feature in geojson.get("features", [])]
            )
        
        # Filter out None values
        self.geometries = [geom for geom in valid_geometries if geom is not None]

        if not self.geometries:
            raise ValueError("No valid geometries found in GeoJSON")

        self.main_geometry = MultiPolygon(self.geometries)
        self.prepared_geometry = prep(self.main_geometry)
        logger.info(f"Loaded GeoJSON with {len(self.geometries)} geometries.")

    @staticmethod
    def _process_feature(feature):
        """Static method to process a single feature, can be multiprocessed."""
        try:
            geom = shape(feature.get("geometry", {}))
            if isinstance(geom, (Polygon, MultiPolygon)):
                return geom if isinstance(geom, Polygon) else list(geom.geoms)[0]
        except Exception as e:
            logger.warning(f"Skipping invalid feature: {e}")
        return None

    def save_graph_chunk(self):
        """Save graph chunk to a compressed file with improved file handling."""
        chunk = {
            "metadata": self.graph["metadata"],
            "nodes": self.graph["nodes"],
            "edges": self.graph["edges"]
        }

        filename = f"{self.config['OUTPUT_FILE']}_part_{self.current_file_index}.json.gz"
        try:
            with gzip.open(filename, "wt", encoding='utf-8') as f:
                json.dump(chunk, f, ensure_ascii=False)

            self.output_files.append(filename)
            self.current_file_index += 1

            self.graph["metadata"]["total_nodes"] += len(self.graph["nodes"])
            self.graph["metadata"]["total_edges"] += len(self.graph["edges"])

            logger.info(f"💾 Saved chunk {self.current_file_index}: {len(self.graph['nodes'])} nodes, {len(self.graph['edges'])} edges")

            self.graph["nodes"] = {}
            self.graph["edges"] = []
        except Exception as e:
            logger.error(f"Error saving graph chunk to {filename}: {e}")

    def create_region_graph(self):
        start_time = time.time()
        self.validate_config()
        self.load_geojson()

        start_lon = self.config["START_NODE"]["lon"]
        start_lat = self.config["START_NODE"]["lat"]
        grid_spacing = self.config["GRID_SPACING"]

        # Use numpy for faster calculations and set generation
        max_nodes = self.config.get("MAX_NODES", float('inf'))
        
        # Generate potential grid points more efficiently
        extent = self.config.get("INDONESIA_EXTENT", None)
        if extent:
            lon_min, lat_min, lon_max, lat_max = extent
            lon_range = np.arange(lon_min, lon_max, grid_spacing)
            lat_range = np.arange(lat_min, lat_max, grid_spacing)
        else:
            # Fallback to bounding box of main geometry
            bounds = self.main_geometry.bounds
            lon_range = np.arange(bounds[0], bounds[2], grid_spacing)
            lat_range = np.arange(bounds[1], bounds[3], grid_spacing)

        # Vectorized point generation and filtering
        total_points = len(lon_range) * len(lat_range)
        logger.info(f"Total potential grid points: {total_points}")

        nodes_generated = 0
        with tqdm(total=max_nodes, 
                  desc="🌍 Generating Region Graph", 
                  unit="node", 
                  dynamic_ncols=True) as pbar:
            for lon in lon_range:
                for lat in lat_range:
                    if nodes_generated >= max_nodes:
                        break

                    grid_key = f"{lon}_{lat}"
                    half_spacing = grid_spacing / 2
                    
                    # Use prepared geometry for faster intersection checks
                    grid_polygon = Polygon([
                        (lon - half_spacing, lat - half_spacing),
                        (lon + half_spacing, lat - half_spacing),
                        (lon + half_spacing, lat + half_spacing),
                        (lon - half_spacing, lat + half_spacing),
                        (lon - half_spacing, lat - half_spacing)
                    ])

                    # Use prepared geometry for faster intersection
                    if not self.prepared_geometry.intersects(grid_polygon):
                        continue

                    # Add node to graph
                    self.graph["nodes"][grid_key] = {
                        "lon": lon,
                        "lat": lat
                    }
                    nodes_generated += 1
                    pbar.update(1)

                    # Save chunk if size exceeds limit
                    current_size = len(json.dumps(self.graph))
                    if current_size > self.max_file_size:
                        self.save_graph_chunk()

        # Save remaining nodes
        if self.graph["nodes"]:
            self.save_graph_chunk()

        self.create_index_file()
        end_time = time.time()
        logger.info(f"Graph generation completed in {end_time - start_time:.2f} seconds.")

    def create_index_file(self):
        """Create an index file to track all generated parts."""
        index_data = {
            "metadata": self.graph["metadata"],
            "parts": self.output_files,
            "summary": {
                "total_parts": len(self.output_files),
                "total_nodes": self.graph["metadata"]["total_nodes"],
                "total_edges": self.graph["metadata"]["total_edges"]
            }
        }

        index_filename = f"{self.config['OUTPUT_FILE']}_index.json"
        try:
            with open(index_filename, "w", encoding='utf-8') as f:
                json.dump(index_data, f, indent=2, ensure_ascii=False)

            logger.info("📋 Index file created.")
            logger.info(f"Total Parts: {len(self.output_files)}")
            logger.info(f"Total Nodes: {self.graph['metadata']['total_nodes']}")
            logger.info(f"Total Edges: {self.graph['metadata']['total_edges']}")
        except Exception as e:
            logger.error(f"Error saving index file: {e}")

def main():
    config = {
        "START_NODE": {"lon": 113.53701042040963, "lat": -4.736794825727632},
        "INDONESIA_EXTENT": [92.0, -15.0, 141.0, 10.0],
        "GRID_SPACING": 1 / 111.32,  # 1km in degrees
        "GEOJSON_FILE": "eez.json",
        "OUTPUT_FILE": "region_graph",
        "MAX_NODES": 500_000,
        "MAX_FILE_SIZE_MB": 100
    }

    generator = RegionGraphGenerator(config)
    generator.create_region_graph()

if __name__ == "__main__":
    main()