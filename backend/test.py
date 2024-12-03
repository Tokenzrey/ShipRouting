import json
import math
from shapely.geometry import Point, Polygon, MultiPolygon, shape

# Konfigurasi
CONFIG = {
    "START_NODE": {"lon": 113.53701042040963, "lat": -4.736794825727632},
    "INDONESIA_EXTENT": [92.0, -15.0, 141.0, 10.0],
    "GRID_SPACING": 1 / 111.32,  # 1km in degrees
    "GEOJSON_FILE": "eez.json",  # Path to your GeoJSON file
    "OUTPUT_FILE": "region_graph.json",  # Output JSON file
}

def haversine_distance(lon1, lat1, lon2, lat2):
    """Calculate the great-circle distance between two points."""
    R = 6371  # Earth's radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c  # Distance in km

def create_region_graph():
    print("Loading GeoJSON file...")
    with open(CONFIG["GEOJSON_FILE"], "r") as f:
        geojson = json.load(f)

    # Extract MultiPolygon or Polygon from GeoJSON
    geometries = []
    for feature in geojson["features"]:
        geom = shape(feature["geometry"])
        if isinstance(geom, Polygon):
            geometries.append(geom)
        elif isinstance(geom, MultiPolygon):
            geometries.extend(list(geom.geoms))  # Use .geoms to iterate over polygons in MultiPolygon

    main_geometry = MultiPolygon(geometries)

    print(f"Loaded GeoJSON with {len(geometries)} geometries.")
    print("Initializing region graph...")

    # Initialize graph
    graph = {
        "nodes": {},  # {key: {lon, lat}}
        "edges": [],  # [{from, to, weight}]
    }

    # Initialize processing queue
    start_lon = CONFIG["START_NODE"]["lon"]
    start_lat = CONFIG["START_NODE"]["lat"]
    grid_spacing = CONFIG["GRID_SPACING"]

    visited = set()
    queue = [(start_lon, start_lat)]
    total_processed = 0

    while queue:
        lon, lat = queue.pop(0)
        grid_key = f"{lon}_{lat}"  # Use full precision for keys

        if grid_key in visited:
            print(f"Skipping already visited node: {grid_key}")
            continue

        visited.add(grid_key)
        total_processed += 1

        # Create grid extent
        half_spacing = grid_spacing / 2
        grid_polygon = Polygon([
            (lon - half_spacing, lat - half_spacing),
            (lon + half_spacing, lat - half_spacing),
            (lon + half_spacing, lat + half_spacing),
            (lon - half_spacing, lat + half_spacing),
            (lon - half_spacing, lat - half_spacing),
        ])

        # Check intersection with main geometry
        if not main_geometry.intersects(grid_polygon):
            print(f"Grid {grid_key} does not intersect with the geometry.")
            continue

        # Add node to graph
        graph["nodes"][grid_key] = {"lon": lon, "lat": lat}
        print(f"Added node: {grid_key}")

        # Process neighbors
        neighbors = [
            (lon - grid_spacing, lat),  # Left
            (lon + grid_spacing, lat),  # Right
            (lon, lat - grid_spacing),  # Bottom
            (lon, lat + grid_spacing),  # Top
            (lon - grid_spacing, lat - grid_spacing),  # Bottom-left
            (lon + grid_spacing, lat - grid_spacing),  # Bottom-right
            (lon - grid_spacing, lat + grid_spacing),  # Top-left
            (lon + grid_spacing, lat + grid_spacing),  # Top-right
        ]

        for neighbor_lon, neighbor_lat in neighbors:
            neighbor_key = f"{neighbor_lon}_{neighbor_lat}"  # Use full precision for keys
            distance = haversine_distance(lon, lat, neighbor_lon, neighbor_lat)

            if neighbor_key not in visited:
                queue.append((neighbor_lon, neighbor_lat))
                print(f"Queued neighbor: {neighbor_key}")

            # Add edge to graph
            graph["edges"].append({
                "from": grid_key,
                "to": neighbor_key,
                "weight": distance,
            })
            print(f"Added edge: {grid_key} -> {neighbor_key}, weight: {distance:.6f} km")

        # Show progress for every 100 nodes processed
        if total_processed % 100 == 0:
            print(f"Progress: {total_processed} nodes processed...")

    # Save graph to JSON
    print("Saving graph to JSON file...")
    with open(CONFIG["OUTPUT_FILE"], "w") as f:
        json.dump(graph, f, indent=2)

    print(f"Graph saved to {CONFIG['OUTPUT_FILE']}")
    print(f"Total nodes: {len(graph['nodes'])}")
    print(f"Total edges: {len(graph['edges'])}")

if __name__ == "__main__":
    create_region_graph()
