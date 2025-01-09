import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple, Dict
import logging

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
    
class WaveDataLocator:
    def __init__(self, wave_data: Dict, wave_file: str):
        """
        Initializes the WaveDataLocator with wave data.
        """
        self.wave_data = wave_data
        self.wave_file = wave_file
        self.required_vars = ["dirpwsfc", "htsgwsfc", "perpwsfc"]

        # Validate wave_data structure
        self._validate_wave_data()

        # Build KDTree for wave data coordinates and aggregate data
        self.wave_tree, self.wave_coords, self.aggregated_wave_data = self._build_wave_tree()

    def _validate_wave_data(self):
        """
        Validates the structure of wave_data.
        """
        if "variables" not in self.wave_data:
            raise KeyError("Wave data missing 'variables' key.")

        for var in self.required_vars:
            if var not in self.wave_data["variables"]:
                raise KeyError(f"Wave data missing '{var}' variable.")
            for key in ["latitude", "longitude", "data"]:
                if key not in self.wave_data["variables"][var]:
                    raise KeyError(f"Wave data variable '{var}' missing '{key}'.")

    def _build_wave_tree(self) -> Tuple[cKDTree, np.ndarray, Dict[str, np.ndarray]]:
        """
        Builds a KDTree for wave data coordinates and aggregates data.
        """
        logger.info("Membangun KDTree untuk koordinat wave data...")
        aggregated_wave_data = {}
        wave_coords = None

        for var in self.required_vars:
            var_data = self.wave_data["variables"][var]
            latitudes = var_data["latitude"]
            longitudes = var_data["longitude"]
            data_lists = var_data["data"]

            if len(latitudes) != len(longitudes) or len(latitudes) != len(data_lists):
                raise ValueError(f"Length mismatch dalam wave data untuk variable {var}.")

            longitudes_np = np.array(longitudes)
            latitudes_np = np.array(latitudes)
            data_np = np.array(data_lists)

            if not np.issubdtype(data_np.dtype, np.number):
                raise ValueError(f"Wave data untuk variable {var} mengandung nilai non-numeric.")

            if wave_coords is None:
                wave_coords = np.column_stack((longitudes_np[0], latitudes_np[0]))
            else:
                current_coords = np.column_stack((longitudes_np[0], latitudes_np[0]))
                if not np.allclose(wave_coords, current_coords, atol=1e-6):
                    raise ValueError(f"Coordinates mismatch antara variables dalam wave data.")

            aggregated_wave_data[var] = np.mean(data_np, axis=0)

        if wave_coords is None or wave_coords.shape[0] == 0:
            raise ValueError("Tidak ada koordinat untuk membangun KDTree.")

        wave_tree = cKDTree(wave_coords)
        logger.info("KDTree untuk wave data berhasil dibangun.")
        return wave_tree, wave_coords, aggregated_wave_data

    def get_wave_data(self, coord: Tuple[float, float]) -> Dict[str, float]:
        """
        Retrieves wave data for the given coordinate.
        """
        distance, idx = self.wave_tree.query(coord)
        if idx < 0 or idx >= len(self.wave_coords):
            raise ValueError(f"Invalid index {idx} untuk koordinat {coord}.")

        wave_data_point = {}
        for var in self.required_vars:
            wave_data_point[var] = self.aggregated_wave_data[var][idx]
        return wave_data_point