import os
import json
import logging
import numpy as np
import netCDF4 as nc

from hashlib import sha256
from datetime import datetime, timedelta, timezone
from typing import Tuple, Dict, Any
from dataclasses import dataclass

from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from pyproj import CRS, Transformer

from constants import Config
from utils import local_file_exists_for_all, load_local_data, save_wave_data
from managers.Djikstra import RouteOptimizer
from managers.WaveDataLocator import WaveDataLocator
# -------------------------------------------------------------------
# Asumsikan modul eksternal "constants" dan "utils" berikut tersedia:
# from constants import Config
# from utils import (
#     local_file_exists_for_all, 
#     load_local_data, 
#     save_wave_data, 
#     get_file_path
# )
# Pastikan menyesuaikan fungsinya dengan environment Anda.
# -------------------------------------------------------------------

# Konfigurasi Logging
logging.basicConfig(
    level=logging.INFO,  # ganti ke DEBUG jika ingin logging lebih detail
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Fungsi tambahan
# -------------------------------------------------------------------

def read_nc_variable_optimized(
    dataset: nc.Dataset,
    var_name: str,
    cache_path: str,
    lat_inds: Tuple[int, int],
    lon_inds: Tuple[int, int]
) -> np.ndarray:
    """
    Membaca variabel dari NetCDF dengan pengolahan cepat dan slicing:
    - Mengganti nilai ekstrem (>1e10 atau < -1e10), 0.0, None, NaN dengan np.nan.
    - Jika semua data tidak valid, ambil dari cache lokal.
    - Memanfaatkan slicing untuk hanya memuat subset data yang diperlukan.

    Args:
    - dataset: NetCDF Dataset.
    - var_name: Nama variabel yang akan dibaca.
    - cache_path: Path cache lokal untuk fallback.
    - lat_inds: Tuple indeks latitude (start, end).
    - lon_inds: Tuple indeks longitude (start, end).

    Returns:
    - data: numpy array 2D/3D yang telah diproses.
    """
    logger.info(f"Membaca variabel: {var_name} dari dataset NetCDF dengan slicing.")
    try:
        # Ambil data variabel hanya pada slice yang diperlukan
        var_data = dataset.variables[var_name][
            0,  # Asumsi variabel memiliki dimensi waktu di indeks 0
            lat_inds[0]:lat_inds[1] + 1,
            lon_inds[0]:lon_inds[1] + 1
        ]

        # Ambil _FillValue untuk mengganti nilai yang tidak valid
        fill_value = getattr(dataset.variables[var_name], "_FillValue", None)

        # Ganti nilai ekstrem, _FillValue, 0.0, None, NaN
        if fill_value is not None:
            var_data = np.where(var_data == fill_value, np.nan, var_data)
        var_data = np.where((var_data == 0.0) | (var_data > 1e10) | (var_data < -1e10), np.nan, var_data)

        # Periksa apakah semua nilai tidak valid
        if np.isnan(var_data).all():
            logger.warning(f"Semua nilai pada variabel '{var_name}' tidak valid. Mengambil cache lokal.")
            if os.path.exists(cache_path):
                with open(cache_path, "r") as cache_file:
                    cached_data = json.load(cache_file)
                    return np.array(cached_data["variables"][var_name]["data"], dtype=np.float32)
            else:
                raise ValueError(f"Cache lokal tidak ditemukan untuk variabel '{var_name}'.")

        logger.info(f"Variabel '{var_name}' berhasil dibaca dan diproses.")
        return var_data

    except Exception as e:
        logger.error(f"Error membaca variabel '{var_name}': {e}", exc_info=True)
        raise

@dataclass
class GridPoint:
    lat: float
    lon: float
    value: float

# -------------------------------------------------------------------
# GridProcessor: Mengelola Transformasi, Interpolasi, dsb.
# -------------------------------------------------------------------
class GridProcessor:
    """
    Class berisi metode statis untuk:
    1. Transformasi lat/lon <-> x/y (UTM).
    2. Mengisi nilai NaN (interpolasi).
    3. (Opsional) IDW dengan cKDTree.
    
    DISLAIMER:
      - Menggunakan satu zona UTM (EPSG:32749) di sini hanya contoh.
      - Untuk area sangat luas di Indonesia, perlu pendekatan multi-zona
        atau proyeksi lain agar distorsi minimal.
    """
    crs_wgs84 = CRS.from_epsg(4326)
    # Contoh: UTM zone 49S (EPSG:32749). Ganti sesuai area Anda.
    crs_utm = CRS.from_epsg(32749)

    @staticmethod
    def latlon_to_utm(lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transformasi array lat/lon (bentuk 2D atau 1D) -> x/y (meter) di UTM zone.
        """
        transformer = Transformer.from_crs(
            GridProcessor.crs_wgs84, 
            GridProcessor.crs_utm, 
            always_xy=True
        )
        lon_flat = lon.flatten()
        lat_flat = lat.flatten()
        x_m, y_m = transformer.transform(lon_flat, lat_flat)
        return x_m.reshape(lat.shape), y_m.reshape(lat.shape)

    @staticmethod
    def utm_to_latlon(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transformasi array x/y (meter) UTM -> lat/lon WGS84.
        """
        transformer = Transformer.from_crs(
            GridProcessor.crs_utm, 
            GridProcessor.crs_wgs84, 
            always_xy=True
        )
        x_flat = x.flatten()
        y_flat = y.flatten()
        lon, lat = transformer.transform(x_flat, y_flat)
        return lat.reshape(x.shape), lon.reshape(x.shape)

    @staticmethod
    def fill_null_values(data: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Mengisi nilai NaN dengan griddata (cubic -> linear -> nearest).
        Koordinat: x,y (meter).
        """
        # Re-check if data is all nan or not
        valid_mask = ~np.isnan(data)
        if not np.any(valid_mask):
            return data  # jika semua NaN, kembalikan saja

        coords = np.column_stack((x[valid_mask], y[valid_mask]))
        values = data[valid_mask]

        # Coba cubic
        filled_data = griddata(
            coords,
            values,
            (x, y),
            method='cubic',
            fill_value=np.nan
        )

        # Lalu linear
        nan_mask = np.isnan(filled_data)
        if np.any(nan_mask):
            filled_data[nan_mask] = griddata(
                coords,
                values,
                (x[nan_mask], y[nan_mask]),
                method='linear',
                fill_value=np.nan
            )

        # Lalu nearest
        nan_mask = np.isnan(filled_data)
        if np.any(nan_mask):
            filled_data[nan_mask] = griddata(
                coords,
                values,
                (x[nan_mask], y[nan_mask]),
                method='nearest',
                fill_value=np.nan
            )

        return filled_data

    @staticmethod
    def fill_null_values_kdtree(
        data: np.ndarray, 
        x: np.ndarray, 
        y: np.ndarray, 
        k: int = 8, 
        p: float = 2.0
    ) -> np.ndarray:
        """
        Pendekatan manual IDW dengan cKDTree untuk mengisi data NaN.
        data, x, y harus punya shape sama (2D).
        k: jumlah tetangga, p: power (IDW).
        """
        valid_mask = ~np.isnan(data)
        if not np.any(valid_mask):
            return data

        coords_valid = np.column_stack((x[valid_mask], y[valid_mask]))
        values_valid = data[valid_mask]

        # Build cKDTree
        tree = cKDTree(coords_valid)
        filled_data = data.copy()

        nan_idx = np.where(np.isnan(data))
        nan_coords = np.column_stack((x[nan_idx], y[nan_idx]))

        # Dapatkan k tetangga
        dist, idx_neighbor = tree.query(nan_coords, k=k)

        for i in range(nan_coords.shape[0]):
            di = dist[i, :]
            nbr_idx = idx_neighbor[i, :]

            # jika ada distance = 0
            zero_mask = (di == 0)
            if np.any(zero_mask):
                val = values_valid[nbr_idx[zero_mask][0]]
                filled_data[nan_idx[0][i], nan_idx[1][i]] = val
                continue

            # IDW
            w = 1.0 / np.power(di, p)
            wsum = np.sum(w)
            val = np.sum(w * values_valid[nbr_idx]) / wsum
            filled_data[nan_idx[0][i], nan_idx[1][i]] = val

        return filled_data

# -------------------------------------------------------------------
# Fungsi utama pemrosesan wave data
# -------------------------------------------------------------------
def process_wave_data(
    data: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    interpolate: bool = True,
    use_kdtree: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pipeline:
    1) lat/lon -> x_m, y_m (UTM).
    2) (Opsional) fill null/NaN (dgn IDW cKDTree atau griddata).
    3) x_m, y_m -> lat, lon (kembalikan agar FE dapat lat/lon).
    
    - data: shape (M, N)
    - lat: shape (M, N) atau (M,) => di-meshgrid
    - lon: shape (N,) => di-meshgrid
    Return: (processed_data, new_lat, new_lon) dgn shape yang sama.
    """
    # Ganti 0.0 => np.nan (jika 0.0 dianggap invalid)
    data = np.where((data == 0.0), np.nan, data)
    
    # Validasi data ekstrem
    if np.nanmax(data) > 1e10 or np.nanmin(data) < -1e10:
        data = np.where((data > 1e10) | (data < -1e10), np.nan, data)
        logger.warning("Data memiliki nilai ekstrem, diganti dengan np.nan")

    # Pastikan array 2D
    data_array = np.array(data, dtype=float)
    lat_array = np.array(lat, dtype=float)
    lon_array = np.array(lon, dtype=float)

    # buat meshgrid jika lat/lon 1D
    if lat_array.ndim == 1 and lon_array.ndim == 1:
        lon_array, lat_array = np.meshgrid(lon_array, lat_array)

    # Pastikan shape sama
    if not (data_array.shape == lat_array.shape == lon_array.shape):
        raise ValueError(
            f"Shape mismatch: data={data_array.shape}, lat={lat_array.shape}, lon={lon_array.shape}"
        )

    # Step 1: transform lat/lon -> x_m, y_m
    x_m, y_m = GridProcessor.latlon_to_utm(lat_array, lon_array)

    # Step 2: Interpolasi isi NaN
    if interpolate:
        if use_kdtree:
            data_array = GridProcessor.fill_null_values_kdtree(data_array, x_m, y_m, k=8, p=2.0)
        else:
            data_array = GridProcessor.fill_null_values(data_array, x_m, y_m)

    # Step 3: transform x_m, y_m -> lat, lon agar FE tetap terima lat/lon
    new_lat, new_lon = GridProcessor.utm_to_latlon(x_m, y_m)

    return data_array, new_lat, new_lon

# -------------------------------------------------------------------
# Mendapatkan dataset terbaru (URL atau local file)
# -------------------------------------------------------------------
def get_latest_dataset_url() -> Tuple[Any, str, str, bool]:
    """
    Menemukan dataset NetCDF terbaru (4 hari ke belakang).
    Mengembalikan (dataset_url, date_str, time_slot, all_local).
    all_local=True berarti semua data ada di file lokal.
    """
    now = datetime.utcnow()
    logger.info("Mencari dataset terbaru (4 hari ke belakang)...")

    for _ in range(4):  # max 4 hari mundur
        date_str = now.strftime("%Y%m%d")
        for time_slot in Config.TIME_SLOTS:
            # Jika semua file local tersedia
            if local_file_exists_for_all(date_str, time_slot):
                logger.info(f"Dataset lokal lengkap: {date_str}-{time_slot}")
                return None, date_str, time_slot, True

            # Jika tidak lengkap di local, coba URL
            dataset_url = f"{Config.BASE_URL}{date_str}/gfswave.global.0p16_{time_slot}"
            try:
                with nc.Dataset(dataset_url) as ds:
                    logger.info(f"Dataset ditemukan di URL: {dataset_url}")
                    return dataset_url, date_str, time_slot, False
            except Exception as e:
                logger.debug(f"Tidak tersedia: {dataset_url} => {e}")

        now -= timedelta(days=1)

    raise ValueError("Tidak ada dataset valid dalam 4 hari terakhir!")

# -------------------------------------------------------------------
# Endpoint/Response generator (misal untuk Flask)
# -------------------------------------------------------------------
def get_wave_data_response_interpolate(args: Dict[str, Any], wave_data_locator: WaveDataLocator, route_optimizer: RouteOptimizer):
    """
    1. Periksa bounding box & param (interpolate, use_kdtree).
    2. Dapatkan dataset URL / local file (get_latest_dataset_url).
    3. Cek cache => jika ada, kembalikan; jika tidak, proses.
    4. Baca data (local atau NetCDF remote).
    5. Filter sesuai bounding box (kalau is_dynamic=True).
    6. Panggil process_wave_data => lat/lon + data final.
    7. Simpan hasil ke cache & return JSON-friendly response.
    """
    try:
        # Param bounding box
        # Param bounding box
        min_lat = float(args.get("min_lat")) if args.get("min_lat") is not None else None
        max_lat = float(args.get("max_lat")) if args.get("max_lat") is not None else None
        min_lon = float(args.get("min_lon")) if args.get("min_lon") is not None else None
        max_lon = float(args.get("max_lon")) if args.get("max_lon") is not None else None

        # Param kontrol
        interpolate = args.get("interpolate", "true").lower() == "true"
        use_kdtree = args.get("use_kdtree", "true").lower() == "true"

        # Apakah bounding box disediakan
        is_dynamic = all([
            min_lat is not None,
            max_lat is not None,
            min_lon is not None,
            max_lon is not None
        ])

        # Ambil dataset
        dataset_url, date_str, time_slot, all_local = get_latest_dataset_url()

        # Buat cache key
        cache_dict = {
            "min_lat": min_lat,
            "max_lat": max_lat,
            "min_lon": min_lon,
            "max_lon": max_lon,
            "interpolate": interpolate,
            "use_kdtree": use_kdtree,
            "is_dynamic": is_dynamic,
            "date_str": date_str,
            "time_slot": time_slot,
        }
        cache_key = sha256(json.dumps(cache_dict, sort_keys=True).encode()).hexdigest()
        cache_dir = "./data/cache"
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{cache_key}.json")

        # Cek cache
        if os.path.exists(cache_path):
            with open(cache_path, "r") as cache_file:
                cached_data = json.load(cache_file)
            logger.info("Data ditemukan di cache => kembalikan respons.")
            return {"success": True, "data": cached_data}, 200

        # Bikin response
        variables = ["htsgwsfc", "dirpwsfc", "perpwsfc"]
        combined_response = {"variables": {}, "metadata": {}}

        # -----------------------------------------------------------
        # CASE 1: Data local
        # -----------------------------------------------------------
        if all_local:
            for var in variables:
                # Baca file local
                var_data_dict = load_local_data(var, date_str, time_slot)

                data_array = np.array(var_data_dict["data"], dtype=float)
                lat_array = np.array(var_data_dict["latitude"], dtype=float)
                lon_array = np.array(var_data_dict["longitude"], dtype=float)

                # meshgrid lat/lon jika 1D
                lon_mesh, lat_mesh = np.meshgrid(lon_array, lat_array)

                # Proses data
                processed_data, processed_lat, processed_lon = process_wave_data(
                    data_array,
                    lat_mesh,
                    lon_mesh,
                    interpolate=interpolate,
                    use_kdtree=use_kdtree
                )

                combined_response["variables"][var] = {
                    "description": var_data_dict.get("variable"),
                    "units": var_data_dict.get("units"),
                    "data": processed_data.tolist(),
                    "latitude": processed_lat.tolist(),
                    "longitude": processed_lon.tolist(),
                }

            meta_data = load_local_data("htsgwsfc", date_str, time_slot).get("metadata", {})
            combined_response["metadata"] = meta_data

        # -----------------------------------------------------------
        # CASE 2: Data remote (NetCDF)
        # -----------------------------------------------------------
        else:
            dataset = nc.Dataset(dataset_url)
            lat_nc = dataset.variables["lat"][:]
            lon_nc = dataset.variables["lon"][:]

            if is_dynamic:
                min_lat, max_lat = sorted([min_lat, max_lat])
                min_lon, max_lon = sorted([min_lon, max_lon])
                lat_inds = np.where((lat_nc >= min_lat) & (lat_nc <= max_lat))[0]
                lon_inds = np.where((lon_nc >= min_lon) & (lon_nc <= max_lon))[0]
            else:
                # default bounding
                lat_inds = np.where(
                    (lat_nc >= Config.INDONESIA_EXTENT["min_lat"]) &
                    (lat_nc <= Config.INDONESIA_EXTENT["max_lat"])
                )[0]
                lon_inds = np.where(
                    (lon_nc >= Config.INDONESIA_EXTENT["min_lon"]) &
                    (lon_nc <= Config.INDONESIA_EXTENT["max_lon"])
                )[0]

            filtered_lat = lat_nc[lat_inds]
            filtered_lon = lon_nc[lon_inds]

            for var in variables:
                var_data_slice = read_nc_variable_optimized(
                    dataset,
                    var_name= var,
                    cache_path=cache_path,
                    lat_inds=(lat_inds.min(), lat_inds.max()),
                    lon_inds=(lon_inds.min(), lon_inds.max())
                )

                # Buat meshgrid
                lon_mesh, lat_mesh = np.meshgrid(filtered_lon, filtered_lat)

                # Proses data
                processed_data, processed_lat, processed_lon = process_wave_data(
                    var_data_slice,
                    lat_mesh,
                    lon_mesh,
                    interpolate=interpolate,
                    use_kdtree=use_kdtree
                )

                # Coba ambil "long_name" dari var, fallback "N/A"
                units = "N/A"
                try:
                    units = dataset.variables[var].getncattr("long_name")
                except:
                    pass

                combined_response["variables"][var] = {
                    "variable": var,
                    "units": units,
                    "data": processed_data.tolist(),
                    "latitude": processed_lat.tolist(),
                    "longitude": processed_lon.tolist(),
                }
                # Simpan data mentah jika bukan dynamic
                if not is_dynamic:
                    raw_response = {
                        "variable": var,
                        "units": units,
                        "data": processed_data.tolist(),
                        "latitude": processed_lat.tolist(),
                        "longitude": processed_lon.tolist(),
                        "metadata": {
                            "dataset_url": dataset_url,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "date": date_str,
                            "time_slot": time_slot,
                            "dynamic_extent": is_dynamic,
                            "extent": {
                                "min_lat": Config.INDONESIA_EXTENT["min_lat"],
                                "max_lat": Config.INDONESIA_EXTENT["max_lat"],
                                "min_lon": Config.INDONESIA_EXTENT["min_lon"],
                                "max_lon": Config.INDONESIA_EXTENT["max_lon"],
                            }
                        }
                    }
                    save_wave_data(raw_response, var, date_str, time_slot)

            # Tambah metadata
            combined_response["metadata"] = {
                "dataset_url": dataset_url,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "date": date_str,
                "time_slot": time_slot,
                "dynamic_extent": is_dynamic,
                "processing": {
                    "interpolated": interpolate,
                    "use_kdtree": use_kdtree,
                },
                "extent": {
                    "min_lat": min_lat if is_dynamic else Config.INDONESIA_EXTENT["min_lat"],
                    "max_lat": max_lat if is_dynamic else Config.INDONESIA_EXTENT["max_lat"],
                    "min_lon": min_lon if is_dynamic else Config.INDONESIA_EXTENT["min_lon"],
                    "max_lon": max_lon if is_dynamic else Config.INDONESIA_EXTENT["max_lon"],
                }
            }

        if not combined_response["variables"]:
            logger.error("Tidak ada variabel yang valid untuk diproses. Mengambil cache terakhir.")
            if os.path.exists(cache_path):
                with open(cache_path, "r") as cache_file:
                    cached_response = json.load(cache_file)
                return {"success": True, "data": cached_response}, 200
            else:
                return {"success": False, "error": "Tidak ada data valid atau cache lokal yang tersedia."}, 500
    
        # -----------------------------------------------------------
        # Simpan ke cache
        # -----------------------------------------------------------
        with open(cache_path, "w") as f:
            json.dump(combined_response, f)

        # -----------------------------------------------------------
        # Perbarui WaveDataLocator dan RouteOptimizer
        # -----------------------------------------------------------
        logger.info("Memperbarui WaveDataLocator...")
        new_wave_data_locator = WaveDataLocator(combined_response)

        logger.info("Memperbarui RouteOptimizer...")
        route_optimizer.update_wave_data_locator(new_wave_data_locator)
        
        return {"success": True, "data": combined_response}, 200

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}, 500
