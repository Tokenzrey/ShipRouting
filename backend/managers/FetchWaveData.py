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
from concurrent.futures import ThreadPoolExecutor, as_completed

from constants import Config
from utils import local_file_exists_for_all, load_local_data, save_wave_data

# -------------------------------------------------------------------
# Konfigurasi Logging
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,  # Ganti ke DEBUG untuk informasi lebih detail
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Fungsi tambahan
# -------------------------------------------------------------------
def read_nc_variable(dataset: nc.Dataset, var_name: str) -> np.ndarray:
    """
    Membaca variabel 'var_name' dari NetCDF, mengganti _FillValue dengan np.nan.
    Return: numpy array 3D atau 2D (tergantung dataset).
    """
    var_data = dataset.variables[var_name][:]
    fill_value = getattr(dataset.variables[var_name], '_FillValue', None)
    if fill_value is not None:
        np.copyto(var_data, np.nan, where=(var_data == fill_value))
    return var_data

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

    DISCLAIMER:
      - Menggunakan satu zona UTM (EPSG:32749) di sini hanya contoh.
      - Untuk area sangat luas di Indonesia, perlu pendekatan multi-zona
        atau proyeksi lain agar distorsi minimal.
    """
    crs_wgs84 = CRS.from_epsg(4326)
    crs_utm = CRS.from_epsg(32749)  # UTM zone 49S

    # Reuse Transformer instances
    transformer_to_utm = Transformer.from_crs(crs_wgs84, crs_utm, always_xy=True)
    transformer_to_wgs84 = Transformer.from_crs(crs_utm, crs_wgs84, always_xy=True)

    @staticmethod
    def latlon_to_utm(lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transformasi array lat/lon (bentuk 2D atau 1D) -> x/y (meter) di UTM zone.
        """
        lon_flat = lon.ravel()
        lat_flat = lat.ravel()
        x_m, y_m = GridProcessor.transformer_to_utm.transform(lon_flat, lat_flat)
        return x_m.reshape(lat.shape), y_m.reshape(lat.shape)

    @staticmethod
    def utm_to_latlon(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transformasi array x/y (meter) UTM -> lat/lon WGS84.
        """
        x_flat = x.ravel()
        y_flat = y.ravel()
        lon, lat = GridProcessor.transformer_to_wgs84.transform(x_flat, y_flat)
        return lat.reshape(x.shape), lon.reshape(x.shape)

    @staticmethod
    def fill_null_values(data: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Mengisi nilai NaN dengan griddata (cubic -> linear -> nearest).
        Koordinat: x,y (meter).
        Operasi inplace digunakan untuk efisiensi memori.
        """
        # Re-check if data is all nan or not
        valid_mask = ~np.isnan(data)
        if not np.any(valid_mask):
            logger.debug("Tidak ada data valid untuk interpolasi.")
            return data  # jika semua NaN, kembalikan saja

        coords = np.column_stack((x[valid_mask], y[valid_mask]))
        values = data[valid_mask]

        # Interpolasi cubic
        filled_data = griddata(
            coords,
            values,
            (x, y),
            method='cubic',
            fill_value=np.nan
        )
        logger.debug(f"Interpolasi cubic selesai dengan {np.isnan(filled_data).sum()} NaNs tersisa.")

        # Interpolasi linear pada NaN yang tersisa
        nan_mask = np.isnan(filled_data)
        if np.any(nan_mask):
            filled_data[nan_mask] = griddata(
                coords,
                values,
                (x[nan_mask], y[nan_mask]),
                method='linear',
                fill_value=np.nan
            )
            logger.debug(f"Interpolasi linear selesai dengan {np.isnan(filled_data).sum()} NaNs tersisa.")

        # Interpolasi nearest pada NaN yang masih tersisa
        nan_mask = np.isnan(filled_data)
        if np.any(nan_mask):
            filled_data[nan_mask] = griddata(
                coords,
                values,
                (x[nan_mask], y[nan_mask]),
                method='nearest',
                fill_value=np.nan
            )
            logger.debug(f"Interpolasi nearest selesai dengan {np.isnan(filled_data).sum()} NaNs tersisa.")

        # Update data inplace
        np.copyto(data, filled_data)
        logger.debug("Data diupdate inplace setelah interpolasi.")

        return data

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
        Operasi inplace digunakan untuk efisiensi memori.
        """
        valid_mask = ~np.isnan(data)
        if not np.any(valid_mask):
            logger.debug("Tidak ada data valid untuk interpolasi dengan cKDTree.")
            return data

        coords_valid = np.column_stack((x[valid_mask], y[valid_mask]))
        values_valid = data[valid_mask]

        # Build cKDTree
        tree = cKDTree(coords_valid)

        nan_mask = np.isnan(data)
        nan_coords = np.column_stack((x[nan_mask], y[nan_mask]))

        if nan_coords.size == 0:
            logger.debug("Tidak ada NaN yang perlu diisi dengan cKDTree.")
            return data

        # Query tanpa n_jobs
        dist, idx_neighbor = tree.query(nan_coords, k=k)

        # Handle cases where k=1
        if k == 1:
            dist = dist[:, np.newaxis]
            idx_neighbor = idx_neighbor[:, np.newaxis]

        # Initialize filled values
        filled_values = np.full(nan_coords.shape[0], np.nan, dtype=data.dtype)

        # Identify where distance is zero
        zero_mask = dist == 0
        has_zero = zero_mask.any(axis=1)

        # Assign exact matches
        filled_values[has_zero] = values_valid[idx_neighbor[has_zero, 0]]

        # Assign IDW untuk non-zero distances
        non_zero = ~has_zero
        if np.any(non_zero):
            weights = 1.0 / np.power(dist[non_zero], p)
            weights /= np.sum(weights, axis=1)[:, np.newaxis]
            idw_values = np.sum(weights * values_valid[idx_neighbor[non_zero]], axis=1)
            filled_values[non_zero] = idw_values

        # Update data inplace
        data[nan_mask] = filled_values

        # Logging jumlah NaNs yang diisi
        filled_nans = np.isnan(data).sum()
        logger.debug(f"Jumlah NaNs setelah pengisian dengan cKDTree: {filled_nans}")

        return data

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
    # Ganti 0.0 => np.nan inplace
    data[data == 0.0] = np.nan
    logger.debug(f"NaNs setelah mengganti 0.0: {np.isnan(data).sum()}")

    # Validasi data ekstrem
    max_val = np.nanmax(data)
    min_val = np.nanmin(data)
    if max_val > 1e10 or min_val < -1e10:
        logger.warning("Data memiliki nilai ekstrem, diganti dengan np.nan.")
        data[(data > 1e10) | (data < -1e10)] = np.nan
        logger.debug(f"NaNs setelah mengganti nilai ekstrem: {np.isnan(data).sum()}")

    # Pastikan array 2D
    data_array = data.astype(np.float32, copy=False)
    lat_array = lat.astype(np.float32, copy=False)
    lon_array = lon.astype(np.float32, copy=False)

    # Buat meshgrid jika lat/lon 1D
    if lat_array.ndim == 1 and lon_array.ndim == 1:
        lon_array, lat_array = np.meshgrid(lon_array, lat_array)
        logger.debug("Meshgrid lat/lon dibuat.")

    # Pastikan shape sama
    if not (data_array.shape == lat_array.shape == lon_array.shape):
        raise ValueError(
            f"Shape mismatch: data={data_array.shape}, lat={lat_array.shape}, lon={lon_array.shape}"
        )

    # Step 1: transform lat/lon -> x_m, y_m
    x_m, y_m = GridProcessor.latlon_to_utm(lat_array, lon_array)
    logger.debug("Transform lat/lon ke UTM selesai.")

    # Step 2: Interpolasi isi NaN
    if interpolate:
        if use_kdtree:
            data_array = GridProcessor.fill_null_values_kdtree(data_array, x_m, y_m, k=8, p=2.0)
            logger.debug("Interpolasi dengan cKDTree selesai.")
        else:
            data_array = GridProcessor.fill_null_values(data_array, x_m, y_m)
            logger.debug("Interpolasi dengan griddata selesai.")

    # Cek jumlah NaNs setelah interpolasi
    remaining_nans = np.isnan(data_array).sum()
    logger.debug(f"NaNs setelah interpolasi: {remaining_nans}")

    if remaining_nans > 0:
        logger.warning(f"Ada {remaining_nans} NaNs yang masih tersisa setelah interpolasi.")

    # Step 3: transform x_m, y_m -> lat, lon
    new_lat, new_lon = GridProcessor.utm_to_latlon(x_m, y_m)
    logger.debug("Transform UTM ke lat/lon selesai.")

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
def get_wave_data_response_interpolate(args: Dict[str, Any]):
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
        min_lat = args.get("min_lat", type=float)
        max_lat = args.get("max_lat", type=float)
        min_lon = args.get("min_lon", type=float)
        max_lon = args.get("max_lon", type=float)

        # Param kontrol
        interpolate = args.get("interpolate", default="true").lower() == "true"
        use_kdtree = args.get("use_kdtree", default="true").lower() == "true"

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

        # Definisikan fungsi untuk memproses satu variabel
        def process_variable_local(var):
            var_data_dict = load_local_data(var, date_str, time_slot)
            logger.info(f"Memproses variabel lokal: {var}")

            data_array = np.array(var_data_dict["data"], dtype=np.float32)
            lat_array = np.array(var_data_dict["latitude"], dtype=np.float32)
            lon_array = np.array(var_data_dict["longitude"], dtype=np.float32)

            # Meshgrid lat/lon jika 1D
            if lat_array.ndim == 1 and lon_array.ndim == 1:
                lon_mesh, lat_mesh = np.meshgrid(lon_array, lat_array)
            else:
                lon_mesh, lat_mesh = lon_array, lat_array

            # Proses data
            processed_data, processed_lat, processed_lon = process_wave_data(
                data_array,
                lat_mesh,
                lon_mesh,
                interpolate=interpolate,
                use_kdtree=use_kdtree
            )

            logger.info(f"Variabel {var} diproses dengan {np.isnan(processed_data).sum()} NaNs tersisa.")

            return (var, {
                "description": var_data_dict.get("variable"),
                "units": var_data_dict.get("units"),
                "data": processed_data.tolist(),
                "latitude": processed_lat.tolist(),
                "longitude": processed_lon.tolist(),
            })

        def process_variable_remote(var, filtered_lat, filtered_lon, var_nc_slice):
            # Proses data
            processed_data, processed_lat, processed_lon = process_wave_data(
                var_nc_slice,
                filtered_lat,
                filtered_lon,
                interpolate=interpolate,
                use_kdtree=use_kdtree
            )

            # Coba ambil "long_name" dari var, fallback "N/A"
            units = "N/A"
            try:
                units = dataset.variables[var].getncattr("long_name")
            except:
                pass

            # Simpan data mentah ke local
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
                        "min_lat": min_lat,
                        "max_lat": max_lat,
                        "min_lon": min_lon,
                        "max_lon": max_lon,
                    }
                }
            }
            save_wave_data(raw_response, var, date_str, time_slot)
            logger.info(f"Variabel {var} disimpan ke lokal dan cache.")

            # Setelah menyimpan, gunakan proses lokal untuk mengembalikan data tanpa NaN
            return process_variable_local(var)

        # Implementasi paralel untuk memproses variabel
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            if all_local:
                for var in variables:
                    futures.append(executor.submit(process_variable_local, var))
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
                    # Default bounding Indonesia
                    min_lat = Config.INDONESIA_EXTENT["min_lat"]
                    max_lat = Config.INDONESIA_EXTENT["max_lat"]
                    min_lon = Config.INDONESIA_EXTENT["min_lon"]
                    max_lon = Config.INDONESIA_EXTENT["max_lon"]
                    lat_inds = np.where(
                        (lat_nc >= min_lat) &
                        (lat_nc <= max_lat)
                    )[0]
                    lon_inds = np.where(
                        (lon_nc >= min_lon) &
                        (lon_nc <= max_lon)
                    )[0]

                filtered_lat = lat_nc[lat_inds]
                filtered_lon = lon_nc[lon_inds]

                for var in variables:
                    var_nc = read_nc_variable(dataset, var)
                    # Asumsikan shape (time, lat, lon) => ambil time=0
                    var_data_slice = var_nc[
                        0,
                        lat_inds.min(): lat_inds.max() + 1,
                        lon_inds.min(): lon_inds.max() + 1
                    ]

                    futures.append(executor.submit(
                        process_variable_remote, var, filtered_lat, filtered_lon, var_data_slice
                    ))

            for future in as_completed(futures):
                var, var_response = future.result()
                combined_response["variables"][var] = var_response

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

        # Simpan ke cache
        with open(cache_path, "w") as f:
            json.dump(combined_response, f)
        logger.info(f"Data disimpan ke cache: {cache_path}")

        return {"success": True, "data": combined_response}, 200

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}, 500
