import os
import logging
from flask import Flask, jsonify, request
from flask_cors import CORS
import netCDF4 as nc
import numpy as np
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import json

# Konfigurasi Logging
logging.basicConfig(
    level=logging.DEBUG,  # Ubah ke DEBUG untuk informasi detail
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Aktifkan CORS untuk semua route

# Pastikan direktori data ada
DATA_DIR = os.path.join(os.getcwd(), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

class Config:
    BASE_URL = "http://nomads.ncep.noaa.gov/dods/wave/gfswave/"
    TIME_SLOTS = ["18z", "12z", "06z", "00z"]
    INDONESIA_EXTENT = {
        "min_lat": -11.0,
        "max_lat": 6.0,
        "min_lon": 95.0,
        "max_lon": 141.0,
    }
    CACHE_TTL = 3600  # Cache selama 1 jam

def get_latest_dataset_url():
    """
    Tentukan URL dataset terbaru dengan caching.
    Periksa juga apakah dataset sudah ada di folder `/data`.
    """
    now = datetime.utcnow()
    logger.info("DEBUG: Memulai pencarian dataset terbaru...")
    
    for _ in range(4):  # Periksa hingga 4 hari sebelumnya
        date_str = now.strftime("%Y%m%d")
        for time_slot in Config.TIME_SLOTS:
            # Cek apakah file lokal sudah ada
            filename = f"{date_str}-{time_slot}.json"
            filepath = os.path.abspath(os.path.join(DATA_DIR, filename))
            print(filepath)
            if os.path.exists(filepath):
                logger.info(f"Dataset ditemukan di lokal: {filepath}")
                return None, date_str, time_slot, filepath

            # Jika file lokal tidak ada, periksa di URL
            dataset_url = f"{Config.BASE_URL}{date_str}/gfswave.global.0p16_{time_slot}"
            try:
                with nc.Dataset(dataset_url) as ds:
                    logger.info(f"Dataset ditemukan di URL: {dataset_url}")
                    return dataset_url, date_str, time_slot, None
            except Exception as e:
                logger.warning(f"Dataset tidak tersedia: {dataset_url} ({e})")
        
        # Mundur ke hari sebelumnya
        now -= timedelta(days=1)

    # Jika tidak ada dataset yang valid
    raise ValueError("Tidak ada dataset yang valid dalam 4 hari terakhir!")

def save_wave_data(data, date_str, time_slot):
    """
    Simpan data gelombang ke dalam file JSON terstruktur
    """
    filename = f"{date_str}-{time_slot}.json"
    filepath = os.path.join(DATA_DIR, filename)
    
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Data berhasil disimpan: {filepath}")
        print(f"DEBUG: Data disimpan ke file - {filepath}")
    except Exception as e:
        logger.error(f"Gagal menyimpan data: {e}")
        print(f"ERROR: Gagal menyimpan data ke file - {e}")

@app.route('/api/wave_data', methods=['GET'])
def get_wave_data():
    try:
        include_full_grid = request.args.get('full_grid', 'false').lower() == 'true'
        
        # Dapatkan URL dataset dan informasi tanggal
        dataset_url, date_str, time_slot, local_filepath = get_latest_dataset_url()
        
        if local_filepath:
            # Jika file lokal ada, gunakan file lokal
            logger.info(f"DEBUG: Menggunakan file lokal: {local_filepath}")
            with open(local_filepath, 'r') as f:
                response = json.load(f)
            logger.info(f"DEBUG: Data dari lokal berhasil dimuat.")
        else:
            # Akses dataset dari URL
            logger.info(f"DEBUG: Mengakses dataset dari URL: {dataset_url}")
            dataset = nc.Dataset(dataset_url)
            
            

            lat = dataset.variables["lat"][:]
            lon = dataset.variables["lon"][:]
            htsgwsfc = dataset.variables["htsgwsfc"]
            
            units = htsgwsfc.getncattr("long_name") if "long_name" in htsgwsfc.ncattrs() else "N/A"
            
            lat_indices = np.where(
                (lat >= Config.INDONESIA_EXTENT["min_lat"]) & 
                (lat <= Config.INDONESIA_EXTENT["max_lat"])
            )[0]
            lon_indices = np.where(
                (lon >= Config.INDONESIA_EXTENT["min_lon"]) & 
                (lon <= Config.INDONESIA_EXTENT["max_lon"])
            )[0]

            filtered_lat = lat[lat_indices]
            filtered_lon = lon[lon_indices]
            filtered_data = htsgwsfc[0, 
                lat_indices.min():lat_indices.max() + 1, 
                lon_indices.min():lon_indices.max() + 1
            ]

            response = {
                "variable": "Significant Wave Height (htsgwsfc)",
                "units": units,
                "data": filtered_data.tolist(),
                "latitude": filtered_lat.tolist(),
                "longitude": filtered_lon.tolist(),
                "metadata": {
                    "dataset_url": dataset_url,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "date": date_str,
                    "time_slot": time_slot
                }
            }

            # Simpan data ke file
            save_wave_data(response, date_str, time_slot)

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error dalam mendapatkan data gelombang: {e}")
        return jsonify({
            "error": "Gagal mengambil data gelombang",
            "details": str(e)
        }), 500
    
@app.route('/api/data_history', methods=['GET'])
def get_data_history():
    """
    Endpoint untuk melihat riwayat data yang tersimpan
    """
    try:
        # Ambil daftar file JSON di direktori data
        print("DEBUG: Mengambil riwayat file data dari direktori.")
        data_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json')]
        
        # Urutkan file berdasarkan tanggal (terbaru dulu)
        sorted_files = sorted(data_files, reverse=True)
        print(f"DEBUG: Ditemukan {len(sorted_files)} file data.")

        return jsonify({
            "total_files": len(sorted_files),
            "files": sorted_files
        })
    except Exception as e:
        print(f"ERROR: Gagal mengambil riwayat data - {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(
        debug=os.getenv('FLASK_DEBUG', 'false').lower() == 'true', 
        host='0.0.0.0', 
        port=int(os.getenv('PORT', 5000))
    )
