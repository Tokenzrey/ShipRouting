from pickle import NONE
from flask import Flask, jsonify
from flask_cors import CORS
from controllers import get_wave_data, djikstra_route
from constants import (
    DATA_DIR_HTSGWSFC,
    DATA_DIR_DIRPWSFC,
    DATA_DIR_PERPWSFC,
    DATA_DIR_CACHE,
    DATA_DIR,
    Config
)
from managers import GridLocator, RouteOptimizer
from managers.WaveDataLocator import WaveDataLocator
import os
import json
import numpy as np
import logging

# Buat logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # Menampilkan log di terminal
    ]
)

logger = logging.getLogger(__name__)

# Buat direktori jika belum ada
for directory in [DATA_DIR, DATA_DIR_HTSGWSFC, DATA_DIR_DIRPWSFC, DATA_DIR_PERPWSFC, DATA_DIR_CACHE]:
    os.makedirs(directory, exist_ok=True)

# Global instances
grid_locator = None
wave_data_locator = None
route_optimizer = None

# Variable untuk menyimpan file wave data terakhir
last_wave_file = None
last_wave_file_mtime = None

def initialize_global_instances():
    global grid_locator, wave_data_locator, route_optimizer

    try:
        # Inisialisasi WaveDataLocator
        logger.info("Inisialisasi WaveDataLocator...")
        wave_files = [f for f in os.listdir(DATA_DIR_CACHE) if f.endswith('.json')]
        if not wave_files:
            raise FileNotFoundError("Tidak ada file wave data ditemukan di folder cache.")

        latest_wave_file = max(wave_files, key=lambda f: os.path.getmtime(os.path.join(DATA_DIR_CACHE, f)))
        logger.info(f"Memuat wave data terbaru dari {latest_wave_file}...")
        with open(os.path.join(DATA_DIR_CACHE, latest_wave_file), 'r') as f:
            wave_data = json.load(f)
            wave_data_locator = WaveDataLocator(wave_data)

        logger.info("Inisialisasi Route Optimizer...")
        route_optimizer = RouteOptimizer(
            graph_file=Config.GRAPH_FILE,
            wave_data_locator=wave_data_locator,
            model_path=Config.MODEL_PATH,
            input_scaler_pkl=Config.INPUT_SCALER,
            output_scaler_pkl=Config.OUTPUT_SCALER,
            grid_locator=NONE
        )

        logger.info("Inisialisasi Grid Locator...")
        # Setelah RouteOptimizer diinisialisasi, kita perlu inisialisasi GridLocator dengan koordinat graf
        graph_coords = np.array([coords for coords in route_optimizer.graph["nodes"].values()])
        grid_locator = GridLocator(graph_coords)
        route_optimizer.grid_locator = grid_locator  # Set GridLocator dalam RouteOptimizer

        logger.info("Inisialisasi global instances selesai.")
    except Exception as e:
        logger.error(f"Error saat inisialisasi global instances: {e}")
        raise

# Jalankan inisialisasi global instances saat aplikasi mulai
initialize_global_instances()

app = Flask(__name__)
CORS(app)

@app.before_request
def refresh_wave_data_locator():
    """
    Fungsi ini memuat ulang WaveDataLocator jika file wave data berubah.
    """
    global wave_data_locator, last_wave_file, last_wave_file_mtime

    try:
        # Cari semua file JSON di folder cache
        wave_files = [f for f in os.listdir(DATA_DIR_CACHE) if f.endswith('.json')]
        if not wave_files:
            logger.error("Tidak ada file wave data di folder cache.")
            return

        # Temukan file terbaru berdasarkan waktu modifikasi
        latest_wave_file = max(
            wave_files, key=lambda f: os.path.getmtime(os.path.join(DATA_DIR_CACHE, f))
        )
        latest_wave_file_path = os.path.join(DATA_DIR_CACHE, latest_wave_file)
        latest_wave_file_mtime = os.path.getmtime(latest_wave_file_path)

        # Periksa apakah file terbaru berbeda dari file terakhir
        if last_wave_file == latest_wave_file and last_wave_file_mtime == latest_wave_file_mtime:
            logger.info("Wave data tidak berubah, tidak perlu diperbarui.")
            return

        # Muat data dari file terbaru
        with open(latest_wave_file_path, 'r') as f:
            wave_data = json.load(f)
            wave_data_locator = WaveDataLocator(wave_data)
            route_optimizer.update_wave_data_locator(wave_data_locator)
            logger.info(f"WaveDataLocator diperbarui dengan file {latest_wave_file}.")

        # Simpan file terakhir dan waktu modifikasi terakhir
        last_wave_file = latest_wave_file
        last_wave_file_mtime = latest_wave_file_mtime

    except Exception as e:
        logger.error(f"Error saat memperbarui WaveDataLocator: {e}")

@app.errorhandler(Exception)
def handle_global_error(e):
    """
    Handler global untuk semua exception.
    """
    logger.error(f"Terjadi error: {e}")
    return jsonify({"success": False, "error": str(e)}), 500

# Tambahkan endpoint dengan menggunakan instance global
app.add_url_rule('/api/wave_data', 'get_wave_data', lambda: get_wave_data(wave_data_locator, route_optimizer), methods=['GET'])
app.add_url_rule('/api/djikstra', 'djikstra_route', lambda: djikstra_route(grid_locator, route_optimizer), methods=['POST'])

if __name__ == '__main__':
    app.run(
        debug=os.getenv('FLASK_DEBUG', 'false').lower() == 'true',
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000))
    )
