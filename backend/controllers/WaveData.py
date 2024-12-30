import logging
from flask import jsonify, request
from managers import get_wave_data_response_interpolate

# Konfigurasi Logging
logging.basicConfig(
    level=logging.DEBUG,  # Ubah ke DEBUG untuk informasi detail
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def get_wave_data(wave_data_locator, route_optimizer):
    """
    Controller untuk mendapatkan data gelombang.
    """
    try:
        # Log parameter request
        logger.debug(f"Menerima request dengan parameter: {request.args}")

        # Validasi input
        if not wave_data_locator:
            raise ValueError("WaveDataLocator tidak tersedia.")
        if not route_optimizer:
            raise ValueError("RouteOptimizer tidak tersedia.")

        # Mengambil respons dari manajer
        response, status_code = get_wave_data_response_interpolate(request.args, wave_data_locator, route_optimizer)
        
        # Log hasil respon
        logger.debug(f"Respon berhasil dikembalikan dengan status code: {status_code}")
        return jsonify(response), status_code

    except ValueError as ve:
        logger.warning(f"Kesalahan validasi: {ve}")
        return jsonify({"success": False, "error": str(ve)}), 400

    except Exception as e:
        logger.error(f"Error tidak terduga dalam controller get_wave_data: {e}")
        return jsonify({"success": False, "error": "Internal server error."}), 500
