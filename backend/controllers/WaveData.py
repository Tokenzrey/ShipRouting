import logging
from flask import jsonify, request
from managers import get_wave_data_response_interpolate

# Konfigurasi Logging
logging.basicConfig(
    level=logging.DEBUG,  # Ubah ke DEBUG untuk informasi detail
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def get_wave_data():
    """
    Controller untuk mendapatkan data gelombang.
    """
    try:
        # Mengambil respons dari manajer
        response, status_code = get_wave_data_response_interpolate(request.args)
        return jsonify(response), status_code

    except Exception as e:
        logger.error(f"Error dalam controller: {e}")
        return jsonify({"success": False, "error": str(e)}), 500