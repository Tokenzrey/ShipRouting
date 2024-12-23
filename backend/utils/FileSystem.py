import os
import json
import logging
from constants import DATA_DIR_HTSGWSFC, DATA_DIR_DIRPWSFC, DATA_DIR_PERPWSFC

# Konfigurasi Logging
logging.basicConfig(
    level=logging.DEBUG,  # Ubah ke DEBUG untuk informasi detail
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def get_file_path(variable, date_str, time_slot):
    if variable == 'htsgwsfc':
        dir_path = DATA_DIR_HTSGWSFC
    elif variable == 'dirpwsfc':
        dir_path = DATA_DIR_DIRPWSFC
    elif variable == 'perpwsfc':
        dir_path = DATA_DIR_PERPWSFC
    else:
        raise ValueError("Variabel tidak dikenal.")
    filename = f"{date_str}-{time_slot}.json"
    return os.path.join(dir_path, filename)

def local_file_exists_for_all(date_str, time_slot):
    """
    Cek apakah semua file untuk ketiga variabel sudah ada di lokal.
    """
    htsgwsfc_file = get_file_path('htsgwsfc', date_str, time_slot)
    dirpwsfc_file = get_file_path('dirpwsfc', date_str, time_slot)
    perpwsfc_file = get_file_path('perpwsfc', date_str, time_slot)

    return (os.path.exists(htsgwsfc_file) and
            os.path.exists(dirpwsfc_file) and
            os.path.exists(perpwsfc_file))

def save_wave_data(data, variable, date_str, time_slot):
    """
    Simpan data gelombang per variabel ke dalam file JSON terstruktur.
    """
    filepath = get_file_path(variable, date_str, time_slot)
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Data {variable} berhasil disimpan: {filepath}")
    except Exception as e:
        logger.error(f"Gagal menyimpan data {variable}: {e}")

def load_local_data(variable,date_str, time_slot):
    """
    Membaca data lokal berdasarkan variabel.
    """
    filepath = get_file_path(variable, date_str, time_slot)
    with open(filepath, 'r') as f:
        return json.load(f)