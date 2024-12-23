import os

class Config:
    BASE_URL = "http://nomads.ncep.noaa.gov/dods/wave/gfswave/"
    TIME_SLOTS = ["18z", "12z", "06z", "00z"]
    INDONESIA_EXTENT = {
        "min_lat": -15.0,
        "max_lat": 10.0,
        "min_lon": 92.0,
        "max_lon": 141.0,
    }
    CACHE_TTL = 3600  # Cache selama 1 jam
    CHUNK_SIZE_LAT = 500
    CHUNK_SIZE_LON = 500

# Pastikan direktori data ada
DATA_DIR = os.path.join(os.getcwd(), 'data')

# Direktori khusus untuk setiap variabel
DATA_DIR_HTSGWSFC = os.path.join(DATA_DIR, 'htsgwsfc')
DATA_DIR_DIRPWSFC = os.path.join(DATA_DIR, 'dirpwsfc')
DATA_DIR_PERPWSFC = os.path.join(DATA_DIR, 'perpwsfc')
DATA_DIR_CACHE = os.path.join(DATA_DIR, 'cache')