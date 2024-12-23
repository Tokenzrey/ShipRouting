from flask import Flask
from flask_cors import CORS
from controllers import get_wave_data
from constants import DATA_DIR_HTSGWSFC, DATA_DIR_DIRPWSFC, DATA_DIR_PERPWSFC, DATA_DIR_CACHE, DATA_DIR
import os

# Buat direktori jika belum ada
for directory in [DATA_DIR, DATA_DIR_HTSGWSFC, DATA_DIR_DIRPWSFC, DATA_DIR_PERPWSFC, DATA_DIR_CACHE]:
    os.makedirs(directory, exist_ok=True)

app = Flask(__name__)
CORS(app)

app.add_url_rule('/api/wave_data', 'get_wave_data', get_wave_data, methods=['GET'])

if __name__ == '__main__':
    app.run(
        debug=os.getenv('FLASK_DEBUG', 'false').lower() == 'true',
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000))
    )
