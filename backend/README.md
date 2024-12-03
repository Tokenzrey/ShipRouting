# Wave Data API for Indonesian Marine Conditions

## Project Overview

A Flask-based API that retrieves wave height data for the Indonesian maritime region using NOAA's Global Wave Model (GFS Wave) dataset.

### Key Features

- Automatically find the most recent wave dataset
- Filter data specifically for Indonesian waters
- Flexible API with optional parameters
- Robust error handling and logging

### Technologies

- Python
- Flask
- netCDF4
- NumPy

## Prerequisites

### System Requirements

- Python 3.8+
- pip

### Installation

1. Clone the repository

```bash
git clone <your-repository-url>
cd <repository-directory>
```

2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies

```bash
pip install flask flask-cors netCDF4 numpy
```

## Configuration

### Environment Variables

- `WAVE_DATA_BASE_URL`: Custom base URL for wave datasets (optional)
- `FLASK_DEBUG`: Enable debug mode (true/false)
- `PORT`: Custom server port (default: 5000)

## Running the API

### Development Mode

```bash
# Standard run
python app.py

# With custom configuration
FLASK_DEBUG=true PORT=8000 python app.py
```

## API Endpoint

### GET `/api/wave_data`

#### Parameters

- `full_grid` (optional):
  - When set to `true`, returns complete latitude and longitude grid
  - Default: `false`

#### Response

```json
{
    "variable": "Significant Wave Height",
    "units": "meters",
    "data": [[wave_height_values]],
    "latitude": [latitude_values],
    "longitude": [longitude_values],
    "metadata": {
        "dataset_url": "...",
        "timestamp": "..."
    }
}
```

#### Example Request

```bash
# Basic request
curl http://localhost:5000/api/wave_data

# Request with full grid
curl "http://localhost:5000/api/wave_data?full_grid=true"
```

## Coverage Area

- **Latitude**: -11.0° to 6.0°
- **Longitude**: 95.0° to 141.0°

## Error Handling

- Comprehensive logging
- Detailed error responses
- Automatic dataset fallback mechanism

## Deployment Considerations

- Use a production WSGI server like Gunicorn
- Set up proper environment variables
- Configure CORS as needed

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Create a Pull Request

## Troubleshooting

- Ensure internet connection for dataset retrieval
- Check network access to NOAA servers
- Verify Python and library versions

## License

[Specify your license]

## Contact

[Your contact information]
