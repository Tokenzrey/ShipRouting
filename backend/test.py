import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Pilih file cache
cache_folder = os.path.join(os.getcwd(), "backend", "data", "cache")

if not os.path.exists(cache_folder):
    print(f"Cache folder {cache_folder} does not exist.")
    exit(1)

cache_files = [f for f in os.listdir(cache_folder) if f.endswith('.json')]
if not cache_files:
    print(f"No cache files found in {cache_folder}.")
    exit(1)

print(f"Available cache files:")
for idx, file in enumerate(cache_files):
    print(f"[{idx}] {file}")

try:
    selected_idx = int(input("Select a cache file index: ").strip())
    selected_cache_file = cache_files[selected_idx]
except (ValueError, IndexError):
    print("Invalid cache file index.")
    exit(1)

cache_file_path = os.path.join(cache_folder, selected_cache_file)

# Step 2: Load cache file JSON
try:
    with open(cache_file_path, 'r') as f:
        combined_response = json.load(f)
except Exception as e:
    print(f"Error loading cache file: {e}")
    exit(1)

# Step 3: Pilih variabel untuk divisualisasikan
variables = list(combined_response.get("variables", {}).keys())
if not variables:
    print("No variables found in the cache file.")
    exit(1)

print(f"Available variables: {', '.join(variables)}")
selected_var = input("Enter variable name: ").strip()

if selected_var not in variables:
    print(f"Invalid variable. Please choose from {', '.join(variables)}")
    exit(1)

# Step 4: Ambil data variabel
try:
    variable_data = combined_response["variables"][selected_var]
    data = np.array(variable_data["data"], dtype=np.float32)
    latitudes = np.array(variable_data["latitude"], dtype=np.float32)
    longitudes = np.array(variable_data["longitude"], dtype=np.float32)
except KeyError as e:
    print(f"Error: Key {e} not found in selected variable.")
    exit(1)
except ValueError as e:
    print(f"Error: Unable to convert data to float. {e}")
    exit(1)

# Debugging tipe data
print(f"Data shape: {data.shape}, dtype: {data.dtype}")
print(f"Latitudes shape: {latitudes.shape}, dtype: {latitudes.dtype}")
print(f"Longitudes shape: {longitudes.shape}, dtype: {longitudes.dtype}")

# Step 5: Bersihkan nilai non-finite
if not np.isfinite(data).all():
    print("Warning: Data contains non-finite values. Replacing with 0.")
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

# Step 6: Buat meshgrid jika perlu
if latitudes.ndim == 1 and longitudes.ndim == 1:
    print("cih meshgrid")
    longitudes, latitudes = np.meshgrid(longitudes, latitudes)

# Debugging dimensi
print(f"Meshgrid shapes - Longitudes: {longitudes.shape}, Latitudes: {latitudes.shape}")

# Step 7: Visualisasi data
plt.figure(figsize=(10, 6))
plt.contourf(longitudes, latitudes, data, cmap='viridis')
plt.colorbar(label=f"{variable_data.get('description', 'Value')} ({variable_data.get('units', '')})")
plt.title(f"{selected_var} Visualization")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
