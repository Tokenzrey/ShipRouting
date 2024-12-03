import { Vector as VectorLayer } from 'ol/layer';
import { Vector as VectorSource } from 'ol/source';
import { Feature } from 'ol';
import { LineString } from 'ol/geom';
import { Stroke, Style } from 'ol/style';

// Interface untuk konfigurasi pembuatan layer grid
interface CreateGridLayerOptions {
  spacingDegrees?: number; // Jarak antar garis dalam derajat (default: 5)
  strokeColor?: string; // Warna garis (default: rgba(200, 200, 200, 0.5))
  strokeWidth?: number; // Ketebalan garis (default: 1)
}

// Kelas untuk membuat dan mengelola layer grid
export class GridLayer {
  private static instance: VectorSource | null = null; // Cache untuk sumber data grid

  /**
   * Membuat atau mengambil layer grid dengan opsi konfigurasi.
   * @param options Opsi konfigurasi pembuatan grid.
   * @returns VectorLayer berisi garis lintang dan bujur.
   */
  static create({
    spacingDegrees = 5,
    strokeColor = 'rgba(200, 200, 200, 0.5)',
    strokeWidth = 1,
  }: CreateGridLayerOptions = {}): VectorLayer<VectorSource> {
    // Kembalikan instance yang di-cache jika sudah ada
    if (this.instance) {
      console.log('Grid loaded from cache');
      return new VectorLayer({
        source: this.instance,
        style: new Style({
          stroke: new Stroke({
            color: strokeColor,
            width: strokeWidth,
          }),
        }),
      });
    }

    // Jika cache kosong, buat instance baru
    const vectorSource = new VectorSource();

    // Buat garis bujur (longitude)
    for (let lon = -180; lon <= 180; lon += spacingDegrees) {
      const line = new LineString([
        [lon, -90],
        [lon, 90],
      ]).transform('EPSG:4326', 'EPSG:3857'); // Transformasi ke proyeksi EPSG:3857
      vectorSource.addFeature(new Feature(line)); // Tambahkan fitur ke sumber data
    }

    // Buat garis lintang (latitude)
    for (let lat = -90; lat <= 90; lat += spacingDegrees) {
      const line = new LineString([
        [-180, lat],
        [180, lat],
      ]).transform('EPSG:4326', 'EPSG:3857'); // Transformasi ke proyeksi EPSG:3857
      vectorSource.addFeature(new Feature(line)); // Tambahkan fitur ke sumber data
    }

    // Simpan sumber data ke cache
    this.instance = vectorSource;

    // Kembalikan layer grid baru
    return new VectorLayer({
      source: vectorSource,
      style: new Style({
        stroke: new Stroke({
          color: strokeColor,
          width: strokeWidth,
        }),
      }),
    });
  }

  /**
   * Menghapus cache sumber data grid.
   * Berguna jika ingin membuat ulang grid dengan konfigurasi baru.
   */
  static clearCache(): void {
    this.instance = null;
  }
}

export default GridLayer;
