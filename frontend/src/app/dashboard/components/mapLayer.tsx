import VectorSource from 'ol/source/Vector';
import VectorLayer from 'ol/layer/Vector';
import { Style, Icon, Text, Fill, Stroke } from 'ol/style';
import { Feature } from 'ol';
import { Point, LineString } from 'ol/geom';
import { fromLonLat } from 'ol/proj';
import { markerData } from '@/contents/markerData';
import { useRouteStore } from '@/lib/GlobalState/state';

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

/**
 * Membuat layer marker dari data markerData.
 * @returns VectorLayer berisi marker sesuai kategori.
 */
export const createMarkerLayer = () => {
  const vectorSource = new VectorSource();

  Object.entries(markerData).forEach(([category, markers]) => {
    markers.forEach(({ name, coord }) => {
      const markerFeature = new Feature({
        geometry: new Point(fromLonLat(coord)),
        name,
        category,
      });

      const iconSrc =
        category === 'LNG' ? '/images/lng.png' : '/images/gas.png';
      const colorIcon = category === 'LNG' ? '#ff0404' : '#ffac04';
      markerFeature.setStyle(
        new Style({
          image: new Icon({
            src: iconSrc,
            scale: 1.3,
          }),
          text: new Text({
            text: name,
            font: 'bold 14px Arial', // Font teks lebih besar
            fill: new Fill({ color: colorIcon }), // Warna teks putih
            stroke: new Stroke({ color: '#000', width: 3 }), // Outline hitam untuk kontras
            offsetY: -25, // Geser teks ke atas marker
          }),
        }),
      );

      vectorSource.addFeature(markerFeature);
    });
  });

  return new VectorLayer({
    source: vectorSource,
    zIndex: 1200,
  });
};

export const createLocationMarkers = (): VectorLayer => {
  const vectorSource = new VectorSource();

  // Ambil lokasi `from` dan `destination` dari global state
  const { locations } = useRouteStore.getState();

  // Tambahkan marker untuk setiap lokasi
  locations.forEach((location) => {
    const markerFeature = new Feature({
      geometry: new Point(fromLonLat([location.longitude, location.latitude])),
      name: location.name,
      category: location.type, // Tipe marker
    });

    markerFeature.setStyle(
      new Style({
        image: new Icon({
          src: '/images/pin.png', // Ikon marker
          scale: 1.8,
        }),
        text: new Text({
          text: location.name,
          font: 'bold 14px Arial', // Font teks
          fill: new Fill({ color: '#00ccff' }), // Warna teks
          stroke: new Stroke({ color: '#000', width: 3 }), // Outline hitam
          offsetY: -25, // Geser teks ke atas marker
        }),
      }),
    );

    vectorSource.addFeature(markerFeature);
  });

  return new VectorLayer({
    source: vectorSource,
    zIndex: 1200, // Prioritas di atas layer lainnya
  });
};
