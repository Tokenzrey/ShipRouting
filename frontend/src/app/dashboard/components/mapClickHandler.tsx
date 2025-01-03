import { Feature, Overlay } from 'ol';
import { Coordinate } from 'ol/coordinate';
import { toLonLat } from 'ol/proj';
import { useRouteStore } from '@/lib/GlobalState/state';

// Interface untuk opsi konfigurasi MapClickHandler
interface MapClickHandlerOptions {
  apiKey: string; // API key untuk MapTiler
  onLoadingChange?: (loading: boolean) => void; // Callback saat status loading berubah
  onPopupDataChange?: (data: PopupData | null) => void; // Callback untuk data popup
  overlay: Overlay; // Overlay yang digunakan untuk menampilkan popup
}

// Interface untuk struktur data popup
export interface PopupData {
  placeName?: string; // Nama tempat (jika tersedia)
  latitude?: number; // Latitude koordinat
  longitude?: number; // Longitude koordinat
  waveData?: {
    dirpwsfc: number;
    htsgwsfc: number;
    perpwsfc: number;
  };
}

// Kelas untuk menangani klik pada peta
export class MapClickHandler {
  private apiKey: string; // API key untuk digunakan dalam permintaan API
  private onLoadingChange?: (loading: boolean) => void; // Fungsi callback untuk status loading
  private onPopupDataChange?: (data: PopupData | null) => void; // Fungsi callback untuk data popup
  private overlay: Overlay; // Overlay untuk menampilkan popup pada peta

  /**
   * Constructor untuk MapClickHandler.
   * @param options Opsi konfigurasi MapClickHandler.
   */
  constructor({
    apiKey,
    onLoadingChange,
    onPopupDataChange,
    overlay,
  }: MapClickHandlerOptions) {
    this.apiKey = apiKey;
    this.onLoadingChange = onLoadingChange;
    this.onPopupDataChange = onPopupDataChange;
    this.overlay = overlay;

    // Bind method agar konteks 'this' tetap konsisten
    this.handleClick = this.handleClick.bind(this);
  }

  /**
   * Mengubah status loading dan memanggil callback terkait.
   * @param loading Status loading (true/false).
   */
  private setLoading(loading: boolean) {
    this.onLoadingChange?.(loading);
  }

  /**
   * Melakukan permintaan ke API untuk mendapatkan nama tempat berdasarkan koordinat.
   * @param lon Longitude dari koordinat.
   * @param lat Latitude dari koordinat.
   * @returns Nama tempat (jika tersedia) atau string default.
   */
  private async fetchPlaceName(
    lon: number,
    lat: number,
  ): Promise<string | undefined> {
    try {
      const response = await fetch(
        `https://api.maptiler.com/geocoding/${lon},${lat}.json?key=${this.apiKey}`,
      );
      const data = await response.json();
      return data.features && data.features.length > 0
        ? data.features[0].place_name
        : 'Tidak ada nama';
    } catch (error) {
      console.error('Error fetching place name:', error);
      throw new Error('Gagal mendapatkan nama tempat.');
    }
  }

  private async fetchWaveData(
    lon: number,
    lat: number,
  ): Promise<{ dirpwsfc: number; htsgwsfc: number; perpwsfc: number } | null> {
    try {
      const response = await fetch(
        `http://localhost:5000/api/wave_data_by_coords?lon=${lon}&lat=${lat}`,
      );
      if (!response.ok) {
        console.error('Failed to fetch wave data:', response.statusText);
        return null;
      }
      const data = await response.json();
      console.log(data.data);
      return data.data;
    } catch (error) {
      console.error('Error fetching wave data:', error);
      return null;
    }
  }

  /**
   * Menangani klik pada peta, mendapatkan nama tempat, dan memperbarui popup.
   * @param event Event klik peta yang berisi koordinat.
   */
  public async handleClick(event: {
    coordinate: Coordinate;
    features?: Feature[];
  }) {
    const coordinate = event.coordinate;
    const [lon, lat] = toLonLat(coordinate);

    // Ambil global state untuk lokasi
    const { locationTypeToAdd, addLocation, setLocationTypeToAdd } =
      useRouteStore.getState(); // Akses global state

    // Set status loading ke true
    this.setLoading(true);

    try {
      if (event.features && event.features.length > 0) {
        // Ambil fitur pertama yang diklik
        const feature = event.features[0];
        const properties = feature.getProperties();

        if (properties.category) {
          // Jika marker memiliki kategori (gas/lng)
          const { name } = properties;
          // Ambil koordinat langsung dari geometry properti
          const [markerLon, markerLat] = toLonLat(
            properties.geometry.flatCoordinates,
          );

          const waveData = await this.fetchWaveData(markerLon, markerLat);
          console.log('wavedata: ', waveData);
          console.log('Marker Coordinates:', { markerLon, markerLat });
          // Tambahkan lokasi ke global state
          if (locationTypeToAdd) {
            addLocation({
              type: locationTypeToAdd, // from atau destination
              name: name, // Nama tempat
              longitude: markerLon, // Longitude
              latitude: markerLat, // Latitude
            });
            setLocationTypeToAdd(null);
          }
          // Perbarui data popup dengan data marker
          this.onPopupDataChange?.({
            placeName: name,
            latitude: markerLat,
            longitude: markerLon,
            waveData: waveData || undefined,
          });

          // Tampilkan popup di lokasi marker
          this.overlay.setPosition(coordinate);

          // Tidak perlu melanjutkan ke API geocoding
          return;
        }
      }

      const waveData = await this.fetchWaveData(lon, lat);
      console.log('wavedata: ', waveData);
      // Jika bukan marker, lanjutkan dengan geocoding
      const placeName = await this.fetchPlaceName(lon, lat);

      // Tambahkan lokasi ke global state
      if (locationTypeToAdd) {
        addLocation({
          type: locationTypeToAdd, // from atau destination
          name: placeName || '', // Nama tempat
          longitude: lon, // Longitude
          latitude: lat, // Latitude
        });
        setLocationTypeToAdd(null);
      }

      // Perbarui data popup dengan nama tempat dari geocoding
      this.onPopupDataChange?.({
        placeName,
        latitude: lat,
        longitude: lon,
        waveData: waveData || undefined,
      });

      // Posisikan overlay pada lokasi yang diklik
      this.overlay.setPosition(coordinate);
    } catch (error) {
      // Tampilkan pesan error jika terjadi kesalahan
      alert(error instanceof Error ? error.message : 'Terjadi kesalahan');
    } finally {
      // Set status loading ke false
      this.setLoading(false);
    }
  }

  /**
   * Membersihkan data popup dan menyembunyikan overlay.
   */
  public clearPopup() {
    this.onPopupDataChange?.(null); // Reset data popup ke null
    this.overlay.setPosition(undefined); // Sembunyikan overlay
  }
}

export default MapClickHandler;
