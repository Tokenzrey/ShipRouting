import { Feature, Overlay } from 'ol';
import { Coordinate } from 'ol/coordinate';
import { toLonLat } from 'ol/proj';

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

          // Perbarui data popup dengan data marker
          this.onPopupDataChange?.({
            placeName: name,
            latitude: lat,
            longitude: lon,
          });

          // Tampilkan popup di lokasi marker
          this.overlay.setPosition(coordinate);

          // Tidak perlu melanjutkan ke API geocoding
          return;
        }
      }

      // Jika bukan marker, lanjutkan dengan geocoding
      const placeName = await this.fetchPlaceName(lon, lat);

      // Perbarui data popup dengan nama tempat dari geocoding
      this.onPopupDataChange?.({
        placeName,
        latitude: lat,
        longitude: lon,
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
