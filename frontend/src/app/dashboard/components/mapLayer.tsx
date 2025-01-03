import VectorSource from 'ol/source/Vector';
import VectorLayer from 'ol/layer/Vector';
import { Style, Icon, Text, Fill, Stroke } from 'ol/style';
import { Feature } from 'ol';
import { Point, LineString } from 'ol/geom';
import { fromLonLat } from 'ol/proj';
import { markerData } from '@/contents/markerData';
import { useRouteStore } from '@/lib/GlobalState/state';
import { toRadians, toDegrees } from 'ol/math';

// Interface untuk konfigurasi pembuatan layer grid
interface CreateGridLayerOptions {
  spacingDegrees?: number; // Jarak antar garis dalam derajat (default: 5)
  strokeColor?: string; // Warna garis (default: rgba(200, 200, 200, 0.5))
  strokeWidth?: number; // Ketebalan garis (default: 1)
}

// Interface untuk setiap titik dalam path
interface PathPoint {
  Heave: number;
  Pitch: number;
  Roll: number;
  coordinates: [number, number]; // [longitude, latitude]
  dirpwfsfc: number;
  htsgwsfc: number;
  node_id: string;
  perpwsfc: number;
  rel_heading: number;
}

export class GridLayer {
  private static instance: VectorSource | null = null;

  static create({
    spacingDegrees = 5,
    strokeColor = 'rgba(200, 200, 200, 0.5)',
    strokeWidth = 1,
  }: CreateGridLayerOptions = {}): VectorLayer<VectorSource> {
    if (this.instance) {
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

    const vectorSource = new VectorSource();

    // Menambahkan garis longitudinal
    for (let lon = -180; lon <= 180; lon += spacingDegrees) {
      const line = new LineString([
        [lon, -90],
        [lon, 90],
      ]).transform('EPSG:4326', 'EPSG:3857');
      vectorSource.addFeature(new Feature(line));
    }

    // Menambahkan garis latitudinal
    for (let lat = -90; lat <= 90; lat += spacingDegrees) {
      const line = new LineString([
        [-180, lat],
        [180, lat],
      ]).transform('EPSG:4326', 'EPSG:3857');
      vectorSource.addFeature(new Feature(line));
    }

    this.instance = vectorSource;

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

  static clearCache(): void {
    this.instance = null;
  }
}

export const createMarkerLayer = (): VectorLayer<VectorSource> => {
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
            font: 'bold 14px Arial',
            fill: new Fill({ color: colorIcon }),
            stroke: new Stroke({ color: '#000', width: 3 }),
            offsetY: -25,
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
  const { locations } = useRouteStore.getState();

  locations.forEach((location) => {
    const markerFeature = new Feature({
      geometry: new Point(fromLonLat([location.longitude, location.latitude])),
      name: location.name,
      category: location.type,
    });

    markerFeature.setStyle(
      new Style({
        image: new Icon({
          src: '/images/pin.png',
          scale: 1.8,
        }),
        text: new Text({
          text: location.name,
          font: 'bold 14px Arial',
          fill: new Fill({ color: '#00ccff' }),
          stroke: new Stroke({ color: '#000', width: 3 }),
          offsetY: -25,
        }),
      }),
    );

    vectorSource.addFeature(markerFeature);
  });

  return new VectorLayer({
    source: vectorSource,
    zIndex: 1200,
  });
};

// Create Vector Sources untuk optimal dan safest routes
const optimalSource = new VectorSource();
const safestSource = new VectorSource();

// Create Route Layers
export const createOptimalRouteLayer = (): VectorLayer<VectorSource> => {
  return new VectorLayer({
    source: optimalSource,
    style: new Style({
      stroke: new Stroke({
        color: '#00ff00', // Warna hijau untuk optimal route
        width: 4,
      }),
    }),
    zIndex: 1500,
  });
};

export const createSafestRouteLayer = (): VectorLayer<VectorSource> => {
  return new VectorLayer({
    source: safestSource,
    style: new Style({
      stroke: new Stroke({
        color: '#ffcc00', // Warna kuning untuk safest route
        width: 4,
      }),
    }),
    zIndex: 1500,
  });
};

// Fungsi untuk mensinkronisasi route layers dengan store
export const syncRouteLayers = () => {
  const store = useRouteStore.getState();
  console.log('syncRouteLayers dipanggil');

  const hasFrom = store.locations.some((loc) => loc.type === 'from');
  const hasDestination = store.locations.some(
    (loc) => loc.type === 'destination',
  );

  // Jika salah satu "from" atau "destination" hilang, bersihkan semua route layers
  if (!hasFrom || !hasDestination) {
    console.log('Missing "from" or "destination", clearing routes');
    optimalSource.clear();
    safestSource.clear();
    return;
  }

  // Handle Optimal Route
  if (store.optimalRoute && store.optimalRoute.length > 0) {
    const optimalCoordinates = store.optimalRoute
      .filter(
        (point: PathPoint) =>
          Array.isArray(point.coordinates) && point.coordinates.length === 2,
      )
      .map((point: PathPoint) => fromLonLat(point.coordinates));

    if (optimalCoordinates.length > 0) {
      const optimalFeature = new Feature({
        geometry: new LineString(optimalCoordinates),
      });

      optimalSource.clear();
      optimalSource.addFeature(optimalFeature);
      console.log('Optimal Route Added to Map');
    } else {
      console.warn('Optimal Route memiliki koordinat yang tidak valid.');
      optimalSource.clear();
    }
  } else {
    optimalSource.clear();
    console.log('Optimal Route Cleared from Map');
  }

  // Handle Safest Route
  if (store.safestRoute && store.safestRoute.length > 0) {
    const safestCoordinates = store.safestRoute
      .filter(
        (point: PathPoint) =>
          Array.isArray(point.coordinates) && point.coordinates.length === 2,
      )
      .map((point: PathPoint) => fromLonLat(point.coordinates));

    if (safestCoordinates.length > 0) {
      const safestFeature = new Feature({
        geometry: new LineString(safestCoordinates),
      });

      safestSource.clear();
      safestSource.addFeature(safestFeature);
      console.log('Safest Route Added to Map');
    } else {
      console.warn('Safest Route memiliki koordinat yang tidak valid.');
      safestSource.clear();
    }
  } else {
    safestSource.clear();
    console.log('Safest Route Cleared from Map');
  }
};

// Inisialisasi sinkronisasi pada perubahan store
export const initializeRouteLayerSync = () => {
  console.log('initializeRouteLayerSync dipanggil');

  useRouteStore.subscribe((state) => {
    syncRouteLayers();
  });
  syncRouteLayers();
};

export const calculateBearing = (
  from: [number, number],
  to: [number, number],
): number => {
  const [lon1, lat1] = from.map(toRadians);
  const [lon2, lat2] = to.map(toRadians);

  const deltaLon = lon2 - lon1;
  const y = Math.sin(deltaLon) * Math.cos(lat2);
  const x =
    Math.cos(lat1) * Math.sin(lat2) -
    Math.sin(lat1) * Math.cos(lat2) * Math.cos(deltaLon);

  const bearing = Math.atan2(y, x);
  return ((toDegrees(bearing) + 360) % 360) - 90; // Normalize to 0-360 degrees
};

// Create a vector source and layer for the ship
const shipSource = new VectorSource();
export const createShipLayer = (): VectorLayer<VectorSource> => {
  return new VectorLayer({
    source: shipSource,
    style: new Style({
      image: new Icon({
        src: '/images/ship.png', // Path to the ship icon
        scale: 1.5,
      }),
    }),
    zIndex: 2007, // Ensure it's above other layers
  });
};

// Function to update ship position on the map
export const updateShipPosition = (
  coordinates: [number, number],
  nextCoordinates?: [number, number], // Optional next coordinates for direction
) => {
  const shipFeature = new Feature({
    geometry: new Point(fromLonLat(coordinates)),
  });

  let rotation = 0; // Default rotation

  // Calculate rotation if the nextCoordinates are provided
  if (nextCoordinates) {
    rotation = calculateBearing(coordinates, nextCoordinates) + 90;
  }

  const shipStyle = new Style({
    image: new Icon({
      src: '/images/ship.png', // Path to the ship icon
      scale: 2,
      rotation: (rotation * Math.PI) / 180, // Convert degrees to radians
    }),
  });

  shipFeature.setStyle(shipStyle);

  shipSource.clear(); // Clear the previous position
  shipSource.addFeature(shipFeature); // Add the new position
};
