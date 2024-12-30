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

    for (let lon = -180; lon <= 180; lon += spacingDegrees) {
      const line = new LineString([
        [lon, -90],
        [lon, 90],
      ]).transform('EPSG:4326', 'EPSG:3857');
      vectorSource.addFeature(new Feature(line));
    }

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

export const createMarkerLayer = (): VectorLayer => {
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

// Create Vector Sources
const optimalSource = new VectorSource();
const safestSource = new VectorSource();

// Create Route Layers
export const createOptimalRouteLayer = (): VectorLayer<VectorSource> => {
  return new VectorLayer({
    source: optimalSource,
    style: new Style({
      stroke: new Stroke({
        color: '#00ff00', // Green color for optimal route
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
        color: '#ffcc00', // Yellow color for safest route
        width: 4,
      }),
    }),
    zIndex: 1500,
  });
};

// Function to synchronize route layers with store
export const syncRouteLayers = () => {
  const store = useRouteStore.getState();

  const hasFrom = store.locations.some((loc) => loc.type === 'from');
  const hasDestination = store.locations.some(
    (loc) => loc.type === 'destination',
  );

  // If either "from" or "destination" is missing, clear all route layers
  if (!hasFrom || !hasDestination) {
    optimalSource.clear();
    safestSource.clear();
    return;
  }

  // Handle Optimal Route
  if (store.optimalRoute && store.optimalRoute.length > 0) {
    const optimalCoordinates = store.optimalRoute.map(
      (coord: [number, number]) => fromLonLat(coord),
    );

    const optimalFeature = new Feature({
      geometry: new LineString(optimalCoordinates),
    });

    optimalSource.clear();
    optimalSource.addFeature(optimalFeature);
  } else {
    optimalSource.clear();
  }

  // Handle Safest Route
  if (store.safestRoute && store.safestRoute.length > 0) {
    const safestCoordinates = store.safestRoute.map((coord: [number, number]) =>
      fromLonLat(coord),
    );

    const safestFeature = new Feature({
      geometry: new LineString(safestCoordinates),
    });

    safestSource.clear();
    safestSource.addFeature(safestFeature);
  } else {
    safestSource.clear();
  }
};

// Initialize synchronization on store changes
export const initializeRouteLayerSync = () => {
  useRouteStore.subscribe((state) => {
    const hasFrom = state.locations.some((loc) => loc.type === 'from');
    const hasDestination = state.locations.some(
      (loc) => loc.type === 'destination',
    );

    // If either "from" or "destination" is missing, clear all route layers
    if (!hasFrom || !hasDestination) {
      optimalSource.clear();
      safestSource.clear();
      return;
    }

    // Sync Optimal Route
    if (state.optimalRoute && state.optimalRoute.length > 0) {
      const optimalCoordinates = state.optimalRoute.map(
        (coord: [number, number]) => fromLonLat(coord),
      );
      const optimalFeature = new Feature({
        geometry: new LineString(optimalCoordinates),
      });
      optimalSource.clear();
      optimalSource.addFeature(optimalFeature);
    } else {
      optimalSource.clear();
    }

    // Sync Safest Route
    if (state.safestRoute && state.safestRoute.length > 0) {
      const safestCoordinates = state.safestRoute.map(
        (coord: [number, number]) => fromLonLat(coord),
      );
      const safestFeature = new Feature({
        geometry: new LineString(safestCoordinates),
      });
      safestSource.clear();
      safestSource.addFeature(safestFeature);
    } else {
      safestSource.clear();
    }
  });
};
