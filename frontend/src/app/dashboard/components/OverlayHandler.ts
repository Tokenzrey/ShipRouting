import { Map as OlMap } from 'ol';
import { fromLonLat } from 'ol/proj';

// Constants
const WAVE_HEIGHT_THRESHOLDS = {
  CALM: 0.5,
  MODERATE: 1.5,
  HIGH: 2.5,
};

const BATCH_SIZE = 500; // Number of cells processed per animation frame

// Type Definitions
interface WaveLayerData {
  latitude: number[][];
  longitude: number[][];
  data: number[][];
}

interface Weather {
  variables: Variables;
  metadata: Metadata;
}

interface Variables {
  dirpwsfc: WaveVariable;
  htsgwsfc: WaveVariable;
  perpwsfc: WaveVariable;
}

interface WaveVariable {
  data: number[][] | null;
  latitude: number[][] | null;
  longitude: number[][] | null;
}

interface Metadata {
  dataset_url: string;
  timestamp: string;
}

// Fetch Wave Data
export const fetchWaveData = async (
  type: 'htsgwsfc' | 'perpwsfc',
): Promise<WaveLayerData | null> => {
  try {
    const response = await fetch('http://localhost:5000/api/wave_data');
    const jsonData: { success: boolean; data: Weather } = await response.json();

    if (!jsonData.success || !jsonData.data.variables[type]) {
      console.error(`Failed to fetch ${type} data.`);
      return null;
    }

    const variableData = jsonData.data.variables[type];

    if (
      !variableData.latitude ||
      !variableData.longitude ||
      !variableData.data
    ) {
      console.error(`Invalid data structure for ${type}.`);
      return null;
    }

    return {
      latitude: variableData.latitude,
      longitude: variableData.longitude,
      data: variableData.data.map((row) =>
        row.map((value) => (value !== null && value !== undefined ? value : 0)),
      ),
    };
  } catch (error) {
    console.error('Error fetching wave data:', error);
    return null;
  }
};

// Get Color for Value
function getColorForValue(
  value: number,
  minValue: number,
  maxValue: number,
  overlayType: 'htsgwsfc' | 'perpwsfc',
): string {
  const ratio = (value - minValue) / (maxValue - minValue || 1);

  if (overlayType === 'htsgwsfc') {
    // Dark Red -> Dark Yellow -> Dark Orange
    const r = Math.round(120 + 100 * ratio); // Base red with a darker tone
    const g = Math.round(80 + 100 * (1 - ratio)); // Yellowish gradient
    const b = Math.round(50 + 50 * (1 - ratio)); // Subtle blue to keep tone calm
    return `rgba(${r},${g},${b},0.8)`; // Slightly transparent for overlay effect
  } else if (overlayType === 'perpwsfc') {
    // Dark Blue -> Teal -> Dark Green
    const r = Math.round(30 * (1 - ratio)); // Minimal red for calmness
    const g = Math.round(100 + 120 * ratio); // Gradient towards green
    const b = Math.round(120 + 50 * (1 - Math.pow(1 - ratio, 0.5))); // Blue fades slightly
    return `rgba(${r},${g},${b},0.8)`; // Slightly transparent for overlay effect
  } else {
    // Default gradient (calm purples)
    const r = Math.round(100 + 100 * Math.pow(ratio, 0.7)); // Purple gradient
    const g = Math.round(50 + 50 * (1 - ratio)); // Muted greenish-blue
    const b = Math.round(150 + 50 * ratio); // Slightly brighter purple
    return `rgba(${r},${g},${b},0.8)`; // Slightly transparent for overlay effect
  }
}

// Render Grid to Canvas
function renderGridToCanvas(
  map: OlMap,
  waveData: WaveLayerData,
  overlayType: 'htsgwsfc' | 'perpwsfc',
) {
  const targetElement = map.getTargetElement() as HTMLElement;

  let canvas = targetElement.querySelector(
    '.canvas-overlay',
  ) as HTMLCanvasElement;
  if (!canvas) {
    canvas = document.createElement('canvas');
    canvas.className = 'canvas-overlay';
    canvas.style.position = 'absolute';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.pointerEvents = 'none';
    targetElement.appendChild(canvas);
  }

  const context = canvas.getContext('2d')!;
  const { latitude, longitude, data } = waveData;

  const values = data.flat();
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  let animationFrameId: number;
  const updateCanvas = () => {
    const mapSize = map.getSize();
    if (!mapSize) return;

    canvas.width = mapSize[0];
    canvas.height = mapSize[1];
    context.clearRect(0, 0, canvas.width, canvas.height);

    for (let i = 0; i < data.length - 1; i++) {
      for (let j = 0; j < data[i].length - 1; j++) {
        const value = data[i][j];
        if (value === null || value === undefined) continue;

        // Get the corner coordinates of the cell
        const corners = [
          [longitude[i][j], latitude[i][j]], // Top-left
          [longitude[i + 1][j], latitude[i + 1][j]], // Bottom-left
          [longitude[i + 1][j + 1], latitude[i + 1][j + 1]], // Bottom-right
          [longitude[i][j + 1], latitude[i][j + 1]], // Top-right
        ];

        // Transform coordinates to pixels
        const screenCorners = corners.map((corner) =>
          map.getPixelFromCoordinate(fromLonLat(corner)),
        );

        if (screenCorners.every((corner) => corner !== null)) {
          context.beginPath();
          context.moveTo(screenCorners[0]![0], screenCorners[0]![1]);
          for (let k = 1; k < screenCorners.length; k++) {
            context.lineTo(screenCorners[k]![0], screenCorners[k]![1]);
          }
          context.closePath();
          context.fillStyle = getColorForValue(
            value,
            minValue,
            maxValue,
            overlayType,
          );
          context.fill();
        }
      }
    }
  };

  const updateContinuous = () => {
    updateCanvas();
    animationFrameId = requestAnimationFrame(updateContinuous);
  };

  map.on('moveend', updateCanvas);
  updateContinuous();

  return () => {
    cancelAnimationFrame(animationFrameId);
    map.un('moveend', updateCanvas);
    canvas.remove();
  };
}

// Update Dynamic Grid Layer
export const updateDynamicGridLayer = async (
  map: OlMap,
  overlayType: 'htsgwsfc' | 'perpwsfc' | 'none',
  previousCleanup: (() => void) | null,
): Promise<() => void | null> => {
  if (previousCleanup) {
    previousCleanup();
  }

  if (!overlayType || overlayType === 'none') {
    const canvas = map.getTargetElement()?.querySelector('.canvas-overlay');
    if (canvas) {
      canvas.remove();
    }
    return () => {};
  }

  const waveData = await fetchWaveData(overlayType);
  if (!waveData) {
    console.error(`Failed to fetch wave data for ${overlayType}`);
    return () => {};
  }

  return renderGridToCanvas(map, waveData, overlayType);
};

// Schedule Wave Overlay Update
export const scheduleWaveOverlayUpdate = (
  map: OlMap,
  overlayType: 'htsgwsfc' | 'perpwsfc' | 'none',
  previousCleanup: (() => void) | null,
) => {
  const fetchDataOnSchedule = () => {
    const now = new Date();
    const hours = now.getUTCHours();
    if ([0, 6, 12, 18].includes(hours)) {
      updateDynamicGridLayer(map, overlayType, previousCleanup);
    }
  };

  const interval = setInterval(fetchDataOnSchedule, 60 * 60 * 1000); // Every hour
  return () => clearInterval(interval);
};
