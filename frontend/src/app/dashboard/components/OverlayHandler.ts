import { Map as OlMap } from 'ol';
import { Extent } from 'ol/extent';
import { fromLonLat } from 'ol/proj';

// Constants
export const WAVE_HEIGHT_THRESHOLDS = {
  CALM: 0.5,
  MODERATE: 1.5,
  HIGH: 2.5,
};

export const WAVE_PERIOD_THRESHOLDS = {
  LOW: 2,
  MEDIUM: 5,
  HIGH: 8,
};

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

export type OverlayType = 'htsgwsfc' | 'perpwsfc' | 'none';

// Fetch Wave Data
export const fetchWaveData = async (
  type: 'htsgwsfc' | 'perpwsfc',
  date: Date,
  isCurrentDate: boolean,
  selectedTime: '00' | '06' | '12' | '18',
): Promise<WaveLayerData | null> => {
  try {
    const dateStr = date.toISOString().split('T')[0].replace(/-/g, '');
    console.log(dateStr);
    const response = await fetch(
      `http://localhost:5000/api/wave_data?date=${dateStr}&time_slot=${selectedTime}&currentdate=${isCurrentDate}`,
    );

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
export function getColorForValue(
  value: number,
  minValue: number,
  maxValue: number,
  overlayType: OverlayType,
): string {
  if (overlayType === 'none' || maxValue <= minValue) {
    // fallback: misal warna abu
    return 'rgba(128, 128, 128, 0.8)';
  }
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
  // Warna default jika tipe overlay tidak dikenali
  return 'rgba(128, 128, 128, 0.8)';
}

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

  let isInteracting = false;
  let renderTimeout: number;

  const getVisibleCells = () => {
    const extent = map.getView().calculateExtent(map.getSize()) as Extent;
    const [minX, minY, maxX, maxY] = extent;
    const buffer = 0.3;
    const width = maxX - minX;
    const height = maxY - minY;

    const visibleCells: [number, number][] = [];

    for (let i = 0; i < data.length - 1; i++) {
      for (let j = 0; j < data[i].length - 1; j++) {
        const cellCoord = fromLonLat([longitude[i][j], latitude[i][j]]);
        if (
          cellCoord[0] >= minX - width * buffer &&
          cellCoord[0] <= maxX + width * buffer &&
          cellCoord[1] >= minY - height * buffer &&
          cellCoord[1] <= maxY + height * buffer
        ) {
          visibleCells.push([i, j]);
        }
      }
    }

    return visibleCells;
  };

  const renderVisibleGrid = () => {
    if (!map.getSize()) return;

    canvas.width = map.getSize()![0];
    canvas.height = map.getSize()![1];
    context.clearRect(0, 0, canvas.width, canvas.height);

    if (isInteracting) return;

    const visibleCells = getVisibleCells();

    visibleCells.forEach(([i, j]) => {
      const value = data[i][j];
      if (value === null || value === undefined) return;

      const corners = [
        [longitude[i][j], latitude[i][j]],
        [longitude[i + 1][j], latitude[i + 1][j]],
        [longitude[i + 1][j + 1], latitude[i + 1][j + 1]],
        [longitude[i][j + 1], latitude[i][j + 1]],
      ];

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
    });
  };

  const handleInteractionStart = () => {
    isInteracting = true;
    canvas.style.display = 'none';
  };

  const handleInteractionEnd = () => {
    isInteracting = false;
    canvas.style.display = 'block';
    clearTimeout(renderTimeout);
    renderTimeout = window.setTimeout(renderVisibleGrid, 200);
  };

  map.on('movestart', handleInteractionStart);
  map.on('moveend', handleInteractionEnd);

  renderVisibleGrid();

  return () => {
    map.un('movestart', handleInteractionStart);
    map.un('moveend', handleInteractionEnd);
    clearTimeout(renderTimeout);
    canvas.remove();
  };
}

// Update Dynamic Grid Layer
export async function updateDynamicGridLayer(
  map: OlMap,
  overlayType: 'htsgwsfc' | 'perpwsfc' | 'none',
  date: Date,
  isCurrentDate: boolean,
  selectedTime: '00' | '06' | '12' | '18',
  previousCleanup: (() => void) | null,
): Promise<{
  cleanup: () => void;
  minValue: number;
  maxValue: number;
} | null> {
  if (previousCleanup) {
    previousCleanup();
  }

  // If none, remove the overlay and return null
  if (!overlayType || overlayType === 'none') {
    const canvas = map.getTargetElement()?.querySelector('.canvas-overlay');
    if (canvas) {
      canvas.remove();
    }
    return null;
  }

  // fetch wave data
  const waveData = await fetchWaveData(
    overlayType,
    date,
    isCurrentDate,
    selectedTime,
  );
  if (!waveData) {
    console.error(`Failed to fetch wave data for ${overlayType}`);
    return null;
  }

  // Flatten to get min and max
  const values = waveData.data.flat();
  const minVal = Math.min(...values);
  const maxVal = Math.max(...values);

  // render ke canvas
  const cleanup = renderGridToCanvas(map, waveData, overlayType);

  return {
    cleanup,
    minValue: minVal,
    maxValue: maxVal,
  };
}

// Schedule Wave Overlay Update
export const scheduleWaveOverlayUpdate = (
  map: OlMap,
  overlayType: 'htsgwsfc' | 'perpwsfc' | 'none',
  date: Date,
  isCurrentDate: boolean,
  selectedTime: '00' | '06' | '12' | '18',
  previousCleanup: (() => void) | null,
) => {
  const fetchDataOnSchedule = () => {
    const now = new Date();
    const hours = now.getUTCHours();
    if ([0, 6, 12, 18].includes(hours)) {
      updateDynamicGridLayer(
        map,
        overlayType,
        date,
        isCurrentDate,
        selectedTime,
        previousCleanup,
      );
    }
  };

  const interval = setInterval(fetchDataOnSchedule, 60 * 60 * 1000); // Every hour
  return () => clearInterval(interval);
};
