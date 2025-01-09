import { Map as OlMap } from 'ol';
import { Extent } from 'ol/extent';
import { fromLonLat } from 'ol/proj';
import { WAVE_HEIGHT_THRESHOLDS } from './OverlayHandler';
import { getColorForValue } from './OverlayHandler';

// Constants
const RAD_CONVERSION = Math.PI / 180;
const ARROW_SCALE_FACTOR = 0.8;
const BASE_LINE_WIDTH = 2;

// Type Definitions
interface WaveLayerData {
  height: WaveLayerDataPerVariable;
  direction: WaveLayerDataPerVariable;
  period: WaveLayerDataPerVariable;
}

interface WaveLayerDataPerVariable {
  latitude: number[][];
  longitude: number[][];
  data: number[][];
}

interface WavePoint {
  coordinates: [number, number];
  height: number;
  direction: number;
  period: number;
  delay: number; // Randomized delay for animation
}

// Fetch Wave Data
const fetchWaveData = async (): Promise<WaveLayerData | null> => {
  try {
    const response = await fetch('http://localhost:5000/api/wave_data');
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const jsonData: { success: boolean; data: any } = await response.json();
    if (!jsonData.success) {
      console.error('Failed to fetch wave data.');
      return null;
    }

    const variables = jsonData.data.variables;

    if (
      !variables['htsgwsfc'] ||
      !variables['dirpwsfc'] ||
      !variables['perpwsfc'] ||
      !variables['htsgwsfc'].latitude ||
      !variables['htsgwsfc'].longitude ||
      !variables['dirpwsfc'].latitude ||
      !variables['dirpwsfc'].longitude ||
      !variables['perpwsfc'].latitude ||
      !variables['perpwsfc'].longitude ||
      !variables['htsgwsfc'].data ||
      !variables['dirpwsfc'].data ||
      !variables['perpwsfc'].data
    ) {
      console.error('Wave data is incomplete or missing.');
      return null;
    }

    return {
      height: {
        latitude: variables['htsgwsfc'].latitude,
        longitude: variables['htsgwsfc'].longitude,
        data: variables['htsgwsfc'].data,
      },
      direction: {
        latitude: variables['dirpwsfc'].latitude,
        longitude: variables['dirpwsfc'].longitude,
        data: variables['dirpwsfc'].data,
      },
      period: {
        latitude: variables['perpwsfc'].latitude,
        longitude: variables['perpwsfc'].longitude,
        data: variables['perpwsfc'].data,
      },
    };
  } catch (error) {
    console.error('Error fetching wave data:', error);
    return null;
  }
};

// Get Color Based on Wave Height (High Contrast Colors)
const getColorByHeight = (height: number): string => {
  if (height < WAVE_HEIGHT_THRESHOLDS.CALM) {
    return 'rgba(176, 224, 230, 0.9)'; // Bright blue for calm waves
  }
  if (height < WAVE_HEIGHT_THRESHOLDS.MODERATE) {
    return 'rgba(255, 128, 0, 0.9)'; // Bright orange for moderate waves
  }
  if (height < WAVE_HEIGHT_THRESHOLDS.HIGH) {
    return 'rgba(255, 0, 128, 0.9)'; // Bright pink for high waves
  }
  return 'rgba(128, 0, 255, 0.9)'; // Bright purple for very high waves
};

// Determine Delay Based on Wave Height
const getDelayForHeight = (height: number): number => {
  if (height < WAVE_HEIGHT_THRESHOLDS.CALM) {
    return 2 + Math.random() * 3.5;
  }
  if (height < WAVE_HEIGHT_THRESHOLDS.MODERATE) {
    return 1.5 + Math.random() * 3;
  }
  if (height < WAVE_HEIGHT_THRESHOLDS.HIGH) {
    return 1 + Math.random() * 2.5;
  }
  return Math.random() * 1;
};

const getLineWidthByPeriod = (period: number): number => {
  // Normalize period (example: assume min=2, max=10)
  const minPeriod = 2;
  const maxPeriod = 8;
  const normalized = (period - minPeriod) / (maxPeriod - minPeriod);
  return 1 + normalized * 3; // Line width ranges from 1 to 4
};

// Create Canvas Overlay for Animations
const createCanvasOverlay = (
  map: OlMap,
  data: WavePoint[],
  getColor?: (val: number) => string,
) => {
  const targetElement = map.getTargetElement() as HTMLElement;
  const canvas = document.createElement('canvas');
  canvas.className = 'canvas-wave';
  canvas.style.position = 'absolute';
  canvas.style.top = '0';
  canvas.style.left = '0';
  canvas.style.width = '100%';
  canvas.style.height = '100%';
  canvas.style.pointerEvents = 'none';
  canvas.width = targetElement.offsetWidth;
  canvas.height = targetElement.offsetHeight;
  targetElement.appendChild(canvas);

  const context = canvas.getContext('2d')!;
  let animationFrameId: number;
  const animationDuration = 1200;
  let startTime = performance.now();
  let isInteracting = false;
  let renderTimeout: number;

  const getVisiblePoints = () => {
    const extent = map.getView().calculateExtent(map.getSize()) as Extent;
    const buffer = 0.5; // 50% buffer around viewport
    const [minX, minY, maxX, maxY] = extent;
    const width = maxX - minX;
    const height = maxY - minY;

    return data.filter((point) => {
      const [x, y] = fromLonLat(point.coordinates);
      return (
        x >= minX - width * buffer &&
        x <= maxX + width * buffer &&
        y >= minY - height * buffer &&
        y <= maxY + height * buffer
      );
    });
  };

  const updateCanvas = () => {
    if (isInteracting) return;

    context.clearRect(0, 0, canvas.width, canvas.height);
    const currentTime = performance.now();
    const visiblePoints = getVisiblePoints();

    visiblePoints.forEach((point) => {
      const elapsedTime =
        (currentTime - startTime - point.delay * 1000) % animationDuration;
      if (elapsedTime < 0) return;

      const progress = elapsedTime / animationDuration;
      const [x1, y1] = map.getPixelFromCoordinate(
        fromLonLat(point.coordinates),
      );
      const angle = (point.direction + 90) * (Math.PI / 180);

      const arrowLength = Math.max(20, Math.min(point.height * 10 * 0.8, 90));
      const x2End = x1 + Math.cos(angle) * point.height * arrowLength;
      const y2End = y1 + Math.sin(angle) * point.height * arrowLength;

      const x2 = x1 + progress * (x2End - x1);
      const y2 = y1 + progress * (y2End - y1);

      context.beginPath();
      context.moveTo(x1, y1);
      context.lineTo(x2, y2);
      context.strokeStyle = getColorByHeight(point.height);
      context.lineWidth = getLineWidthByPeriod(point.period);
      context.stroke();
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
    renderTimeout = window.setTimeout(() => {
      animateArrows();
    }, 200);
  };

  const animateArrows = () => {
    if (!isInteracting) {
      updateCanvas();
      animationFrameId = requestAnimationFrame(animateArrows);
    }
  };

  map.on('movestart', handleInteractionStart);
  map.on('moveend', handleInteractionEnd);
  animateArrows();

  return () => {
    cancelAnimationFrame(animationFrameId);
    map.un('movestart', handleInteractionStart);
    map.un('moveend', handleInteractionEnd);
    clearTimeout(renderTimeout);
    canvas.remove();
  };
};

// Add Wave Layer to Map
export const addWaveLayerToMap = async (
  map: OlMap,
  animationEnabled: boolean,
  previousCleanup: (() => void) | null,
  setWaveRangeAnim?: (range: { min: number; max: number } | null) => void,
): Promise<() => void | null> => {
  if (previousCleanup) {
    previousCleanup();
  }

  if (!animationEnabled) {
    return () => {};
  }

  try {
    const response = await fetchWaveData();

    if (!response) {
      console.error('No wave data available. Fetch returned null.');
      return () => {};
    }

    const wavePoints: WavePoint[] = [];
    const { height, direction, period } = response;

    for (let i = 0; i < height.latitude.length; i++) {
      for (let j = 0; j < height.latitude[i].length; j++) {
        wavePoints.push({
          coordinates: [height.longitude[i][j], height.latitude[i][j]],
          height: height.data[i][j],
          direction: direction.data[i][j],
          period: period.data[i][j],
          delay: getDelayForHeight(height.data[i][j]), // Assign delay based on height
        });
      }
    }

    const heights = wavePoints.map((wp) => wp.height);
    const minVal = Math.min(...heights);
    const maxVal = Math.max(...heights);

    if (setWaveRangeAnim) {
      setWaveRangeAnim({ min: minVal, max: maxVal });
    }

    return createCanvasOverlay(map, wavePoints, (val) =>
      getColorForValue(val, minVal, maxVal, 'htsgwsfc'),
    );
  } catch (error) {
    console.error('Error adding wave layer to map:', error);
    return () => {};
  }
};
