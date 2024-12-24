import { Map as OlMap } from 'ol';
import { fromLonLat } from 'ol/proj';

// Constants
const RAD_CONVERSION = Math.PI / 180;
const ARROW_SCALE_FACTOR = 0.8;
const BASE_LINE_WIDTH = 2;
const WAVE_HEIGHT_THRESHOLDS = {
  CALM: 0.5,
  MODERATE: 1.5,
  HIGH: 2.5,
};

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
    return 'rgba(0, 128, 255, 0.9)'; // Bright blue for calm waves
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
    return 3 + Math.random() * 4.5;
  }
  if (height < WAVE_HEIGHT_THRESHOLDS.MODERATE) {
    return 2 + Math.random() * 3.5;
  }
  if (height < WAVE_HEIGHT_THRESHOLDS.HIGH) {
    return 1 + Math.random() * 2.5;
  }
  return Math.random() * 1.5;
};

// Create Canvas Overlay for Animations
const createCanvasOverlay = (map: OlMap, data: WavePoint[]) => {
  const targetElement = map.getTargetElement() as HTMLElement;
  const existingCanvas = targetElement.querySelector(
    '.canvas-wave',
  ) as HTMLCanvasElement;

  if (existingCanvas) {
    existingCanvas.remove();
  }

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
  const animationDuration = 1200; // Duration of one animation loop in ms
  let startTime = performance.now();

  const updateCanvas = () => {
    context.clearRect(0, 0, canvas.width, canvas.height);

    const currentTime = performance.now();

    data.forEach((point) => {
      const delay = point.delay * 1000; // Convert delay to ms
      const elapsedTime = (currentTime - startTime - delay) % animationDuration;
      if (elapsedTime < 0) return; // Skip if animation is in delay period

      const progress = elapsedTime / animationDuration; // Value between 0 and 1

      const [x1, y1] = map.getPixelFromCoordinate(
        fromLonLat(point.coordinates),
      );
      const angle = (point.direction + 90) * RAD_CONVERSION;

      const x2Start = x1;
      const y2Start = y1;

      const arrowLength = Math.max(
        20,
        Math.min(point.height * 10 * ARROW_SCALE_FACTOR, 90),
      );

      const x2End = x1 + Math.cos(angle) * point.height * arrowLength;
      const y2End = y1 + Math.sin(angle) * point.height * arrowLength;

      // Animate arrow position
      const x2 = x2Start + progress * (x2End - x2Start);
      const y2 = y2Start + progress * (y2End - y2Start);

      context.beginPath();
      context.moveTo(x1, y1);
      context.lineTo(x2, y2);
      context.strokeStyle = getColorByHeight(point.height);
      context.lineWidth = BASE_LINE_WIDTH;
      context.stroke();
    });
  };

  const animateArrows = () => {
    updateCanvas();
    animationFrameId = requestAnimationFrame(animateArrows);
  };

  map.on('moveend', updateCanvas);
  animateArrows();

  return () => {
    cancelAnimationFrame(animationFrameId);
    map.un('moveend', updateCanvas);
    canvas.remove();
  };
};

// Add Wave Layer to Map
export const addWaveLayerToMap = async (
  map: OlMap,
  animationEnabled: boolean,
  previousCleanup: (() => void) | null,
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

    return createCanvasOverlay(map, wavePoints);
  } catch (error) {
    console.error('Error adding wave layer to map:', error);
    return () => {};
  }
};
