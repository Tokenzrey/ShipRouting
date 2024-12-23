// src/utils/fetchWaveData.ts

import { Vector as VectorLayer } from 'ol/layer';
import { Vector as VectorSource } from 'ol/source';
import Feature, { FeatureLike } from 'ol/Feature';
import { Polygon, Point, Geometry } from 'ol/geom';
import { fromLonLat } from 'ol/proj';
import { RefObject, MutableRefObject } from 'react';
import { Map } from 'ol';
import Style from 'ol/style/Style';
import Fill from 'ol/style/Fill';
import Stroke from 'ol/style/Stroke';
import CircleStyle from 'ol/style/Circle';
import 'ol/ol.css';

/**
 * Type Definitions
 */
export interface Weather {
  variables: Variables;
  metadata: Metadata;
}

export interface Variables {
  dirpwsfc: DirpwsfcOrHtsgwsfcOrPerpwsfc;
  htsgwsfc: DirpwsfcOrHtsgwsfcOrPerpwsfc;
  perpwsfc: DirpwsfcOrHtsgwsfcOrPerpwsfc;
}

export interface DirpwsfcOrHtsgwsfcOrPerpwsfc {
  description: string;
  units: string;
  data?: number[][] | null;
  latitude?: number[][] | null;
  longitude?: number[][] | null;
}

export interface Metadata {
  dataset_url: string;
  timestamp: string;
  date: string;
  time_slot: string;
  dynamic_extent: boolean;
  processing: Processing;
  extent: Extent;
}

export interface Processing {
  interpolated: boolean;
  use_kdtree: boolean;
}

export interface Extent {
  min_lat: number;
  max_lat: number;
  min_lon: number;
  max_lon: number;
}

interface WaveLayerData {
  latitude: number[][];
  longitude: number[][];
  data: number[][];
}

/**
 * Fetch wave data from API
 */
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

    const latitude = variableData.latitude;
    const longitude = variableData.longitude;

    // Clean data to replace null or undefined values with 0
    const cleanData = variableData.data.map((row) =>
      row.map((value) => (value !== null && value !== undefined ? value : 0)),
    );

    return {
      latitude,
      longitude,
      data: cleanData,
    };
  } catch (error) {
    console.error('Error fetching wave data:', error);
    return null;
  }
};

/**
 * Create grid features for the map
 */
function createGridFeatures(waveData: WaveLayerData): Feature<Geometry>[] {
  const features: Feature<Geometry>[] = [];
  const lat = waveData.latitude;
  const lon = waveData.longitude;
  const data = waveData.data;

  let minValue = Infinity;
  let maxValue = -Infinity;

  // Find min and max values in the data
  for (let i = 0; i < data.length; i++) {
    for (let j = 0; j < data[i].length; j++) {
      const val = data[i][j];
      if (val !== null && val !== undefined) {
        minValue = Math.min(minValue, val);
        maxValue = Math.max(maxValue, val);
      }
    }
  }

  // Create grid polygons
  for (let i = 0; i < data.length - 1; i++) {
    for (let j = 0; j < data[i].length - 1; j++) {
      const value = data[i][j];
      if (value === null || value === undefined) continue;

      const cellCoords = [
        [lon[i][j], lat[i][j]],
        [lon[i][j + 1], lat[i][j + 1]],
        [lon[i + 1][j + 1], lat[i + 1][j + 1]],
        [lon[i + 1][j], lat[i + 1][j]],
        [lon[i][j], lat[i][j]],
      ].map((coord) => fromLonLat(coord));

      const polygon = new Polygon([cellCoords]);
      const feature = new Feature(polygon);
      feature.set('value', value);
      feature.set('minValue', minValue);
      feature.set('maxValue', maxValue);
      features.push(feature);
    }
  }

  return features;
}

/**
 * Style function for coloring grid cells with distinct palettes for each overlay type
 */
function getColorForValue(
  value: number,
  minValue: number,
  maxValue: number,
  overlayType: 'htsgwsfc' | 'perpwsfc' | 'dirpwsfc',
): string {
  // Normalize value to range [0, 1]
  const ratio = (value - minValue) / (maxValue - minValue || 1);

  if (overlayType === 'htsgwsfc') {
    // Red-Yellow-White for 'htsgwsfc'
    const r = Math.round(255 * ratio); // Dominant red for higher values
    const g = Math.round(255 * (1 - Math.pow(1 - ratio, 2))); // Smooth yellow transition
    const b = Math.round(200 * Math.pow(1 - ratio, 2)); // Subtle blue tone for lower values
    return `rgba(${r},${g},${b},0.9)`;
  } else if (overlayType === 'perpwsfc') {
    // Blue-Cyan-Green for 'perpwsfc'
    const r = Math.round(50 * (1 - ratio)); // Low red for muted tones
    const g = Math.round(255 * ratio); // Dominant green for higher values
    const b = Math.round(255 * (1 - Math.pow(1 - ratio, 0.5))); // Bright cyan transition
    return `rgba(${r},${g},${b},0.9)`;
  } else {
    // Default gradient (fallback for 'dirpwsfc' or others)
    const r = Math.round(255 * Math.pow(ratio, 0.7));
    const g = Math.round(255 * (1 - ratio));
    const b = Math.round(255 * ratio);
    return `rgba(${r},${g},${b},0.8)`;
  }
}

/**
 * Style function to dynamically apply color styles based on overlay type
 */
function styleFunction(
  feature: FeatureLike,
  overlayType: 'htsgwsfc' | 'perpwsfc' | 'dirpwsfc',
): Style {
  const value = feature.get('value') as number;
  const minValue = feature.get('minValue') as number;
  const maxValue = feature.get('maxValue') as number;

  const fillColor = getColorForValue(value, minValue, maxValue, overlayType);

  return new Style({
    fill: new Fill({ color: fillColor }),
  });
}

/**
 * Create a grid layer
 */
export const createGridLayer = (
  waveData: WaveLayerData,
  overlayType: 'htsgwsfc' | 'perpwsfc' | 'dirpwsfc',
): VectorLayer<VectorSource<Feature<Geometry>>> => {
  const vectorSource = new VectorSource<Feature<Geometry>>();
  const features = createGridFeatures(waveData);
  vectorSource.addFeatures(features);

  return new VectorLayer({
    source: vectorSource,
    style: (feature) => styleFunction(feature, overlayType), // Pass overlay type for color logic
    zIndex: -1,
  });
};

/**
 * Update grid and point layers
 */
export const updateDynamicGridLayer = async (
  mapInstanceRef: RefObject<Map>,
  waveLayerRef: MutableRefObject<VectorLayer<
    VectorSource<Feature<Geometry>>
  > | null>,
  overlayType: 'htsgwsfc' | 'perpwsfc',
) => {
  try {
    const waveData = await fetchWaveData(overlayType);

    if (!waveData) {
      console.error(`Failed to fetch wave data for ${overlayType}.`);
      return;
    }

    const map = mapInstanceRef.current;
    if (!map) {
      console.error('Map instance not found');
      return;
    }

    if (waveLayerRef.current) {
      map.removeLayer(waveLayerRef.current);
      waveLayerRef.current = null;
    }

    const newLayer = createGridLayer(waveData, overlayType);
    map.addLayer(newLayer);
    waveLayerRef.current = newLayer;

    console.log(
      `Grid and point layers updated successfully for ${overlayType}`,
    );
  } catch (error) {
    console.error('Error updating grid and point layers:', error);
  }
};

/**
 * Schedule periodic updates for wave overlay (opsional)
 */
export const scheduleWaveOverlayUpdate = (
  mapInstanceRef: RefObject<Map>,
  waveLayerRef: MutableRefObject<VectorLayer<
    VectorSource<Feature<Geometry>>
  > | null>,
  overlayType: 'htsgwsfc' | 'perpwsfc',
) => {
  const fetchDataOnSchedule = () => {
    const now = new Date();
    const hours = now.getUTCHours();
    if ([0, 6, 12, 18].includes(hours)) {
      updateDynamicGridLayer(mapInstanceRef, waveLayerRef, overlayType);
    }
  };

  const interval = setInterval(fetchDataOnSchedule, 60 * 60 * 1000); // Check every hour
  return () => clearInterval(interval);
};
