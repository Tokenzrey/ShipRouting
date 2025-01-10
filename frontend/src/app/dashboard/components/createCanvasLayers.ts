// src/app/dashboard/components/createCanvasLayers.ts

import { Map as OlMap } from 'ol';
import { fromLonLat, toLonLat } from 'ol/proj';
import { Coordinate } from 'ol/coordinate';
import {
  PathPoint,
  BlockedEdge,
  Keyframes,
  useRouteStore,
} from '@/lib/GlobalState/state';
import axios from 'axios';
import { debounce } from 'lodash';

// Constants
const EDGE_BATCH_DELAY = 100;
const PARTIAL_PATH_DELAY = 100;
const FINAL_PATH_DELAY = 1000;
const MAX_EDGES = 100000;

/**
 * Utility function to create a Canvas element for visualization layers.
 */
const createCanvasElement = (
  map: OlMap,
  className: string,
): HTMLCanvasElement => {
  const mapSize = map.getSize();
  if (!mapSize) throw new Error('Map size is undefined');

  const targetElement = map.getTargetElement();
  if (!targetElement) throw new Error('Map target element is undefined');

  let existingCanvas = targetElement.querySelector(
    `canvas.${className}`,
  ) as HTMLCanvasElement;

  if (existingCanvas) {
    existingCanvas.remove();
  }

  const canvas = document.createElement('canvas');
  canvas.className = className;
  Object.assign(canvas.style, {
    position: 'absolute',
    top: '0',
    left: '0',
    width: '100%',
    height: '100%',
    pointerEvents: 'none',
    zIndex: '3000',
  });

  canvas.width = mapSize[0];
  canvas.height = mapSize[1];
  targetElement.appendChild(canvas);

  return canvas;
};

/**
 * Function to draw a blocked or normal edge.
 */
const drawBlockedEdge = (
  map: OlMap,
  context: CanvasRenderingContext2D,
  edge: BlockedEdge,
) => {
  if (!edge.source_coords || !edge.target_coords) {
    console.error('Edge coordinates are missing:', edge);
    return;
  }

  const sourcePixel = map.getPixelFromCoordinate(
    fromLonLat(edge.source_coords),
  );
  const targetPixel = map.getPixelFromCoordinate(
    fromLonLat(edge.target_coords),
  );
  if (!sourcePixel || !targetPixel) return;

  context.beginPath();
  context.moveTo(sourcePixel[0], sourcePixel[1]);
  context.lineTo(targetPixel[0], targetPixel[1]);

  context.strokeStyle = edge.isBlocked ? '#FF0000' : '#00FF00';
  context.lineWidth = edge.isBlocked ? 6 : 2;
  context.setLineDash(edge.isBlocked ? [10, 10] : []);
  context.stroke();
};

/**
 * Create Partial Path Layer
 */
export const createPartialPathCanvas = (map: OlMap): HTMLCanvasElement => {
  return createCanvasElement(map, 'partial-path-canvas');
};

/**
 * Create Final Path Layer
 */
export const createFinalPathCanvas = (map: OlMap): HTMLCanvasElement => {
  return createCanvasElement(map, 'final-path-canvas');
};

/**
 * Create All Edges Layer
 */
export const createAllEdgesCanvas = (map: OlMap): HTMLCanvasElement => {
  return createCanvasElement(map, 'all-edges-canvas');
};

/**
 * Initialize Canvas Layers
 */
export const initializeCanvasLayers = (
  map: OlMap,
  shipSpeed: number,
  loadCondition: string,
) => {
  createPartialPathCanvas(map);
  createFinalPathCanvas(map);
  createAllEdgesCanvas(map);

  fetchAndDrawBlockedEdges(map, shipSpeed, loadCondition);

  const debouncedFetch = debounce(() => {
    fetchAndDrawBlockedEdges(map, shipSpeed, loadCondition);
  }, 500);

  map.on('moveend', debouncedFetch);
};

/**
 * Function to fetch blocked edges and draw them on canvas.
 */
export const fetchAndDrawBlockedEdges = async (
  map: OlMap,
  shipSpeed: number,
  loadCondition: string,
) => {
  const allEdgesCanvas = document.querySelector(
    'canvas.all-edges-canvas',
  ) as HTMLCanvasElement;
  const allEdgesContext = allEdgesCanvas?.getContext('2d');
  if (!allEdgesContext) return;

  const mapSize = map.getSize();
  if (!mapSize) return;

  const view = map.getView();
  const extent = view.calculateExtent(mapSize);
  const [minLon, minLat] = toLonLat([extent[0], extent[1]]);
  const [maxLon, maxLat] = toLonLat([extent[2], extent[3]]);

  console.log('Fetching blocked edges in view:', {
    minLon,
    minLat,
    maxLon,
    maxLat,
    shipSpeed,
    loadCondition,
  });

  try {
    const response = await axios.post(
      'http://localhost:5000/get_blocked_edges_in_view',
      {
        view_bounds: [minLon, minLat, maxLon, maxLat],
        ship_speed: shipSpeed,
        condition: loadCondition === 'ballast' ? 1 : 0,
      },
    );

    let blockedEdges: BlockedEdge[] = response.data.blocked_edges;
    blockedEdges = blockedEdges.slice(0, MAX_EDGES);

    console.log(`Received ${blockedEdges.length} blocked edges`);

    allEdgesContext.clearRect(
      0,
      0,
      allEdgesCanvas.width,
      allEdgesCanvas.height,
    );

    blockedEdges.forEach((edge, index) => {
      console.log(`Drawing Edge ${index + 1}/${blockedEdges.length}`, edge);
      drawBlockedEdge(map, allEdgesContext, edge);
    });
  } catch (error) {
    console.error('Error fetching blocked edges:', error);
  }
};

// Utility function to validate path point
const isValidPathPoint = (point: any): point is PathPoint => {
  return (
    point &&
    Array.isArray(point.coordinates) &&
    point.coordinates.length === 2 &&
    typeof point.coordinates[0] === 'number' &&
    typeof point.coordinates[1] === 'number'
  );
};

/**
 * Function to animate keyframes and render paths.
 */
export const animateKeyframes = (
  map: OlMap,
  keyframes: Keyframes,
): (() => void) | null => {
  console.log('Starting animation with keyframes:', keyframes);

  const partialPathCanvas = document.querySelector(
    'canvas.partial-path-canvas',
  ) as HTMLCanvasElement;
  const finalPathCanvas = document.querySelector(
    'canvas.final-path-canvas',
  ) as HTMLCanvasElement;

  if (!partialPathCanvas || !finalPathCanvas) {
    console.error('Canvas elements not found');
    return null;
  }

  const partialPathContext = partialPathCanvas.getContext('2d');
  const finalPathContext = finalPathCanvas.getContext('2d');

  if (!partialPathContext || !finalPathContext) {
    console.error('Canvas contexts are undefined');
    return null;
  }

  // Validate partial path data
  const partialPath = Array.isArray(keyframes.partial_path)
    ? keyframes.partial_path
    : [];
  if (partialPath.length === 0) {
    console.warn('Partial path is empty or invalid');
  }

  console.log(`Total Partial Path Points: ${partialPath.length}`);
  console.log(
    `Total Final Path Points: ${keyframes.final_path?.path?.length || 0}`,
  );

  partialPathContext.clearRect(
    0,
    0,
    partialPathCanvas.width,
    partialPathCanvas.height,
  );
  finalPathContext.clearRect(
    0,
    0,
    finalPathCanvas.width,
    finalPathCanvas.height,
  );

  useRouteStore.getState().setIsCalculating(true);
  let startTime = performance.now();

  let currentPartialIndex = 0;

  const drawPartialPathSequentially = () => {
    if (currentPartialIndex >= partialPath.length) {
      console.log(
        `Partial path animation completed in ${(performance.now() - startTime).toFixed(2)}ms`,
      );
      setTimeout(drawFinalPath, FINAL_PATH_DELAY);
      return;
    }

    const point = partialPath[currentPartialIndex];

    // Validate point before using it
    if (!isValidPathPoint(point)) {
      console.error('Invalid path point:', point);
      currentPartialIndex++;
      setTimeout(drawPartialPathSequentially, PARTIAL_PATH_DELAY);
      return;
    }

    try {
      const coord = fromLonLat(point.coordinates);
      const pixel = map.getPixelFromCoordinate(coord);

      if (pixel) {
        partialPathContext.beginPath();
        partialPathContext.arc(pixel[0], pixel[1], 3, 0, 2 * Math.PI);
        partialPathContext.fillStyle = '#FFA500';
        partialPathContext.fill();
      }

      currentPartialIndex++;
      setTimeout(drawPartialPathSequentially, PARTIAL_PATH_DELAY);
    } catch (error) {
      console.error('Error drawing path point:', error);
      currentPartialIndex++;
      setTimeout(drawPartialPathSequentially, PARTIAL_PATH_DELAY);
    }
  };

  const drawFinalPath = () => {
    console.log('Drawing Final Path...');
    const finalPath = keyframes.final_path?.path || [];

    if (finalPath.length < 2) {
      console.warn('Final path has insufficient points');
      useRouteStore.getState().setIsCalculating(false);
      return;
    }

    try {
      finalPathContext.beginPath();

      // Validate and draw first point
      if (!isValidPathPoint(finalPath[0])) {
        throw new Error('Invalid first point in final path');
      }

      const firstCoord = fromLonLat(finalPath[0].coordinates);
      const firstPixel = map.getPixelFromCoordinate(firstCoord);
      if (!firstPixel) {
        throw new Error('Could not get pixel coordinates for first point');
      }

      finalPathContext.moveTo(firstPixel[0], firstPixel[1]);

      // Draw lines to subsequent points
      for (let i = 1; i < finalPath.length; i++) {
        if (!isValidPathPoint(finalPath[i])) {
          console.warn(`Skipping invalid point at index ${i}`);
          continue;
        }

        const coord = fromLonLat(finalPath[i].coordinates);
        const pixel = map.getPixelFromCoordinate(coord);
        if (!pixel) continue;

        finalPathContext.lineTo(pixel[0], pixel[1]);
      }

      finalPathContext.strokeStyle = '#0000FF';
      finalPathContext.lineWidth = 4;
      finalPathContext.stroke();
    } catch (error) {
      console.error('Error drawing final path:', error);
    } finally {
      useRouteStore.getState().setIsCalculating(false);
    }
  };

  drawPartialPathSequentially();
  return () => {};
};
