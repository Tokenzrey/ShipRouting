// src/app/dashboard/components/createCanvasLayers.ts

import { Map as OlMap, View } from 'ol';
import { fromLonLat, toLonLat } from 'ol/proj';
import {
  PathPoint,
  BlockedEdge,
  Keyframes,
  useRouteStore,
} from '@/lib/GlobalState/state';
import axios from 'axios'; // Untuk melakukan request API
import { Coordinate } from 'ol/coordinate';
import { debounce } from 'lodash';

// Constants
const KM_TO_DEGREE = 1 / 111.32;

// Utility function to create a Canvas element for visualization layers
const createCanvasElement = (
  map: OlMap,
  className: string,
): HTMLCanvasElement => {
  const mapSize = map.getSize();
  if (!mapSize) throw new Error('Map size is undefined');

  const targetElement = map.getTargetElement();
  if (!targetElement) throw new Error('Map target element is undefined');

  const existingCanvas = targetElement.querySelector(`canvas.${className}`);
  if (existingCanvas) {
    existingCanvas.remove(); // Remove existing canvas if any
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
    zIndex: '3000', // Tetapkan z-index tinggi
  });

  canvas.width = mapSize[0];
  canvas.height = mapSize[1];
  targetElement.appendChild(canvas);

  return canvas;
};

/**
 * Draws a blocked or normal edge based on the `isBlocked` flag.
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

  if (edge.isBlocked) {
    context.strokeStyle = '#FF0000'; // Red for blocked edges
    context.lineWidth = 3;
    context.setLineDash([5, 5]);
  } else {
    context.strokeStyle = '#00FF00'; // Green for unblocked edges
    context.lineWidth = 2;
    context.setLineDash([]);
  }
  context.stroke();
};

/**
 * Layer for Partial Path visualization
 */
export const createPartialPathCanvas = (map: OlMap): HTMLCanvasElement => {
  const canvas = createCanvasElement(map, 'partial-path-canvas');
  const context = canvas.getContext('2d');
  if (!context) throw new Error('Canvas context is undefined');

  const drawPartialPath = (path: PathPoint[]) => {
    context.clearRect(0, 0, canvas.width, canvas.height);
    if (path.length < 2) return;

    // Draw Partial Path
    context.beginPath();
    path.forEach((point, index) => {
      if (!point.coordinates || point.coordinates.length !== 2) {
        console.error('Invalid point coordinates:', point);
        return;
      }
      const coord = fromLonLat(point.coordinates);
      const pixel = map.getPixelFromCoordinate(coord);
      if (!pixel) return;
      index === 0
        ? context.moveTo(pixel[0], pixel[1])
        : context.lineTo(pixel[0], pixel[1]);
    });
    context.strokeStyle = '#FFA500'; // Orange for partial path
    context.lineWidth = 3;
    context.setLineDash([10, 10]);
    context.stroke();
    context.setLineDash([]);
  };

  return canvas;
};

/**
 * Layer for Final Path visualization
 */
export const createFinalPathCanvas = (map: OlMap): HTMLCanvasElement => {
  const canvas = createCanvasElement(map, 'final-path-canvas');
  const context = canvas.getContext('2d');
  if (!context) throw new Error('Canvas context is undefined');

  const drawFinalPath = (finalPath: PathPoint[]) => {
    context.clearRect(0, 0, canvas.width, canvas.height);
    if (finalPath.length < 2) return;

    context.beginPath();
    finalPath.forEach((point: PathPoint, index: number) => {
      if (!point.coordinates || point.coordinates.length !== 2) {
        console.error('Invalid point coordinates:', point);
        return;
      }
      const coord = fromLonLat(point.coordinates);
      const pixel = map.getPixelFromCoordinate(coord);
      if (!pixel) return;
      index === 0
        ? context.moveTo(pixel[0], pixel[1])
        : context.lineTo(pixel[0], pixel[1]);
    });
    context.strokeStyle = '#0000FF'; // Blue for final path
    context.lineWidth = 4;
    context.stroke();

    // Optionally, display distance or other info
    // For example, add a text label at the end point
    if (finalPath.length > 0) {
      const lastPoint = finalPath[finalPath.length - 1];
      if (!lastPoint.coordinates || lastPoint.coordinates.length !== 2) {
        console.error('Invalid last point coordinates:', lastPoint);
        return;
      }
      const coord = fromLonLat(lastPoint.coordinates);
      const pixel = map.getPixelFromCoordinate(coord);
      if (pixel) {
        context.fillStyle = '#000000'; // Black text
        context.font = '16px Arial';
        context.fillText(
          `${lastPoint.rel_heading?.toFixed(2) ?? 0}°`,
          pixel[0] + 5,
          pixel[1] - 5,
        );
      }
    }
  };

  return canvas;
};

/**
 * Layer for All Edges visualization
 */
export const createAllEdgesCanvas = (map: OlMap): HTMLCanvasElement => {
  const canvas = createCanvasElement(map, 'all-edges-canvas');
  const context = canvas.getContext('2d');
  if (!context) throw new Error('Canvas context is undefined');

  /**
   * Draws all edges. Before drawing, it clears the canvas to remove previous visualizations.
   */
  const drawAllEdges = (edges: BlockedEdge[]) => {
    context.clearRect(0, 0, canvas.width, canvas.height);
    edges.forEach((edge: BlockedEdge) => {
      drawBlockedEdge(map, context, edge);
    });
  };

  return canvas;
};

/**
 * Initialize Canvas layers for route visualization.
 * @param map - OpenLayers Map object.
 */
export const initializeCanvasLayers = (
  map: OlMap,
  shipSpeed: number,
  loadCondition: string,
) => {
  createPartialPathCanvas(map);
  createFinalPathCanvas(map);
  createAllEdgesCanvas(map);

  // Initial fetch and draw
  fetchAndDrawBlockedEdges(map, shipSpeed, loadCondition);

  // Debounced function to fetch and draw edges
  const debouncedFetchAndDraw = debounce(() => {
    fetchAndDrawBlockedEdges(map, shipSpeed, loadCondition);
  }, 500); // 500ms delay

  // Attach event listener for 'moveend' to fetch and draw edges
  map.on('moveend', debouncedFetchAndDraw);
};

/**
 * Function to animate keyframes.
 * @param map - OpenLayers Map object.
 * @param keyframes - Keyframes data.
 * @returns Cleanup function to stop animation and remove event listeners.
 */
export const animateKeyframes = (
  map: OlMap,
  keyframes: Keyframes,
): (() => void) | null => {
  console.log('Animating keyframes:', keyframes);
  // References to canvas elements
  const partialPathCanvas = document.querySelector(
    'canvas.partial-path-canvas',
  ) as HTMLCanvasElement;
  const finalPathCanvas = document.querySelector(
    'canvas.final-path-canvas',
  ) as HTMLCanvasElement;
  const allEdgesCanvas = document.querySelector(
    'canvas.all-edges-canvas',
  ) as HTMLCanvasElement;

  if (!partialPathCanvas || !finalPathCanvas || !allEdgesCanvas) {
    console.error('Canvas elements not found');
    return null;
  }

  const partialPathContext = partialPathCanvas.getContext('2d');
  const finalPathContext = finalPathCanvas.getContext('2d');
  const allEdgesContext = allEdgesCanvas.getContext('2d');

  if (!partialPathContext || !finalPathContext || !allEdgesContext) {
    console.error('Canvas contexts are undefined');
    return null;
  }

  // Clear all canvases before starting
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
  allEdgesContext.clearRect(0, 0, allEdgesCanvas.width, allEdgesCanvas.height);

  // Set isCalculating to true at the start of animation
  useRouteStore.getState().setIsCalculating(true);

  // Animation parameters
  const edgeBatchSize = 10; // Number of edges to draw per batch
  const edgeBatchDelay = 100; // milliseconds between edge batches
  const partialPathDelay = 100; // 100 milliseconds delay between partial path points
  const finalPathDelay = 1000; // 1 second delay before drawing final path

  let currentEdgeIndex = 0;
  const totalEdges = keyframes.all_edges.length;

  let animationTimeouts: NodeJS.Timeout[] = [];

  // Function to draw edges in batches using BFS traversal order
  const drawEdgesSequentially = () => {
    if (currentEdgeIndex >= totalEdges) {
      const timeout = setTimeout(drawPartialPathSequentially, partialPathDelay);
      animationTimeouts.push(timeout);
      return;
    }

    const edgesToDraw = keyframes.all_edges.slice(
      currentEdgeIndex,
      currentEdgeIndex + edgeBatchSize,
    );
    edgesToDraw.forEach((edge: BlockedEdge) => {
      drawBlockedEdge(map, allEdgesContext, edge);
    });
    currentEdgeIndex += edgeBatchSize;
    const timeout = setTimeout(drawEdgesSequentially, edgeBatchDelay);
    animationTimeouts.push(timeout);
  };

  // Function to draw partial path one by one with delay
  const partialPath = keyframes.partial_path || [];
  const totalPartialPath = partialPath.length;
  let currentPartialPathIndex = 0;

  const drawPartialPathSequentially = () => {
    if (currentPartialPathIndex >= totalPartialPath) {
      const timeout = setTimeout(drawFinalPath, finalPathDelay);
      animationTimeouts.push(timeout);
      return;
    }

    const point = partialPath[currentPartialPathIndex];
    if (!point || !point.coordinates || point.coordinates.length !== 2) {
      console.error('Invalid point in partial_path:', point);
      currentPartialPathIndex++;
      setTimeout(drawPartialPathSequentially, partialPathDelay);
      return;
    }

    const coord = fromLonLat(point.coordinates);
    const pixel = map.getPixelFromCoordinate(coord);
    if (!pixel) return;

    partialPathContext.beginPath();
    if (currentPartialPathIndex === 0) {
      partialPathContext.moveTo(pixel[0], pixel[1]);
    } else {
      const prevPoint = partialPath[currentPartialPathIndex - 1];
      if (
        !prevPoint ||
        !prevPoint.coordinates ||
        prevPoint.coordinates.length !== 2
      ) {
        console.error('Invalid previous point in partial_path:', prevPoint);
        currentPartialPathIndex++;
        setTimeout(drawPartialPathSequentially, partialPathDelay);
        return;
      }
      const prevCoord = fromLonLat(prevPoint.coordinates);
      const prevPixel = map.getPixelFromCoordinate(prevCoord);
      if (prevPixel) {
        partialPathContext.moveTo(prevPixel[0], prevPixel[1]);
        partialPathContext.lineTo(pixel[0], pixel[1]);
      }
    }
    partialPathContext.strokeStyle = '#FFA500'; // Orange for partial path
    partialPathContext.lineWidth = 3;
    partialPathContext.setLineDash([10, 10]);
    partialPathContext.stroke();
    partialPathContext.setLineDash([]);

    currentPartialPathIndex++;
    const timeout = setTimeout(drawPartialPathSequentially, partialPathDelay);
    animationTimeouts.push(timeout);
  };

  // Function to draw the final path with different color
  const drawFinalPath = () => {
    const finalPath = keyframes.final_path?.path || [];
    const distance = keyframes.final_path?.distance ?? 0;

    if (finalPath.length >= 2) {
      finalPathContext.beginPath();
      finalPath.forEach((point: PathPoint, index: number) => {
        if (!point.coordinates || point.coordinates.length !== 2) {
          console.error('Invalid point in final_path:', point);
          return;
        }
        const coord = fromLonLat(point.coordinates);
        const pixel = map.getPixelFromCoordinate(coord);
        if (!pixel) return;
        index === 0
          ? finalPathContext.moveTo(pixel[0], pixel[1])
          : finalPathContext.lineTo(pixel[0], pixel[1]);
      });
      finalPathContext.strokeStyle = '#0000FF'; // Blue for final path
      finalPathContext.lineWidth = 4;
      finalPathContext.stroke();
    }

    // Optionally, display distance or other info
    // For example, add a text label at the end point
    if (finalPath.length > 0) {
      const lastPoint = finalPath[finalPath.length - 1];
      if (!lastPoint.coordinates || lastPoint.coordinates.length !== 2) {
        console.error('Invalid last point coordinates:', lastPoint);
        return;
      }
      const coord = fromLonLat(lastPoint.coordinates);
      const pixel = map.getPixelFromCoordinate(coord);
      if (pixel) {
        finalPathContext.fillStyle = '#000000'; // Black text
        finalPathContext.font = '16px Arial';
        finalPathContext.fillText(
          `${distance.toFixed(2)} km`,
          pixel[0] + 5,
          pixel[1] - 5,
        );
      }
    }

    // Set isCalculating to false when animation completes
    useRouteStore.getState().setIsCalculating(false);
  };

  // Function to handle map interactions
  const handleInteractionStart = () => {
    // Hide all canvases during interaction
    partialPathCanvas.style.display = 'none';
    finalPathCanvas.style.display = 'none';
    allEdgesCanvas.style.display = 'none';
  };

  const handleInteractionEnd = () => {
    // Show canvases after interaction
    partialPathCanvas.style.display = 'block';
    finalPathCanvas.style.display = 'block';
    allEdgesCanvas.style.display = 'block';
    // Restart the animation
    currentEdgeIndex = 0;
    currentPartialPathIndex = 0;
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
    allEdgesContext.clearRect(
      0,
      0,
      allEdgesCanvas.width,
      allEdgesCanvas.height,
    );
    useRouteStore.getState().setIsCalculating(true); // Set isCalculating to true when animation restarts
    drawEdgesSequentially();
  };

  // Cleanup function
  const cleanup = () => {
    // Clear all pending timeouts
    animationTimeouts.forEach((timeout) => clearTimeout(timeout));
    animationTimeouts = [];

    // Remove event listeners
    map.un('movestart', handleInteractionStart);
    map.un('moveend', handleInteractionEnd);

    // Clear canvases
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
    allEdgesContext.clearRect(
      0,
      0,
      allEdgesCanvas.width,
      allEdgesCanvas.height,
    );

    // Set isCalculating to false when animation is cleaned up
    useRouteStore.getState().setIsCalculating(false);
  };

  // Attach interaction handlers
  map.on('movestart', handleInteractionStart);
  map.on('moveend', handleInteractionEnd);

  // Start the animations
  drawEdgesSequentially();

  // Return the cleanup function
  return cleanup;
};

/**
 * Add Route Layer to Map and handle animation
 */
export const addRouteLayerToMap = async (
  map: OlMap,
  keyframes: Keyframes,
  previousCleanup: (() => void) | null,
): Promise<(() => void) | null> => {
  if (previousCleanup) {
    previousCleanup();
  }

  try {
    const cleanup = animateKeyframes(map, keyframes);
    return cleanup;
  } catch (error) {
    console.error('Error animating keyframes:', error);
    return () => {};
  }
};

/**
 * New Function: Fetch Blocked Edges from API and Draw Them
 */
export const fetchAndDrawBlockedEdges = async (
  map: OlMap,
  shipSpeed: number,
  loadCondition: string,
) => {
  const allEdgesCanvas = document.querySelector(
    'canvas.all-edges-canvas',
  ) as HTMLCanvasElement;
  const allEdgesContext = allEdgesCanvas.getContext('2d');
  if (!allEdgesContext)
    throw new Error('All Edges Canvas context is undefined');

  const view = map.getView();
  const extent = view.calculateExtent(map.getSize()); // [minX, minY, maxX, maxY]

  // Mengonversi extent ke koordinat longitude dan latitude menggunakan toLonLat
  const minCoord: Coordinate = toLonLat([extent[0], extent[1]]);
  const maxCoord: Coordinate = toLonLat([extent[2], extent[3]]);

  const minLon = minCoord[0];
  const minLat = minCoord[1];
  const maxLon = maxCoord[0];
  const maxLat = maxCoord[1];

  // Debug: Log koordinat untuk verifikasi
  console.log('Converted Coordinates:', { minLon, minLat, maxLon, maxLat });

  try {
    // Panggil API get_blocked_edges_in_view
    const response = await axios.post(
      'http://localhost:5000/get_blocked_edges_in_view',
      {
        view_bounds: [minLon, minLat, maxLon, maxLat],
        ship_speed: shipSpeed,
        condition: loadCondition === 'ballast' ? 1 : 0,
      },
    );

    const blockedEdges: BlockedEdge[] = response.data.blocked_edges;

    // Debug: Log jumlah edges yang diterima
    console.log(`Received ${blockedEdges.length} blocked edges`);

    // Clear previous edges
    allEdgesContext.clearRect(
      0,
      0,
      allEdgesCanvas.width,
      allEdgesCanvas.height,
    );

    // Draw new edges
    blockedEdges.forEach((edge: BlockedEdge) => {
      drawBlockedEdge(map, allEdgesContext, edge);
    });
  } catch (error) {
    console.error('Error fetching blocked edges:', error);
  }
};
