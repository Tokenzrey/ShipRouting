'use client';
import React, { useEffect, useState, useCallback, useRef } from 'react';
import { Map, View } from 'ol';
import { fromLonLat, transformExtent } from 'ol/proj';
import TileLayer from 'ol/layer/Tile';
import VectorTileLayer from 'ol/layer/VectorTile';
import VectorTileSource from 'ol/source/VectorTile';
import XYZSource from 'ol/source/XYZ';
import { MVT } from 'ol/format';
import { Vector as VectorLayer } from 'ol/layer';
import { Vector as VectorSource } from 'ol/source';
import { GeoJSON } from 'ol/format';
import { Point } from 'ol/geom';
import { Stroke, Style, Fill, Circle as CircleStyle } from 'ol/style';
import { Feature } from 'ol';
import { Graph } from '@dagrejs/graphlib';
import { saveAs } from 'file-saver';
import 'ol/ol.css';

const CONFIG = {
  INDONESIA_EXTENT: [93.0, -13.0, 143.0, 10.0],
  GRID_SPACING: 1 / 111.32,
  MAP_CENTER: [118.0, -2.0],
  MAP_ZOOM: 6,
  MAX_ZOOM: 14,
  MIN_ZOOM: 4,
  PBF_URL: 'http://localhost:8080/data/v3/{z}/{x}/{y}.pbf',
  GEOJSON_URL: '/contents/eez.json',
  START_GRID: { lon: 117.802813, lat: -5.066445 },
};

const MapWithStreamUpdate: React.FC = () => {
  const [gridSource] = useState(new VectorSource());
  const [map, setMap] = useState<Map | null>(null);
  const [vectorTileLayer, setVectorTileLayer] = useState<VectorTileLayer | null>(null);
  const [graph, setGraph] = useState<Graph | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);

  const graphRef = useRef(new Graph({ directed: false }));
  const gridCoordsRef = useRef<{ lon: number; lat: number }[]>([]);
  const processingRef = useRef(false);

  

  useEffect(() => {
    const extent = transformExtent(
      CONFIG.INDONESIA_EXTENT,
      'EPSG:4326',
      'EPSG:3857',
    );

    console.log('[DEBUG] Setting up map with extent:', extent);

    const newMap = new Map({
      target: 'map',
      view: new View({
        center: fromLonLat(CONFIG.MAP_CENTER),
        zoom: CONFIG.MAP_ZOOM,
        extent: extent,
        constrainResolution: true,
        maxZoom: CONFIG.MAX_ZOOM,
        minZoom: CONFIG.MIN_ZOOM,
      }),
      layers: [
        new TileLayer({
          source: new XYZSource({
            url: CONFIG.PBF_URL,
          }),
        }),
      ],
    });

    console.log('[DEBUG] Map initialized successfully.');

    const localTileLayer = new VectorTileLayer({
      source: new VectorTileSource({
        format: new MVT(),
        url: CONFIG.PBF_URL,
      }),
      style: (feature) => {
        const properties = feature.getProperties();
        return new Style({
          stroke: new Stroke({
            color: properties.class === 'ocean' ? 'blue' : 'black',
            width: 1,
          }),
          fill: new Fill({
            color:
              properties.class === 'ocean'
                ? 'rgba(0, 0, 255, 0.3)'
                : 'rgba(0, 0, 0, 0.1)',
          }),
        });
      },
    });

    setVectorTileLayer(localTileLayer);
    // newMap.addLayer(localTileLayer);

    setMap(newMap);
    console.log('[DEBUG] Tile layer added to map.');

    return () => {
      newMap.setTarget(undefined);
      console.log('[DEBUG] Map destroyed.');
    };
  }, []);

  useEffect(() => {
      const { INDONESIA_EXTENT, GRID_SPACING, GEOJSON_URL } = CONFIG;

      const loadGeoJSONAndGenerateGrid = async () => {
      try {
        console.log('[DEBUG] Starting GeoJSON load...');
        const response = await fetch(CONFIG.GEOJSON_URL);
        if (!response.ok) throw new Error(`[DEBUG] Failed to load GeoJSON: ${response.statusText}`);

        const geojsonData = await response.json();
        console.log('[DEBUG] GeoJSON loaded successfully:', geojsonData);

        const format = new GeoJSON();
        const features = format.readFeatures(geojsonData, {
          featureProjection: 'EPSG:3857',
        });

        console.log('[DEBUG] Features extracted from GeoJSON:', features);

        const geojsonSource = new VectorSource({ features });
        const combinedGeometry = geojsonSource.getFeatures()[0]?.getGeometry();

        if (!combinedGeometry) {
          console.error('[DEBUG] Failed to extract GeoJSON geometry.');
          return;
        }

        console.log('[DEBUG] Combined geometry for filtering:', combinedGeometry);

        // Start grid growing algorithm
        const startGrid = CONFIG.START_GRID;
        const queue: { lon: number; lat: number }[] = [startGrid];
        const processedGrids = new Set<string>();

        const gridCoords: { lon: number; lat: number }[] = []; // Final grid list

        while (queue.length > 0) {
          const { lon, lat } = queue.shift()!;
          const gridKey = `${lon.toFixed(6)}_${lat.toFixed(6)}`;

          if (processedGrids.has(gridKey)) {
            console.log(`[DEBUG] Skipping already processed grid: ${gridKey}`);
            continue;
          }

          processedGrids.add(gridKey);

          // Check if the grid is inside the GeoJSON boundary
          const coordinate = fromLonLat([lon, lat]);
          if (!combinedGeometry.intersectsCoordinate(coordinate)) {
            console.log(`[DEBUG] Grid ${gridKey} is outside GeoJSON boundaries. Skipping.`);
            continue;
          }

          console.log(`[DEBUG] Grid ${gridKey} is inside GeoJSON boundaries.`);
          gridCoords.push({ lon, lat });

          // Add grid to map with black marker
          const gridFeature = new Feature({
            geometry: new Point(coordinate),
          });

          gridFeature.setStyle(
            new Style({
              image: new CircleStyle({
                radius: 5,
                fill: new Fill({ color: 'black' }), // Mark grid with black color
                stroke: new Stroke({ color: 'white', width: 10 }),
              }),
            }),
          );

          gridSource.addFeature(gridFeature);

          // Add neighbors to the queue
          const neighbors = [
            { lon: lon + CONFIG.GRID_SPACING, lat },
            { lon: lon - CONFIG.GRID_SPACING, lat },
            { lon, lat: lat + CONFIG.GRID_SPACING },
            { lon, lat: lat - CONFIG.GRID_SPACING },
          ];

          neighbors.forEach((neighbor) => {
            const neighborKey = `${neighbor.lon.toFixed(6)}_${neighbor.lat.toFixed(6)}`;
            if (!processedGrids.has(neighborKey)) {
              queue.push(neighbor);
            }
          });
        }

        gridCoordsRef.current = gridCoords;
        console.log(`[DEBUG] Generated ${gridCoords.length} grid points using growing grid algorithm.`);
      } catch (error) {
        console.error(`[DEBUG] Error loading GeoJSON: ${error}`);
      }
    };
      loadGeoJSONAndGenerateGrid();
  }, []);
  
  const checkOceanGrid = useCallback(
    async (currentLon: number, currentLat: number) => {
      if (!map || !vectorTileLayer) return false;

      const pixel = map.getPixelFromCoordinate(fromLonLat([currentLon, currentLat]));
      if (!pixel) {
        console.warn('[DEBUG] Pixel not found for coordinate:', currentLon, currentLat);
        return false;
      }

      let isOcean = false;
      map.forEachFeatureAtPixel(
        pixel,
        (feature) => {
          const properties = feature.getProperties();
          if (properties?.class === 'ocean') {
            isOcean = true;
          }
        },
        {
          layerFilter: (layer) => layer === vectorTileLayer,
          hitTolerance: 0,
        },
      );

      console.log('[DEBUG] Ocean grid check result:', { lon: currentLon, lat: currentLat, isOcean });
      return isOcean;
    },
    [map, vectorTileLayer],
  );

  const processNextGrid = useCallback(async () => {
    if (processingRef.current || !gridCoordsRef.current.length) {
      if (!gridCoordsRef.current.length) {
        console.log('[DEBUG] All grids processed.');
        setIsProcessing(false);
        setGraph(new Graph(graphRef.current.graph()));
      }
      return;
    }

    processingRef.current = true;

    const { lon, lat } = gridCoordsRef.current.shift()!;

    console.log('[DEBUG] Processing grid:', { lon, lat });

    try {
      const isOcean = await checkOceanGrid(lon, lat);

      const gridFeature = new Feature({
        geometry: new Point(fromLonLat([lon, lat])),
        properties: { isOcean },
      });

      gridFeature.setStyle(
        new Style({
          image: new CircleStyle({
            radius: 5,
            fill: new Fill({
              color: isOcean ? 'blue' : 'red',
            }),
            stroke: new Stroke({
              color: 'black',
              width: 1,
            }),
          }),
        }),
      );

      gridSource.addFeature(gridFeature);

      if (isOcean) {
        const gridKey = `${lon.toFixed(6)}_${lat.toFixed(6)}`;

        graphRef.current.setNode(gridKey, {
          category: 1,
          details: 'Ocean',
          lon,
          lat,
        });

        console.log('[DEBUG] Added grid to graph:', gridKey);
      }

      setProgress((prev) => prev + 1);

      setTimeout(() => {
        processingRef.current = false;
        processNextGrid();
      }, 0);
    } catch (error) {
      console.error(`[DEBUG] Error processing grid: ${error}`);
      setIsProcessing(false);
    }
  }, [checkOceanGrid, gridSource]);

  const startProcessing = useCallback(() => {
    console.log('[DEBUG] Starting grid processing...');
    graphRef.current = new Graph({ directed: false });
    setProgress(0);
    setIsProcessing(true);
    processNextGrid();
  }, [processNextGrid]);

  const exportGraph = useCallback(() => {
    if (graph) {
      console.log('[DEBUG] Exporting graph...');
      const graphJSON = JSON.stringify(graph, null, 2);
      const blob = new Blob([graphJSON], { type: 'application/json' });
      saveAs(blob, 'indonesia_ocean_graph.json');
      console.log('[DEBUG] Graph exported successfully.');
    } else {
      console.error('[DEBUG] No graph data to export.');
      alert('No graph data to export!');
    }
  }, [graph]);

  return (
    <div>
      <div id='map' style={{ width: '100%', height: '80vh' }} />
      <div style={{ display: 'flex', gap: '10px', marginTop: '1rem' }}>
        <button
          onClick={startProcessing}
          disabled={isProcessing}
          style={{
            padding: '0.5rem 1rem',
            backgroundColor: isProcessing ? '#cccccc' : '#4CAF50',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: isProcessing ? 'not-allowed' : 'pointer',
          }}
        >
          {isProcessing ? 'Processing...' : 'Start Processing'}
        </button>
        <button
          onClick={exportGraph}
          disabled={!graph}
          style={{
            padding: '0.5rem 1rem',
            backgroundColor: graph ? '#4CAF50' : '#cccccc',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: graph ? 'pointer' : 'not-allowed',
          }}
        >
          Export Graph as JSON
        </button>
      </div>
      <div>Progress: {progress} grids processed</div>
    </div>
  );
};

export default MapWithStreamUpdate;
