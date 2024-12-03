'use client';
import React, { useEffect, useState } from 'react';
import { Map, View } from 'ol';
import VectorLayer from 'ol/layer/Vector';
import VectorSource from 'ol/source/Vector';
import { Style, Stroke, Fill, Circle as CircleStyle } from 'ol/style';
import { Point, Polygon } from 'ol/geom';
import { Feature } from 'ol';
import { GeoJSON } from 'ol/format';
import { fromLonLat, transformExtent } from 'ol/proj';
import 'ol/ol.css';

const CONFIG = {
  MAP_CENTER: [118.0, -2.0],
  MAP_ZOOM: 6,
  MAX_ZOOM: 14,
  MIN_ZOOM: 4,
  GEOJSON_URL: '/contents/eez.json',
  START_NODE: {
    lon: 113.53701042040963,
    lat: -4.736794825727632,
  },
  INDONESIA_EXTENT: [92.0, -15.0, 141.0, 10.0],
  GRID_SPACING: 1 / 111.32, // 1km in degrees
};

const initializeMap = () => {
  const extent = transformExtent(
    CONFIG.INDONESIA_EXTENT,
    'EPSG:4326',
    'EPSG:3857',
  );

  const map = new Map({
    target: 'map',
    view: new View({
      center: fromLonLat(CONFIG.MAP_CENTER),
      zoom: CONFIG.MAP_ZOOM,
      extent,
      constrainResolution: true,
      maxZoom: CONFIG.MAX_ZOOM,
      minZoom: CONFIG.MIN_ZOOM,
    }),
  });

  return map;
};

const addGeoJSONLayer = async (map: Map): Promise<VectorSource | null> => {
  try {
    const response = await fetch(CONFIG.GEOJSON_URL);
    if (!response.ok) throw new Error('Failed to load GeoJSON');

    const geojsonData = await response.json();

    const geojsonSource = new VectorSource({
      features: new GeoJSON().readFeatures(geojsonData, {
        featureProjection: 'EPSG:3857',
      }),
    });

    const geojsonLayer = new VectorLayer({
      source: geojsonSource,
      style: new Style({
        stroke: new Stroke({
          color: 'blue',
          width: 2,
        }),
        fill: new Fill({
          color: 'rgba(0, 0, 255, 0.1)',
        }),
      }),
    });

    map.addLayer(geojsonLayer);
    return geojsonSource;
  } catch (error) {
    console.error('Error loading GeoJSON:', error);
    return null;
  }
};

const haversineDistance = (
  lon1: number,
  lat1: number,
  lon2: number,
  lat2: number,
) => {
  const R = 6371; // Earth's radius in km
  const toRad = (deg: number) => (deg * Math.PI) / 180;

  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

  return R * c; // Distance in km
};

const growRegionWithEdges = async (
  map: Map,
  geojsonGeometry: any,
  updateStatus: (status: string) => void,
  exportGraph: (graph: any) => void,
) => {
  const gridSource = new VectorSource();
  const regionLayer = new VectorLayer({
    source: gridSource,
    style: new Style({
      stroke: new Stroke({
        color: 'rgba(0, 0, 0, 0.5)',
        width: 1,
      }),
      fill: new Fill({
        color: 'rgba(0, 255, 0, 0.3)',
      }),
    }),
  });

  map.addLayer(regionLayer);

  const startLon = CONFIG.START_NODE.lon;
  const startLat = CONFIG.START_NODE.lat;
  const gridSpacing = CONFIG.GRID_SPACING;

  const visited = new Set<string>();
  const queue = [[startLon, startLat]];

  const graph = {
    nodes: {} as Record<string, { lon: number; lat: number }>,
    edges: [] as { from: string; to: string; weight: number }[],
  };

  while (queue.length > 0) {
    const [lon, lat] = queue.shift()!;
    const gridKey = `${lon}_${lat}`;

    if (visited.has(gridKey)) continue;
    visited.add(gridKey);

    const projectedExtent = transformExtent(
      [
        lon - gridSpacing / 2,
        lat - gridSpacing / 2,
        lon + gridSpacing / 2,
        lat + gridSpacing / 2,
      ],
      'EPSG:4326',
      'EPSG:3857',
    );

    const intersects = geojsonGeometry.intersectsExtent(projectedExtent);
    if (!intersects) continue;

    // Add grid to region
    const polygon = new Polygon([
      [
        fromLonLat([lon - gridSpacing / 2, lat - gridSpacing / 2]),
        fromLonLat([lon + gridSpacing / 2, lat - gridSpacing / 2]),
        fromLonLat([lon + gridSpacing / 2, lat + gridSpacing / 2]),
        fromLonLat([lon - gridSpacing / 2, lat + gridSpacing / 2]),
        fromLonLat([lon - gridSpacing / 2, lat - gridSpacing / 2]),
      ],
    ]);
    gridSource.addFeature(new Feature(polygon));

    // Add node to graph
    graph.nodes[gridKey] = { lon, lat };

    // Add edges to neighboring nodes
    const neighbors = [
      [lon - gridSpacing, lat],
      [lon + gridSpacing, lat],
      [lon, lat - gridSpacing],
      [lon, lat + gridSpacing],
    ];

    for (const [neighborLon, neighborLat] of neighbors) {
      const neighborKey = `${neighborLon}_${neighborLat}`;
      const distance = haversineDistance(lon, lat, neighborLon, neighborLat);

      if (
        visited.has(neighborKey) ||
        !geojsonGeometry.intersectsExtent(
          transformExtent(
            [
              neighborLon - gridSpacing / 2,
              neighborLat - gridSpacing / 2,
              neighborLon + gridSpacing / 2,
              neighborLat + gridSpacing / 2,
            ],
            'EPSG:4326',
            'EPSG:3857',
          ),
        )
      )
        continue;

      graph.edges.push({ from: gridKey, to: neighborKey, weight: distance });
      queue.push([neighborLon, neighborLat]);
    }

    // Update status
    updateStatus(`Processing grid: ${lon}, ${lat}`);
    await new Promise((resolve) => setTimeout(resolve, 50)); // Small delay for rendering
  }

  updateStatus('Finished growing region');
  exportGraph(graph);
};

const EfficientMapWithRegionGrowing: React.FC = () => {
  const [status, setStatus] = useState('Initializing map...');
  const [graphData, setGraphData] = useState<any>(null);

  const handleExport = () => {
    if (graphData) {
      const blob = new Blob([JSON.stringify(graphData, null, 2)], {
        type: 'application/json',
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'region_graph.json';
      a.click();
      URL.revokeObjectURL(url);
    }
  };

  useEffect(() => {
    const map = initializeMap();

    const loadLayers = async () => {
      setStatus('Loading GeoJSON...');
      const geojsonSource = await addGeoJSONLayer(map);

      if (geojsonSource) {
        const geojsonGeometry = geojsonSource.getFeatures()[0]?.getGeometry();

        if (geojsonGeometry) {
          setStatus('Growing region...');
          await growRegionWithEdges(
            map,
            geojsonGeometry,
            setStatus,
            setGraphData,
          );
        } else {
          setStatus('Error: GeoJSON geometry not found.');
        }
      }
    };

    loadLayers();

    return () => {
      map.setTarget(undefined);
    };
  }, []);

  return (
    <div>
      <div id='map' style={{ width: '100%', height: '90vh' }} />
      <div
        style={{
          position: 'absolute',
          bottom: 0,
          left: 0,
          width: '100%',
          backgroundColor: 'white',
          padding: '10px',
        }}
      >
        <p>Status: {status}</p>
        <button onClick={handleExport} disabled={!graphData}>
          Export Graph to JSON
        </button>
      </div>
    </div>
  );
};

export default EfficientMapWithRegionGrowing;
