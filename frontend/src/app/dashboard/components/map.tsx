'use client';

import React, { useEffect, useRef, useState } from 'react';
import { Map, View } from 'ol';
import VectorTileLayer from 'ol/layer/VectorTile';
import VectorTileSource from 'ol/source/VectorTile';
import VectorLayer from 'ol/layer/Vector';
import VectorSource from 'ol/source/Vector';
import { MVT } from 'ol/format';
import { fromLonLat, toLonLat, transformExtent } from 'ol/proj';
import { apply } from 'ol-mapbox-style';
import { Overlay } from 'ol';
import { GeoJSON } from 'ol/format';
import { Stroke, Style, Fill } from 'ol/style';
import Popup from './popup';
import GridLayer from './gridLayer';
import MapClickHandler, { PopupData } from './mapClickHandler';

// Konfigurasi MapTiler dan Peta
const MAPTILER_API_KEY = 'hKcj2yBSfnwPvNZKwL8F';
const CONFIG = {
  MAP_CENTER: [117.92, -2.56] as [number, number], // Tambahkan tipe tuple eksplisit
  MAP_ZOOM: 5,
  MIN_ZOOM: 4,
  MAX_ZOOM: 14,
  INDONESIA_EXTENT: [92.0, -15.0, 141.0, 10.0] as [
    number,
    number,
    number,
    number,
  ], // Gunakan tipe tuple eksplisit
  TILE_URL: 'http://localhost:8080/data/v3/{z}/{x}/{y}.pbf',
  GEOJSON_URL: '/contents/eez.json',
  STYLE_URL: `https://api.maptiler.com/maps/9c6239d7-c933-48b0-bb76-bb4aa3d0dde9/style.json?key=${MAPTILER_API_KEY}`,
};

// Tipe untuk pusat grid
interface GridCenter {
  lon: number;
  lat: number;
}

// Fungsi untuk mencari pusat grid terdekat
const findNearestGridCenter = (clickedLon: number, clickedLat: number): GridCenter => {
  const [minLon, minLat, maxLon, maxLat] = CONFIG.INDONESIA_EXTENT;
  const gridSpacing = 1 / 111.32; // 1 km ≈ 1/111.32 derajat

  const lonOffset = clickedLon - minLon;
  const latOffset = clickedLat - minLat;

  const gridLon = minLon + Math.floor(lonOffset / gridSpacing) * gridSpacing + gridSpacing / 2;
  const gridLat = minLat + Math.floor(latOffset / gridSpacing) * gridSpacing + gridSpacing / 2;

  return {
    lon: Math.max(minLon, Math.min(gridLon, maxLon)),
    lat: Math.max(minLat, Math.min(gridLat, maxLat)),
  };
};

// Fungsi untuk membuat extent dari lon/lat ke EPSG:3857
const fromLonLatExtent = (
  extent: [number, number, number, number],
): [number, number, number, number] => {
  const [minLon, minLat, maxLon, maxLat] = extent;
  const bottomLeft = fromLonLat([minLon, minLat]) as [number, number];
  const topRight = fromLonLat([maxLon, maxLat]) as [number, number];
  return [bottomLeft[0], bottomLeft[1], topRight[0], topRight[1]];
};

// Komponen utama
const MeterGridMap: React.FC = () => {
  const mapRef = useRef<HTMLDivElement | null>(null);
  const popupRef = useRef<HTMLDivElement | null>(null);
  const [loading, setLoading] = useState(false);
  const [popupData, setPopupData] = useState<PopupData | null>(null);
  const clickHandlerRef = useRef<MapClickHandler | null>(null);

  useEffect(() => {
    if (!mapRef.current || !popupRef.current) return;

    // Inisialisasi peta
    const map = new Map({
      target: mapRef.current,
      view: new View({
        center: fromLonLat(CONFIG.MAP_CENTER),
        zoom: CONFIG.MAP_ZOOM,
        minZoom: CONFIG.MIN_ZOOM,
        maxZoom: CONFIG.MAX_ZOOM,
        constrainResolution: true,
        extent: fromLonLatExtent(CONFIG.INDONESIA_EXTENT),
      }),
    });

    // Tambahkan lapisan grid
    const gridLayer = GridLayer.create();
    map.addLayer(gridLayer);

    // Tambahkan overlay untuk popup
    const overlay = new Overlay({
      element: popupRef.current,
      positioning: 'bottom-center',
      stopEvent: true,
    });
    map.addOverlay(overlay);

    // Inisialisasi handler klik
    clickHandlerRef.current = new MapClickHandler({
      apiKey: MAPTILER_API_KEY,
      onLoadingChange: setLoading,
      onPopupDataChange: setPopupData,
      overlay,
    });

    // Tambahkan lapisan tile lokal
    const tileLayer = new VectorTileLayer({
      source: new VectorTileSource({
        format: new MVT(),
        url: CONFIG.TILE_URL,
      }),
    });
    map.addLayer(tileLayer);

    // Tambahkan lapisan GeoJSON
    const geojsonLayer = new VectorLayer({
      source: new VectorSource({
        format: new GeoJSON(),
        url: CONFIG.GEOJSON_URL,
      }),
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

    // Terapkan gaya MapTiler
    apply(map, CONFIG.STYLE_URL)
      .then(() => console.log('MapTiler style applied successfully'))
      .catch((error) => {
        console.error('Error loading MapTiler style:', error);
        alert('Gagal memuat peta. Silakan periksa koneksi atau API key.');
      });

    // Event klik untuk menampilkan informasi grid
    map.on('click', (event) => {
      const clickedLonLat = toLonLat(event.coordinate);
      const [clickedLon, clickedLat] = clickedLonLat;

      const nearestCenter = findNearestGridCenter(clickedLon, clickedLat);

      console.log('Clicked Coordinates:', { lon: clickedLon, lat: clickedLat });
      console.log('Nearest Grid Center:', nearestCenter);

      map.forEachFeatureAtPixel(event.pixel, (feature, layer) => {
        console.log('Feature Properties:', feature.getProperties());
        console.log('Feature Layer:', layer);
      });

      clickHandlerRef.current?.handleClick(event);
    });

    return () => {
      map.setTarget(undefined);
    };
  }, []);

  return (
    <>
      <div ref={mapRef} className="h-screen w-full" />
      <div ref={popupRef}>
        {popupData && (
          <Popup
            placeName={popupData.placeName}
            latitude={popupData.latitude}
            longitude={popupData.longitude}
            loading={loading}
            onClose={() => clickHandlerRef.current?.clearPopup()}
          />
        )}
      </div>
    </>
  );
};

export default MeterGridMap;
