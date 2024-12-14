'use client';

import React, { useEffect, useRef, useState } from 'react';
import { Map, View, Overlay, Feature } from 'ol';
import { fromLonLat } from 'ol/proj';
import { apply } from 'ol-mapbox-style';
import 'ol/ol.css';
import Popup from '../components/popup';
import MapClickHandler, { PopupData } from '../components/mapClickHandler';
import { MAPTILER_API_KEY, CONFIG } from '@/constant/map';
import { fromLonLatExtent } from '@/lib/utils';
import {
  createLocationMarkers,
  createMarkerLayer,
} from '../components/mapLayer';
import { useRouteStore } from '@/lib/GlobalState/state';
import VectorLayer from 'ol/layer/Vector';

/**
 * Komponen MeterGridMap
 * Menampilkan peta interaktif menggunakan OpenLayers dengan fitur grid, marker, dan popup.
 */
const MeterGridMap: React.FC = () => {
  // Referensi DOM untuk elemen map dan popup
  const mapRef = useRef<HTMLDivElement | null>(null);
  const popupRef = useRef<HTMLDivElement | null>(null);

  // Referensi untuk instance Map dan layer lokasi
  const mapInstanceRef = useRef<Map | null>(null);
  const locationLayerRef = useRef<VectorLayer | null>(null);

  // State untuk mengelola status loading dan data popup
  const [loading, setLoading] = useState(false);
  const [popupData, setPopupData] = useState<PopupData | null>(null);

  // Referensi untuk instance MapClickHandler
  const clickHandlerRef = useRef<MapClickHandler | null>(null);

  // Global state dari RouteStore
  const locations = useRouteStore((state) => state.locations);

  useEffect(() => {
    if (!mapRef.current || !popupRef.current) return;

    /**
     * Inisialisasi peta dengan konfigurasi awal
     */
    const map = new Map({
      target: mapRef.current,
      view: new View({
        center: fromLonLat(CONFIG.MAP_CENTER), // Koordinat pusat peta
        zoom: CONFIG.MAP_ZOOM, // Zoom awal
        minZoom: CONFIG.MIN_ZOOM, // Zoom minimum
        maxZoom: CONFIG.MAX_ZOOM, // Zoom maksimum
        constrainResolution: true, // Membatasi resolusi zoom
        extent: fromLonLatExtent(CONFIG.INDONESIA_EXTENT), // Batas area peta
      }),
    });

    // Simpan referensi instance Map
    mapInstanceRef.current = map;

    /**
     * Tambahkan overlay untuk popup
     */
    const overlay = new Overlay({
      element: popupRef.current, // Elemen popup
      positioning: 'bottom-center', // Posisi popup relatif ke klik
      stopEvent: true, // Mencegah propagasi event ke peta
    });
    map.addOverlay(overlay);

    /**
     * Inisialisasi MapClickHandler untuk menangani klik pada peta
     */
    clickHandlerRef.current = new MapClickHandler({
      apiKey: MAPTILER_API_KEY,
      onLoadingChange: setLoading,
      onPopupDataChange: setPopupData,
      overlay,
    });

    /**
     * Terapkan style MapTiler ke peta
     */
    apply(map, CONFIG.STYLE_URL)
      .then(() => console.log('MapTiler style berhasil diterapkan'))
      .catch((error) => {
        console.error('Gagal memuat style MapTiler:', error);
        alert('Gagal memuat peta. Periksa koneksi atau API key Anda.');
      });

    /**
     * Tambahkan layer marker default
     */
    const markerLayer = createMarkerLayer();
    map.addLayer(markerLayer);

    /**
     * Tambahkan layer lokasi dengan marker
     */
    const initialLocationLayer = createLocationMarkers();
    locationLayerRef.current = initialLocationLayer;
    map.addLayer(initialLocationLayer);

    /**
     * Event handler untuk klik pada peta
     */
    map.on('click', (event) => {
      const features: Feature[] = [];
      map.forEachFeatureAtPixel(event.pixel, (feature) => {
        features.push(feature as Feature);
      });

      clickHandlerRef.current?.handleClick({
        coordinate: event.coordinate,
        features,
      });
    });

    /**
     * Bersihkan instance Map saat komponen dilepas
     */
    return () => {
      map.setTarget(undefined);
    };
  }, []);

  /**
   * Sinkronisasi lokasi di global state dengan layer marker
   */
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    // Hapus layer lokasi sebelumnya
    if (locationLayerRef.current) {
      map.removeLayer(locationLayerRef.current);
    }

    // Tambahkan layer lokasi baru berdasarkan state
    const newLocationLayer = createLocationMarkers();
    locationLayerRef.current = newLocationLayer;
    map.addLayer(newLocationLayer);
  }, [locations]);

  return (
    <>
      {/* Elemen container untuk peta */}
      <div ref={mapRef} className='h-screen w-full' />

      {/* Elemen container untuk popup */}
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
