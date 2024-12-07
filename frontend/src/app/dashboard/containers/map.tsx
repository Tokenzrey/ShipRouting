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
import { GridLayer, createMarkerLayer } from '../components/mapLayer';

/**
 * Komponen MeterGridMap
 * Merupakan komponen React untuk menampilkan peta interaktif dengan fitur grid dan marker.
 */
const MeterGridMap: React.FC = () => {
  // Referensi DOM untuk elemen map dan popup
  const mapRef = useRef<HTMLDivElement | null>(null);
  const popupRef = useRef<HTMLDivElement | null>(null);

  // State untuk mengelola status loading dan data popup
  const [loading, setLoading] = useState(false);
  const [popupData, setPopupData] = useState<PopupData | null>(null);

  // Referensi untuk instance MapClickHandler
  const clickHandlerRef = useRef<MapClickHandler | null>(null);

  useEffect(() => {
    // Jika referensi map atau popup belum tersedia, hentikan inisialisasi
    if (!mapRef.current || !popupRef.current) return;

    /**
     * Inisialisasi peta dengan OpenLayers
     */
    const map = new Map({
      target: mapRef.current, // Element HTML sebagai container peta
      view: new View({
        center: fromLonLat(CONFIG.MAP_CENTER), // Pusat peta (koordinat lon/lat)
        zoom: CONFIG.MAP_ZOOM, // Zoom awal
        minZoom: CONFIG.MIN_ZOOM, // Zoom minimum
        maxZoom: CONFIG.MAX_ZOOM, // Zoom maksimum
        constrainResolution: true, // Membatasi resolusi zoom
        extent: fromLonLatExtent(CONFIG.INDONESIA_EXTENT), // Batas area peta
      }),
    });

    /**
     * Tambahkan layer grid ke dalam peta
     */
    const gridLayer = GridLayer.create();
    map.addLayer(gridLayer);

    /**
     * Tambahkan overlay popup untuk menampilkan informasi
     */
    const overlay = new Overlay({
      element: popupRef.current, // Elemen HTML untuk popup
      positioning: 'bottom-center', // Posisi popup
      stopEvent: true, // Mencegah propagasi event ke peta
    });
    map.addOverlay(overlay);

    /**
     * Inisialisasi MapClickHandler untuk menangani event klik pada peta
     */
    clickHandlerRef.current = new MapClickHandler({
      apiKey: MAPTILER_API_KEY,
      onLoadingChange: setLoading, // Callback untuk mengubah status loading
      onPopupDataChange: setPopupData, // Callback untuk mengubah data popup
      overlay, // Overlay untuk menampilkan popup
    });

    /**
     * Terapkan style dari MapTiler
     */
    apply(map, CONFIG.STYLE_URL)
      .then(() => console.log('MapTiler style berhasil diterapkan'))
      .catch((error) => {
        console.error('Gagal memuat style MapTiler:', error);
        alert('Gagal memuat peta. Periksa koneksi atau API key Anda.');
      });

    /**
     * Tambahkan layer marker ke dalam peta
     */
    const markerLayer = createMarkerLayer();
    map.addLayer(markerLayer);

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

    // Membersihkan instance map saat komponen dilepas
    return () => {
      map.setTarget(undefined);
    };
  }, []);

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
