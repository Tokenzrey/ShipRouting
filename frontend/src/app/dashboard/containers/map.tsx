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
import { SlidersVertical } from 'lucide-react';
import { Checkbox } from '@/components/ui/checkbox';

import {
  scheduleWaveOverlayUpdate,
  updateDynamicGridLayer,
} from '../components/OverlayHandler';

import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover';
import { Separator } from '@/components/ui/separator';
import Typography from '@/components/Typography';

const MeterGridMap: React.FC = () => {
  const mapRef = useRef<HTMLDivElement | null>(null);
  const popupRef = useRef<HTMLDivElement | null>(null);
  const mapInstanceRef = useRef<Map | null>(null);
  const locationLayerRef = useRef<VectorLayer | null>(null);
  const waveLayerRef = useRef<VectorLayer | null>(null);
  const clickHandlerRef = useRef<MapClickHandler | null>(null);

  const [loading, setLoading] = useState(false);
  const [popupData, setPopupData] = useState<PopupData | null>(null);
  const [overlayType, setOverlayType] = useState<
    'htsgwsfc' | 'perpwsfc' | null
  >('htsgwsfc');
  const [animationEnabled, setAnimationEnabled] = useState(false);

  const [loadingOverlayType, setLoadingOverlayType] = useState(false);
  const locations = useRouteStore((state) => state.locations);

  // Initialize the map
  useEffect(() => {
    if (!mapRef.current || !popupRef.current) return;

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

    mapInstanceRef.current = map;

    const overlay = new Overlay({
      element: popupRef.current,
      positioning: 'bottom-center',
      stopEvent: true,
    });
    map.addOverlay(overlay);

    clickHandlerRef.current = new MapClickHandler({
      apiKey: MAPTILER_API_KEY,
      onLoadingChange: setLoading,
      onPopupDataChange: setPopupData,
      overlay,
    });

    apply(map, CONFIG.STYLE_URL)
      .then(() => {
        console.log('MapTiler style applied successfully');
        const mapTilerLayer = map.getLayers().item(0);
        if (mapTilerLayer) mapTilerLayer.setZIndex(900);
      })
      .catch((error) => {
        console.error('Failed to load MapTiler style:', error);
        alert('Failed to load the map. Check your connection or API key.');
      });

    const markerLayer = createMarkerLayer();
    map.addLayer(markerLayer);

    const initialLocationLayer = createLocationMarkers();
    locationLayerRef.current = initialLocationLayer;
    map.addLayer(initialLocationLayer);

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

    return () => {
      map.setTarget(undefined);
    };
  }, []);

  // Update overlay layer based on selection
  useEffect(() => {
    if (!mapInstanceRef.current || !overlayType) return;
    setLoadingOverlayType(true); // Start loading state
    updateDynamicGridLayer(mapInstanceRef, waveLayerRef, overlayType).finally(
      () => {
        setLoadingOverlayType(false); // End loading state
      },
    );
  }, [overlayType]);

  // Schedule updates if animation is enabled
  useEffect(() => {
    if (!animationEnabled || !mapInstanceRef.current) return;
    const cleanup = scheduleWaveOverlayUpdate(
      mapInstanceRef,
      waveLayerRef,
      overlayType!,
    );
    return () => cleanup();
  }, [animationEnabled, overlayType]);

  // Sync markers with global locations
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    if (locationLayerRef.current) {
      const source = locationLayerRef.current.getSource();
      source?.clear();
      const newFeatures = createLocationMarkers().getSource()?.getFeatures();
      if (newFeatures) source?.addFeatures(newFeatures);
    }
  }, [locations]);

  return (
    <>
      {/* Map Container */}
      <div ref={mapRef} className='h-screen w-full' />
      {/* Popup Container */}
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
      {/* Control Panel */}
      <div className='absolute left-1/2 top-0 flex -translate-x-1/2 items-center justify-center rounded-b-md'>
        <Popover>
          <PopoverTrigger asChild>
            <button className='rounded-b-md bg-typo-normal-white px-3 py-2'>
              <SlidersVertical color='#1f2937' size={20} />
            </button>
          </PopoverTrigger>
          <PopoverContent className='relative rounded-md'>
            <section className='relative flex justify-between gap-8'>
              <div className='block space-y-2'>
                <div>
                  <Typography
                    className='text-typo-normal-main'
                    variant='h3'
                    weight='bold'
                  >
                    Overlay
                  </Typography>
                  <Separator className='mt-1.5' />
                </div>
                <div className='flex flex-col gap-1.5'>
                  <div className='flex items-center space-x-2'>
                    <Checkbox
                      id='htsgwsfc'
                      checked={overlayType === 'htsgwsfc'}
                      onCheckedChange={(checked) =>
                        setOverlayType(checked ? 'htsgwsfc' : null)
                      }
                      disabled={loadingOverlayType}
                    />
                    <label
                      htmlFor='htsgwsfc'
                      className='text-sm font-medium leading-none text-gray-800'
                    >
                      Surface significant height
                    </label>
                  </div>
                  <div className='flex items-center space-x-2'>
                    <Checkbox
                      id='perpwsfc'
                      checked={overlayType === 'perpwsfc'}
                      onCheckedChange={(checked) =>
                        setOverlayType(checked ? 'perpwsfc' : null)
                      }
                      disabled={loadingOverlayType}
                    />
                    <label
                      htmlFor='perpwsfc'
                      className='text-sm font-medium leading-none text-gray-800'
                    >
                      Surface wave mean period
                    </label>
                  </div>
                </div>
              </div>
              <div className='block space-y-2'>
                <div>
                  <Typography
                    className='text-typo-normal-main'
                    variant='h3'
                    weight='bold'
                  >
                    Animation
                  </Typography>
                  <Separator className='mt-1.5' />
                </div>
                <div className='flex items-center space-x-2'>
                  <Checkbox
                    id='waves'
                    checked={animationEnabled}
                    onCheckedChange={(checked) =>
                      setAnimationEnabled(checked === true)
                    }
                  />
                  <label
                    htmlFor='waves'
                    className='text-sm font-medium leading-none text-gray-800'
                  >
                    Waves
                  </label>
                </div>
              </div>
            </section>
          </PopoverContent>
        </Popover>
      </div>
    </>
  );
};

export default MeterGridMap;
