// MeterGridMap.tsx
'use client';

import React, { useEffect, useRef, useState, useCallback } from 'react';

// OpenLayers
import { Map, View, Overlay, Feature } from 'ol';
import VectorLayer from 'ol/layer/Vector';
import { fromLonLat } from 'ol/proj';
import { apply } from 'ol-mapbox-style';
import 'ol/ol.css';

// Components
import Popup from '../components/popup';
import MapClickHandler, { PopupData } from '../components/mapClickHandler';
import {
  createLocationMarkers,
  createMarkerLayer,
} from '../components/mapLayer';
import { updateDynamicGridLayer } from '../components/OverlayHandler';
import { addWaveLayerToMap } from '../components/AnimateHandler';

// UI Components
import { SlidersVertical, Info } from 'lucide-react';
import { Checkbox } from '@/components/ui/checkbox';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover';
import { Separator } from '@/components/ui/separator';
import Typography from '@/components/Typography';

// Utilities and Constants
import { MAPTILER_API_KEY, CONFIG } from '@/constant/map';
import { fromLonLatExtent } from '@/lib/utils';
import { useRouteStore } from '@/lib/GlobalState/state';
import { debounce } from 'lodash';

const MeterGridMap: React.FC = () => {
  const mapRef = useRef<HTMLDivElement | null>(null);
  const popupRef = useRef<HTMLDivElement | null>(null);
  const mapInstanceRef = useRef<Map | null>(null);
  const locationLayerRef = useRef<VectorLayer | null>(null);
  const clickHandlerRef = useRef<MapClickHandler | null>(null);

  const [loading, setLoading] = useState(false);
  const [popupData, setPopupData] = useState<PopupData | null>(null);
  const [overlayType, setOverlayType] = useState<
    'htsgwsfc' | 'perpwsfc' | 'none'
  >('htsgwsfc');
  const [animationEnabled, setAnimationEnabled] = useState(false);
  const [loadingOverlayType, setLoadingOverlayType] = useState(false);
  const [waveLoading, setWaveLoading] = useState(false);
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
        if (mapTilerLayer) mapTilerLayer.setZIndex(900); // Ensure this layer stays at the back
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

  // Toggle Wave Layer
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    let cleanupWaveLayer: (() => void) | null = null; // Store cleanup function

    const toggleWaveLayer = async () => {
      if (animationEnabled) {
        // Enable wave layer
        setWaveLoading(true); // Start loading
        try {
          // Add wave layer with animation
          cleanupWaveLayer = await addWaveLayerToMap(
            map,
            animationEnabled,
            cleanupWaveLayer,
          );
        } catch (error) {
          console.error('Failed to add wave layer:', error);
        } finally {
          setWaveLoading(false); // End loading
        }
      } else {
        // Remove wave layer
        if (cleanupWaveLayer) {
          cleanupWaveLayer(); // Call cleanup function
          cleanupWaveLayer = null; // Clear reference
          console.log('Wave layer removed.');
        }
      }
    };

    toggleWaveLayer();

    // Cleanup on component unmount
    return () => {
      if (cleanupWaveLayer) {
        cleanupWaveLayer();
        cleanupWaveLayer = null;
      }
    };
  }, [animationEnabled]);

  // Debounced resize handler
  const handleResize = useCallback(
    debounce(() => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.updateSize();
      }
    }, 300),
    [],
  );

  // Attach resize listener
  useEffect(() => {
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      handleResize.cancel();
    };
  }, [handleResize]);

  // Update overlay layer based on selection
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map || !overlayType) return;

    setLoadingOverlayType(true); // Start loading state

    // Cleanup reference to manage dynamic overlay removal
    let cleanupOverlayLayer: (() => void) | null = null;

    const updateOverlay = async () => {
      try {
        // Update the grid layer and manage cleanup
        cleanupOverlayLayer = await updateDynamicGridLayer(
          map,
          overlayType,
          cleanupOverlayLayer, // Pass previous cleanup function
        );
      } catch (error) {
        console.error('Error updating overlay layer:', error);
      } finally {
        setLoadingOverlayType(false); // End loading state
      }
    };

    updateOverlay();

    // Clean up overlay when component unmounts or overlayType changes
    return () => {
      if (cleanupOverlayLayer) {
        cleanupOverlayLayer(); // Remove the overlay if present
        cleanupOverlayLayer = null;
      }
    };
  }, [overlayType]);

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
      <div ref={mapRef} className='relative z-[1] h-screen w-full' />
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
      {/* Legend Popover */}
      <div className='absolute right-4 top-4 z-[2]'>
        <Popover>
          <PopoverTrigger asChild>
            <button className='rounded-full bg-white p-2 shadow-lg'>
              <Info color='#1f2937' size={20} />
            </button>
          </PopoverTrigger>
          <PopoverContent className='w-[240px] rounded-md bg-white p-2 shadow-lg'>
            {(overlayType !== 'none' || animationEnabled) && (
              <>
                {overlayType !== 'none' && (
                  <div>
                    <Typography
                      className='text-typo-normal-main'
                      weight='bold'
                      variant='h4'
                    >
                      Grid Legend
                    </Typography>
                    <ul className='mt-1 space-y-1'>
                      {overlayType === 'htsgwsfc' && (
                        <>
                          <li className='flex items-center'>
                            <span
                              className='mr-2 inline-block h-4 w-4'
                              style={{
                                backgroundColor: 'rgba(220, 80, 50, 0.8)',
                              }}
                            ></span>
                            <Typography
                              variant='t2'
                              className='text-typo-normal-main'
                            >
                              High (Dark Red - Orange Gradient)
                            </Typography>
                          </li>
                          <li className='flex items-center'>
                            <span
                              className='mr-2 inline-block h-4 w-4'
                              style={{
                                backgroundColor: 'rgba(220, 150, 80, 0.8)',
                              }}
                            ></span>
                            <Typography
                              variant='t2'
                              className='text-typo-normal-main'
                            >
                              Moderate (Orange - Yellow Gradient)
                            </Typography>
                          </li>
                          <li className='flex items-center'>
                            <span
                              className='mr-2 inline-block h-4 w-4'
                              style={{
                                backgroundColor: 'rgba(220, 200, 120, 0.8)',
                              }}
                            ></span>
                            <Typography
                              variant='t2'
                              className='text-typo-normal-main'
                            >
                              Low (Yellowish Gradient)
                            </Typography>
                          </li>
                        </>
                      )}
                      {overlayType === 'perpwsfc' && (
                        <>
                          <li className='flex items-center'>
                            <span
                              className='mr-2 inline-block h-4 w-4'
                              style={{
                                backgroundColor: 'rgba(30, 120, 180, 0.8)',
                              }}
                            ></span>
                            <Typography
                              variant='t2'
                              className='text-typo-normal-main'
                            >
                              High (Teal - Dark Blue Gradient)
                            </Typography>
                          </li>
                          <li className='flex items-center'>
                            <span
                              className='mr-2 inline-block h-4 w-4'
                              style={{
                                backgroundColor: 'rgba(60, 160, 120, 0.8)',
                              }}
                            ></span>
                            <Typography
                              variant='t2'
                              className='text-typo-normal-main'
                            >
                              Moderate (Teal - Green Gradient)
                            </Typography>
                          </li>
                          <li className='flex items-center'>
                            <span
                              className='mr-2 inline-block h-4 w-4'
                              style={{
                                backgroundColor: 'rgba(120, 200, 100, 0.8)',
                              }}
                            ></span>
                            <Typography
                              variant='t2'
                              className='text-typo-normal-main'
                            >
                              Low (Greenish Gradient)
                            </Typography>
                          </li>
                        </>
                      )}
                    </ul>
                  </div>
                )}
                {animationEnabled && (
                  <div className={overlayType !== 'none' ? 'mt-3' : ''}>
                    <Typography
                      className='text-typo-normal-main'
                      weight='bold'
                      variant='h4'
                    >
                      Wave Legend
                    </Typography>
                    <ul className='mt-1 space-y-1'>
                      <li className='flex items-center'>
                        <span className='mr-2 inline-block h-4 w-4 bg-blue-500'></span>{' '}
                        <Typography
                          variant='t2'
                          className='text-typo-normal-main'
                        >
                          Calm
                        </Typography>
                      </li>
                      <li className='flex items-center'>
                        <span className='mr-2 inline-block h-4 w-4 bg-orange-500'></span>{' '}
                        <Typography
                          variant='t2'
                          className='text-typo-normal-main'
                        >
                          Moderate
                        </Typography>
                      </li>
                      <li className='flex items-center'>
                        <span className='mr-2 inline-block h-4 w-4 bg-pink-500'></span>{' '}
                        <Typography
                          variant='t2'
                          className='text-typo-normal-main'
                        >
                          High
                        </Typography>
                      </li>
                      <li className='flex items-center'>
                        <span className='mr-2 inline-block h-4 w-4 bg-purple-500'></span>{' '}
                        <Typography
                          variant='t2'
                          className='text-typo-normal-main'
                        >
                          Very High
                        </Typography>
                      </li>
                    </ul>
                  </div>
                )}
              </>
            )}
          </PopoverContent>
        </Popover>
      </div>

      {/* Control Panel */}
      <div className='absolute left-1/2 top-0 z-[2] flex -translate-x-1/2 items-center justify-center rounded-b-md'>
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
                        setOverlayType(checked ? 'htsgwsfc' : 'none')
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
                        setOverlayType(checked ? 'perpwsfc' : 'none')
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
                    disabled={waveLoading}
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
