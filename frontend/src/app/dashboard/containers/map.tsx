// MeterGridMap.tsx
'use client';

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { format } from 'date-fns';
import { CalendarIcon } from 'lucide-react';
import { cn } from '@/lib/cn';

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
  createOptimalRouteLayer,
  createSafestRouteLayer,
  initializeRouteLayerSync,
  createShipLayer,
  updateShipPosition,
  syncRouteLayers,
} from '../components/mapLayer';
import { updateDynamicGridLayer } from '../components/OverlayHandler';
import { addWaveLayerToMap } from '../components/AnimateHandler';
import {
  initializeCanvasLayers,
  animateKeyframes,
} from '../components/createCanvasLayers';
import { calculateDistanceAndDuration } from '@/components/nav-main';

// UI Components
import { SlidersVertical, Info } from 'lucide-react';
import { Checkbox } from '@/components/ui/checkbox';
import { Calendar } from '@/components/ui/calendar';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Separator } from '@/components/ui/separator';
import { Button } from '@/components/ui/button';
import Typography from '@/components/Typography';

// Utilities and Constants
import { MAPTILER_API_KEY, CONFIG } from '@/constant/map';
import { fromLonLatExtent } from '@/lib/utils';
import { Keyframes, useRouteStore } from '@/lib/GlobalState/state';
import { debounce } from 'lodash';
import VectorTileLayer from 'ol/layer/VectorTile';
import { getLegendSteps } from '@/components/legend';
import { getColorForValue } from '../components/OverlayHandler';

import { WAVE_HEIGHT_THRESHOLDS } from '../components/OverlayHandler';

type WaveRange = {
  min: number;
  max: number;
} | null;

const MeterGridMap: React.FC = () => {
  const mapRef = useRef<HTMLDivElement | null>(null);
  const popupRef = useRef<HTMLDivElement | null>(null);
  const mapInstanceRef = useRef<Map | null>(null);
  const locationLayerRef = useRef<VectorLayer | null>(null);
  const clickHandlerRef = useRef<MapClickHandler | null>(null);
  const animationCleanupRef = useRef<(() => void) | null>(null);

  const [loading, setLoading] = useState(false);
  const [popupData, setPopupData] = useState<PopupData | null>(null);
  const [overlayType, setOverlayType] = useState<
    'htsgwsfc' | 'perpwsfc' | 'none'
  >('htsgwsfc');
  const [animationEnabled, setAnimationEnabled] = useState(false);
  const [loadingOverlayType, setLoadingOverlayType] = useState(false);
  const [waveLoading, setWaveLoading] = useState(false);
  const [shippingLane, setShippingLane] = useState(true);
  const [boundary, setBoundary] = useState(true);
  const [maptilerSet, setMaptilerSet] = useState(false);
  const [date, setDate] = React.useState<Date>(new Date());
  const [isOpen, setIsOpen] = useState(false);
  const [selectedTime, setSelectedTime] = useState<'00' | '06' | '12' | '18'>(
    '00',
  );
  const [isCurrentDate, setIsCurrentDate] = useState(true);
  const [waveRangeOverlay, setWaveRangeOverlay] = useState<WaveRange>(null);
  const [waveRangeAnim, setWaveRangeAnim] = useState<WaveRange>(null);
  const [optimalKeyframes, setOptimalKeyframes] = useState<Keyframes | null>(
    null,
  );
  const [safestKeyframes, setSafestKeyframes] = useState<Keyframes | null>(
    null,
  );
  const [isDrawing, setIsDrawing] = useState(false);

  const optimalRoute = useRouteStore((state) => state.optimalRoute);
  const activeRoute = useRouteStore((state) => state.activeRoute);
  const safestRoute = useRouteStore((state) => state.safestRoute);
  const RouteSelected = useRouteStore((state) => state.routeSelected);

  const { setIsCalculating } = useRouteStore();
  const currentAnimationIndex = useRouteStore(
    (state) => state.currentAnimationIndex,
  );
  const {
    locations,
    shipSpeed,
    loadCondition,
    isCalculating,
    isUpdate,
    setOptimalDistance,
    setSafestDistance,
    setOptimalDuration,
    setSafestDuration,
    setOptimalRoute,
    setSafestRoute,
    resetKeyframes,
    setRouteSelected,
    setIsUpdate,
  } = useRouteStore();

  // Constants for z-index management
  const Z_INDEX = {
    BASE_VECTOR: 900, // Base vector layers
    CUSTOM_ROUTES: 2006, // Custom route layers
    VECTOR_TILES: 2000, // Vector tile layers (labels etc)
    MARKERS: 2005, // Marker layers
    LOCATION: 2008, // Location markers
    SHIP_ROUTES: 2007,
  };

  const handleDateChange = (day: Date | undefined) => {
    if (day) {
      setDate(day); // Pastikan hanya `Date` yang di-set
    } else {
      console.error('Tanggal tidak valid atau undefined.');
    }
  };

  useEffect(() => {
    if (!mapInstanceRef.current) return;

    const map = mapInstanceRef.current;
    setMaptilerSet(true);

    const styleUrl =
      shippingLane && boundary
        ? CONFIG.STYLE_URL_ALL
        : shippingLane
          ? CONFIG.STYLE_URL_SHIPPINGLANE
          : boundary
            ? CONFIG.STYLE_URL_BOUNDARY
            : CONFIG.STYLE_URL_VANILLA;

    // Store existing custom layers
    const customLayers = map
      .getLayers()
      .getArray()
      .filter((layer) => {
        return (
          layer === locationLayerRef.current ||
          layer.get('isMarkerLayer') ||
          layer.get('isOptimalRoute') ||
          layer.get('isSafestRoute') ||
          layer.get('isShipRoute')
        );
      });

    // Remove all layers
    map.getLayers().clear();

    // Apply new style with proper layer ordering
    apply(map, styleUrl)
      .then(() => {
        console.log('Style applied successfully');

        // Get all layers after style application
        const allLayers = map.getLayers().getArray();

        // Group layers by type
        const vectorTileLayers = allLayers.filter(
          (layer) => layer instanceof VectorTileLayer,
        );
        const baseVectorLayers = allLayers.filter(
          (layer) =>
            layer instanceof VectorLayer && !customLayers.includes(layer),
        );

        // Clear layers again to reorder
        map.getLayers().clear();

        // 1. Add base vector layers first
        baseVectorLayers.forEach((layer, index) => {
          layer.setZIndex(Z_INDEX.BASE_VECTOR + index);
          map.addLayer(layer);
        });

        // 2. Add custom route layers
        customLayers.forEach((layer) => {
          // Set specific z-index based on layer type
          if (layer.get('isOptimalRoute') || layer.get('isSafestRoute')) {
            layer.setZIndex(Z_INDEX.CUSTOM_ROUTES);
          } else if (layer.get('isMarkerLayer')) {
            layer.setZIndex(Z_INDEX.MARKERS);
          } else if (layer.get('isShipLayer')) {
            layer.setZIndex(Z_INDEX.SHIP_ROUTES);
          } else if (layer === locationLayerRef.current) {
            layer.setZIndex(Z_INDEX.LOCATION);
          }
          map.addLayer(layer);
        });

        // 3. Add vector tile layers
        vectorTileLayers.forEach((layer, index) => {
          layer.setZIndex(Z_INDEX.VECTOR_TILES + index);
          map.addLayer(layer);
        });

        // Debug layer order
        console.log('Layer order after reorganization:');
        map
          .getLayers()
          .getArray()
          .forEach((layer, index) => {
            console.log(`Layer ${index}:`, {
              type:
                layer instanceof VectorTileLayer
                  ? 'VectorTileLayer'
                  : 'VectorLayer',
              zIndex: layer.getZIndex(),
              isCustom: customLayers.includes(layer),
              isMarker: layer.get('isMarkerLayer'),
              isLocation: layer === locationLayerRef.current,
            });
          });
      })
      .catch((error) => {
        console.error('Style application failed:', error);
        alert('Failed to load the map. Check your connection or API key.');
      })
      .finally(() => {
        setMaptilerSet(false);
      });
  }, [shippingLane, boundary]);

  // Second useEffect - Initial map setup
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

    setMaptilerSet(true);

    // Apply initial style
    const initialStyleUrl = CONFIG.STYLE_URL_ALL;
    apply(map, initialStyleUrl)
      .then(() => {
        console.log('Initial style applied successfully');

        // Set base layer
        const allLayers = map.getLayers().getArray();
        allLayers.forEach((layer, index) => {
          if (layer instanceof VectorTileLayer) {
            layer.setZIndex(Z_INDEX.VECTOR_TILES + index);
          } else {
            layer.setZIndex(Z_INDEX.BASE_VECTOR + index);
          }
        });

        // Create and add marker layers with correct z-index
        const markerLayer = createMarkerLayer();
        markerLayer.set('isMarkerLayer', true);
        markerLayer.setZIndex(Z_INDEX.MARKERS);
        map.addLayer(markerLayer);

        const initialLocationLayer = createLocationMarkers();
        locationLayerRef.current = initialLocationLayer;
        initialLocationLayer.set('isLocationLayer', true);
        initialLocationLayer.setZIndex(Z_INDEX.LOCATION);
        map.addLayer(initialLocationLayer);

        // Add route layers
        const optimalRoute = createOptimalRouteLayer();
        optimalRoute.set('isOptimalRoute', true);
        optimalRoute.setZIndex(Z_INDEX.CUSTOM_ROUTES);
        map.addLayer(optimalRoute);

        const safestRoute = createSafestRouteLayer();
        safestRoute.set('isSafestRoute', true);
        safestRoute.setZIndex(Z_INDEX.CUSTOM_ROUTES);
        map.addLayer(safestRoute);

        const shipLayer = createShipLayer();
        shipLayer.set('isShipLayer', true);
        shipLayer.setZIndex(Z_INDEX.SHIP_ROUTES);
        map.addLayer(shipLayer);

        console.log(
          'Initial layer setup complete:',
          map.getLayers().getArray(),
        );
      })
      .catch((error) => {
        console.error('Style application failed:', error);
        alert('Failed to load the map. Check your connection or API key.');
      })
      .finally(() => {
        setMaptilerSet(false);
      });

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

    setIsCalculating(false);
    initializeRouteLayerSync();

    return () => {
      map.setTarget(undefined);
    };
  }, []);

  useEffect(() => {
    if (!mapInstanceRef.current) return;
    const map = mapInstanceRef.current;

    initializeCanvasLayers(map, shipSpeed, loadCondition);

    return () => {
      // Remove canvas layers on cleanup
      const targetElement = map.getTargetElement();
      if (targetElement) {
        const canvases = targetElement.querySelectorAll(
          'canvas.partial-path-canvas, canvas.final-path-canvas, canvas.all-edges-canvas',
        );
        canvases.forEach((canvas) => canvas.remove());
      }
    };
  }, [shipSpeed, loadCondition]);

  useEffect(() => {
    syncRouteLayers();
  }, [optimalRoute, safestRoute, locations]);

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
            (range) => setWaveRangeAnim(range),
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

    setLoadingOverlayType(true);

    let cleanupOverlayLayer: (() => void) | null = null;

    const updateOverlay = async () => {
      try {
        const result = await updateDynamicGridLayer(
          map,
          overlayType,
          date,
          isCurrentDate,
          selectedTime,
          cleanupOverlayLayer,
        );
        if (result) {
          cleanupOverlayLayer = result.cleanup;
          setWaveRangeOverlay({ min: result.minValue, max: result.maxValue }); // simpan
        } else {
          cleanupOverlayLayer = null;
          setWaveRangeOverlay(null);
        }
      } catch (error) {
        console.error('Error updating overlay layer:', error);
      } finally {
        setLoadingOverlayType(false);
      }
    };

    updateOverlay();

    return () => {
      if (cleanupOverlayLayer) {
        cleanupOverlayLayer();
        cleanupOverlayLayer = null;
      }
    };
  }, [overlayType, date, selectedTime, isCurrentDate]);

  // Control ship movement
  useEffect(() => {
    const route = activeRoute === 'optimal' ? safestRoute : optimalRoute;

    // Safely handle currentAnimationIndex
    const currentIndex = currentAnimationIndex ?? route.length - 1;
    const currentCoord = route[currentIndex]?.coordinates || [0, 0];
    const nextCoord = route[currentIndex + 1]?.coordinates || undefined;

    updateShipPosition(currentCoord, nextCoord);
  }, [activeRoute, currentAnimationIndex, optimalRoute, safestRoute]);

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

  // ⏳ **useEffect 1: Menjalankan perhitungan rute saat tombol ditekan**
  const handleRouteCalculation = async () => {
    if (!isUpdate || isCalculating) return;
    setIsUpdate(false);

    console.log('📌 Calculating route...', RouteSelected);
    const useModel = RouteSelected === 'safest';

    try {
      if (RouteSelected === 'normal' || RouteSelected === 'safest') {
        console.log('🚀 Starting calculation for', RouteSelected);

        await calculateDistanceAndDuration(
          useModel,
          locations,
          shipSpeed,
          loadCondition,
          RouteSelected,
          resetKeyframes,
          setOptimalKeyframes,
          setSafestKeyframes,
          setOptimalDistance,
          setOptimalDuration,
          setOptimalRoute,
          setSafestDistance,
          setSafestDuration,
          setSafestRoute,
          isCalculating,
          setRouteSelected,
        );
      }

      console.log('✅ Calculation done! Setting isDrawing to true.');
      setIsDrawing(true);
    } catch (error) {
      console.error('❌ Error during calculation:', error);
    }
  };

  useEffect(() => {
    handleRouteCalculation();
  }, [isUpdate]);

  // 🎨 **useEffect 2: Menggambar path setelah perhitungan selesai**
  useEffect(() => {
    console.log('dariwing route init....', isDrawing);
    if (!isDrawing) return; // ⛔ Jangan gambar jika belum siap
    setIsDrawing(false); // Reset state agar tidak berulang

    console.log('🎬 Drawing route on map...');
    const map = mapInstanceRef.current;
    if (!map) return;

    let keyframes: Keyframes | null = null;

    if (RouteSelected === 'normal' && optimalKeyframes) {
      keyframes = {
        final_path: { ...optimalKeyframes.final_path },
        partial_path: [...optimalKeyframes.partial_path],
        all_edges: [...optimalKeyframes.all_edges],
      };
    } else if (RouteSelected === 'safest' && safestKeyframes) {
      keyframes = {
        final_path: { ...safestKeyframes.final_path },
        partial_path: [...safestKeyframes.partial_path],
        all_edges: [...safestKeyframes.all_edges],
      };
    }

    if (keyframes) {
      console.log('🎥 Animating keyframes:', keyframes);
      if (animationCleanupRef.current) {
        animationCleanupRef.current(); // Hentikan animasi sebelumnya sebelum memulai yang baru
      }
      animationCleanupRef.current = animateKeyframes(map, keyframes);
    }

    return () => {
      if (animationCleanupRef.current) {
        animationCleanupRef.current();
        animationCleanupRef.current = null;
      }
    };
  }, [isDrawing]); // **Trigger hanya saat isDrawing berubah**

  function renderOverlayLegend() {
    if (overlayType === 'none') {
      return (
        <div className='mt-3'>
          <h4 className='font-bold'>Grid Legend</h4>
          <p>No overlay selected.</p>
        </div>
      );
    }

    if (waveRangeOverlay) {
      const steps = getLegendSteps(
        waveRangeOverlay.min,
        waveRangeOverlay.max,
        4, // 4 interval
        overlayType,
        getColorForValue,
      );

      return (
        <div className='mt-3'>
          <h4 className='font-bold'>Dynamic Grid Legend</h4>
          <ul className='mt-1 space-y-1'>
            {steps.map((step, idx) => (
              <li key={idx} className='flex items-center'>
                <span
                  className='mr-2 inline-block h-4 w-4'
                  style={{ backgroundColor: step.color }}
                />
                <span>
                  {step.from.toFixed(2)} – {step.to.toFixed(2)}
                  {overlayType === 'htsgwsfc' ? ' m' : ' s'}
                </span>
              </li>
            ))}
          </ul>
        </div>
      );
    } else {
      // Tidak ada data, gunakan rentang default
      const defaultMin = 0;
      const defaultMax = 10;
      const steps = getLegendSteps(
        defaultMin,
        defaultMax,
        4, // 4 interval
        overlayType,
        getColorForValue,
      );

      return (
        <div className='mt-3'>
          <h4 className='font-bold'>Dynamic Grid Legend</h4>
          <ul className='mt-1 space-y-1'>
            {steps.map((step, idx) => (
              <li key={idx} className='flex items-center'>
                <span
                  className='mr-2 inline-block h-4 w-4'
                  style={{ backgroundColor: step.color }}
                />
                <span>
                  {step.from.toFixed(2)} – {step.to.toFixed(2)}
                  {overlayType === 'htsgwsfc' ? ' m' : ' s'}
                </span>
              </li>
            ))}
          </ul>
          <p className='mt-2 text-gray-600'>No data available.</p>
        </div>
      );
    }
  }

  function renderAnimLegend() {
    if (!animationEnabled) return null;

    return (
      <div className={overlayType !== 'none' ? 'mt-3' : ''}>
        <Typography
          className='text-typo-normal-main'
          weight='bold'
          variant='h4'
        >
          Wave Legend
        </Typography>
        <ul className='mt-1 space-y-1'>
          {/* Calm */}
          <li className='flex items-center'>
            <span className='mr-2 inline-block h-4 w-4 bg-[rgba(135,206,250,0.9)]'></span>
            <Typography variant='t2' className='text-typo-normal-main'>
              {`Calm (< ${WAVE_HEIGHT_THRESHOLDS.CALM} m)`}
            </Typography>
          </li>

          {/* Moderate */}
          <li className='flex items-center'>
            <span className='mr-2 inline-block h-4 w-4 bg-[rgba(255,128,0,0.9)]'></span>
            <Typography variant='t2' className='text-typo-normal-main'>
              {`Moderate (${WAVE_HEIGHT_THRESHOLDS.CALM} - ${WAVE_HEIGHT_THRESHOLDS.MODERATE} m)`}
            </Typography>
          </li>

          {/* High */}
          <li className='flex items-center'>
            <span className='mr-2 inline-block h-4 w-4 bg-[rgba(255,0,128,0.9)]'></span>
            <Typography variant='t2' className='text-typo-normal-main'>
              {`High (${WAVE_HEIGHT_THRESHOLDS.MODERATE} - ${WAVE_HEIGHT_THRESHOLDS.HIGH} m)`}
            </Typography>
          </li>

          {/* Very High */}
          <li className='flex items-center'>
            <span className='mr-2 inline-block h-4 w-4 bg-[rgba(128,0,255,0.9)]'></span>
            <Typography variant='t2' className='text-typo-normal-main'>
              {`Very High (≥ ${WAVE_HEIGHT_THRESHOLDS.HIGH} m)`}
            </Typography>
          </li>
        </ul>
      </div>
    );
  }

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
            waveData={popupData.waveData}
            loading={loading}
            onClose={() => clickHandlerRef.current?.clearPopup()}
          />
        )}
      </div>
      {/* Legend Popover */}
      <div className='absolute right-4 top-4 z-[2]'>
        <Popover defaultOpen>
          <PopoverTrigger asChild>
            <button className='rounded-full bg-white p-2 shadow-lg'>
              <Info color='#1f2937' size={20} />
            </button>
          </PopoverTrigger>
          <PopoverContent className='mr-3 w-[240px] rounded-md bg-white p-2 shadow-lg'>
            {renderOverlayLegend()}
            {renderAnimLegend()}
          </PopoverContent>
        </Popover>
      </div>

      {/* Control Panel */}
      <div className='absolute left-1/2 top-0 z-[2] flex -translate-x-1/2 items-center justify-center rounded-b-md'>
        <Popover open={isOpen} onOpenChange={setIsOpen}>
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
                    Wave Data
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
                      Wave Direction
                    </label>
                  </div>
                  <div className='mt-2 flex items-center space-x-2'>
                    <Popover>
                      <PopoverTrigger asChild>
                        <Button
                          variant={'outline'}
                          className={cn(
                            'flex h-auto w-full items-center justify-start px-2 py-2 text-left text-[0.6875rem] font-medium leading-[0.9375rem] md:text-[0.6875rem] md:leading-[0.9375rem]',
                            !date && 'text-muted-foreground',
                          )}
                        >
                          <CalendarIcon size={12} className='!h-3 !w-3' />
                          {date ? (
                            format(date, 'PPP')
                          ) : (
                            <Typography
                              variant='t2'
                              className={cn(
                                'text-typo-normal-main',
                                !date && 'text-muted-foreground',
                              )}
                              weight='medium'
                            >
                              Pick a date
                            </Typography>
                          )}
                        </Button>
                      </PopoverTrigger>
                      <PopoverContent className='w-auto p-0' align='start'>
                        <Calendar
                          mode='single'
                          selected={date}
                          onSelect={handleDateChange}
                          initialFocus
                        />
                      </PopoverContent>
                    </Popover>
                    <Select
                      value={selectedTime}
                      onValueChange={(value) =>
                        setSelectedTime(value as '00' | '06' | '12' | '18')
                      }
                      disabled={loadingOverlayType}
                    >
                      <SelectTrigger className='!h-auto w-[70px] px-2 py-2 text-[0.6875rem] font-medium leading-[0.9375rem] !outline-0 !ring-0 md:text-[0.6875rem] md:leading-[0.9375rem]'>
                        <SelectValue placeholder='Select Time'>
                          {selectedTime === '00' && '00:00'}
                          {selectedTime === '06' && '06:00'}
                          {selectedTime === '12' && '12:00'}
                          {selectedTime === '18' && '18:00'}
                        </SelectValue>
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value='00'>00:00</SelectItem>
                        <SelectItem value='06'>06:00</SelectItem>
                        <SelectItem value='12'>12:00</SelectItem>
                        <SelectItem value='18'>18:00</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className='flex items-center justify-end space-x-1'>
                    <label
                      htmlFor='htsgwsfc'
                      className={cn(
                        'text-xs font-medium leading-none',
                        isCurrentDate && 'text-info-dark',
                      )}
                    >
                      Current Date
                    </label>
                    <Checkbox
                      id='htsgwsfc'
                      checked={isCurrentDate}
                      onCheckedChange={(checked) =>
                        setIsCurrentDate(checked === true)
                      }
                      disabled={loadingOverlayType}
                      className={cn(
                        'h-2.5 w-2.5 rounded-[3px]',
                        isCurrentDate &&
                          'border-info-dark data-[state=checked]:bg-info-dark',
                      )}
                      iconClassName='h-2 w-2'
                    />
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
                    Shipping Data
                  </Typography>
                  <Separator className='mt-1.5' />
                </div>
                <div className='flex flex-col gap-1.5'>
                  <div className='flex items-center space-x-2'>
                    <Checkbox
                      id='htsgwsfc'
                      checked={shippingLane}
                      onCheckedChange={(checked) =>
                        setShippingLane(checked === true)
                      }
                      disabled={maptilerSet}
                    />
                    <label
                      htmlFor='htsgwsfc'
                      className='text-sm font-medium leading-none text-gray-800'
                    >
                      Shipping Lane
                    </label>
                  </div>
                  <div className='flex items-center space-x-2'>
                    <Checkbox
                      id='perpwsfc'
                      checked={boundary}
                      onCheckedChange={(checked) =>
                        setBoundary(checked === true)
                      }
                      disabled={maptilerSet}
                    />
                    <label
                      htmlFor='perpwsfc'
                      className='text-sm font-medium leading-none text-gray-800'
                    >
                      Territorial Sea
                    </label>
                  </div>
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
