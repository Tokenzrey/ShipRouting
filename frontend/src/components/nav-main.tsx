'use client';

import * as React from 'react';
import { ChevronDown, Plus, X } from 'lucide-react';
import Typography from './Typography';
import { shipSpecification } from '@/contents/sidebarContents';
import { Separator } from './ui/separator';
import { Input } from './ui/input';
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from './ui/select';
import IconButton from './buttons/IconButton';
import {
  useRouteStore,
  PathPoint,
  BlockedEdge,
  Keyframes,
} from '@/lib/GlobalState/state';
import Button from './buttons/Button';
import axios from 'axios';

export interface ResponseData {
  success: boolean;
  data: {
    path: PathPoint[];
    distance: number;
    partial_paths: PathPoint[];
    all_edges: BlockedEdge[];
  };
}

/**
 * Fungsi untuk menghitung jarak dan durasi serta mengatur keyframes lokal.
 *
 * @param useModel - Boolean untuk menentukan apakah model digunakan.
 * @param locations - Array lokasi dengan tipe 'from' dan 'destination'.
 * @param shipSpeed - Kecepatan kapal.
 * @param loadCondition - Kondisi load kapal ('ballast' atau lainnya).
 * @param routeCondition - Kondisi rute ('normal' atau 'safest').
 * @param resetKeyframes - Fungsi untuk mereset keyframes.
 * @param setOptimalKeyframes - Setter untuk keyframes rute optimal.
 * @param setSafestKeyframes - Setter untuk keyframes rute teraman.
 * @param setOptimalDistance - Setter untuk jarak rute optimal.
 * @param setOptimalDuration - Setter untuk durasi rute optimal.
 * @param setOptimalRoute - Setter untuk path rute optimal.
 * @param setSafestDistance - Setter untuk jarak rute teraman.
 * @param setSafestDuration - Setter untuk durasi rute teraman.
 * @param setSafestRoute - Setter untuk path rute teraman.
 * @param isCalculating - Boolean yang menandakan apakah sedang menghitung.
 * @param setRouteSelected - Setter untuk memilih rute.
 */
export const calculateDistanceAndDuration = async (
  useModel: boolean,
  locations: Array<{ type: string; longitude: number; latitude: number }>,
  shipSpeed: number,
  loadCondition: string,
  routeCondition: 'normal' | 'safest',
  resetKeyframes: () => void,
  setOptimalKeyframes: (keyframes: Keyframes | null) => void,
  setSafestKeyframes: (keyframes: Keyframes | null) => void,
  setOptimalDistance: (distance: number | null) => void,
  setOptimalDuration: (duration: number) => void,
  setOptimalRoute: (path: PathPoint[]) => void,
  setSafestDistance: (distance: number) => void,
  setSafestDuration: (duration: number) => void,
  setSafestRoute: (path: PathPoint[]) => void,
  isCalculating: Boolean,
  setRouteSelected: (route: 'normal' | 'safest' | 'None') => void,
) => {
  const fromLocation = locations.find((loc) => loc.type === 'from');
  const destinationLocation = locations.find(
    (loc) => loc.type === 'destination',
  );

  // Prevent API call if either 'from' or 'destination' is missing
  if (!fromLocation || !destinationLocation) {
    console.warn('Both "from" and "destination" locations must be set.');
    return;
  }

  // Reset keyframes sebelum memulai perhitungan baru
  resetKeyframes();

  try {
    const response = await axios.post<ResponseData>(
      'http://localhost:5000/api/djikstra',
      {
        start: {
          longitude: fromLocation.longitude,
          latitude: fromLocation.latitude,
        },
        end: {
          longitude: destinationLocation.longitude,
          latitude: destinationLocation.latitude,
        },
        ship_speed: shipSpeed,
        condition: loadCondition === 'ballast' ? 1 : 0,
        use_model: useModel,
      },
    );

    const data = response.data;

    // Periksa kesalahan dalam respon
    if (!data.success) {
      throw new Error('Unknown API error');
    }

    // Validasi keyframes
    if (
      data.data &&
      Array.isArray(data.data.partial_paths) &&
      Array.isArray(data.data.all_edges) &&
      data.data.path &&
      Array.isArray(data.data.path)
    ) {
      const keyframes: Keyframes = {
        partial_path: data.data.partial_paths,
        final_path: {
          path: data.data.path,
          distance: data.data.distance,
        },
        all_edges: data.data.all_edges,
      };

      // Tergantung pada kondisi rute, perbarui state yang sesuai
      if (routeCondition === 'normal') {
        setOptimalDistance(keyframes.final_path.distance);
        setOptimalDuration(keyframes.final_path.distance / shipSpeed);
        setOptimalRoute(keyframes.final_path.path);
        setOptimalKeyframes(keyframes); // Menggunakan setter lokal
        if (isCalculating) {
          setRouteSelected('normal');
        }
      } else if (routeCondition === 'safest') {
        setSafestDistance(keyframes.final_path.distance);
        setSafestDuration(keyframes.final_path.distance / shipSpeed);
        setSafestRoute(keyframes.final_path.path);
        setSafestKeyframes(keyframes); // Menggunakan setter lokal
        if (isCalculating) {
          setRouteSelected('safest');
        }
      }

      console.log(keyframes.final_path);
      console.log(keyframes.partial_path);
      console.log(`Route condition: ${routeCondition}`);
      console.log(`Final path: ${JSON.stringify(keyframes.final_path)}`);
      console.log(`Partial paths count: ${keyframes.partial_path.length}`);
      console.log(`All edges count: ${keyframes.all_edges.length}`);
    } else {
      console.error('Invalid keyframes structure from API:', data.data);
      throw new Error('Invalid keyframes structure from API.');
    }
  } catch (error: any) {
    if (error.response && error.response.data && error.response.data.detail) {
      console.error('API Error:', error.response.data.detail);
    } else {
      console.error('Error calculating route:', error.message || error);
    }
    // Opsional: Anda dapat mengatur state error di sini untuk ditampilkan di UI
  }
};

export function NavMain() {
  // Access the global state
  const {
    locations,
    safestDistance,
    safestDuration,
    optimalDistance,
    optimalDuration,
    shipSpeed,
    loadCondition,
    isCalculating,
    setLocationTypeToAdd,
    removeLocation,
    setOptimalDistance,
    setSafestDistance,
    setOptimalDuration,
    setSafestDuration,
    setOptimalRoute,
    setSafestRoute,
    setShipSpeed,
    setLoadCondition,
    resetKeyframes,
    setRouteSelected,
    setIsUpdate,
  } = useRouteStore();

  const [routeCondition, setRouteCondition] = React.useState<
    'normal' | 'safest'
  >('normal');

  // Handle button clicks
  const handleNormalClick = () => {
    setRouteCondition('normal');
    console.log(isCalculating);
    // calculateDistanceAndDuration(false);
    if (!isCalculating) {
      setRouteSelected('normal');
      setIsUpdate(true);
    }
  };

  const handleSafestClick = () => {
    setRouteCondition('safest');
    // calculateDistanceAndDuration(true);
    if (!isCalculating) {
      setRouteSelected('safest');
      setIsUpdate(true);
    }
  };

  // Handle location removal
  const handleRemoveLocation = (index: number) => {
    removeLocation(index);
    // Clear routes and distances if either 'from' or 'destination' is removed
    const hasFrom = locations.some((loc) => loc.type === 'from');
    const hasDestination = locations.some((loc) => loc.type === 'destination');

    if (!hasFrom || !hasDestination) {
      setRouteSelected('None');
      setOptimalDistance(null);
      setSafestDistance(null);
      setOptimalDuration(null);
      setSafestDuration(null);
      setOptimalRoute([]);
      setSafestRoute([]);
      resetKeyframes();
    }
  };

  return (
    <section className='my-2.5'>
      <div className='px-2.5'>
        <Typography weight='semibold' className='mb-[0.2rem]' variant='b2'>
          SHIP PARTICULAR
        </Typography>
        <div className='mb-[0.2rem] space-y-[0.15rem]'>
          {shipSpecification.map((item, index) => (
            <section key={index} className='flex'>
              <Typography weight='medium' className='w-[35%]'>
                {item.label}
              </Typography>
              <Typography weight='medium' className='w-[65%]'>
                : {item.value}
              </Typography>
            </section>
          ))}
        </div>
        <Separator className='mt-2.5' />
      </div>
      <div className='mt-2.5 px-2.5'>
        <Typography weight='semibold' className='mb-[0.2rem]' variant='b2'>
          SHIP OPERATIONAL DATA
        </Typography>
        <div className='mb-[0.15rem] space-y-2.5'>
          {/* Input Ship Speed */}
          <div className='flex items-center'>
            <Typography variant='b4' weight='medium' className='w-[30%]'>
              Ship Speed
            </Typography>
            <div className='flex w-[55%] items-center gap-2'>
              <Input
                type='number'
                value={shipSpeed || ''}
                onChange={(e) => setShipSpeed(Number(e.target.value))}
                className='h-5 w-12 rounded border-0 px-2 ring-0 md:text-[0.5625rem] md:leading-[0.8125rem]'
                placeholder='Enter speed'
                min='0'
              />
              <Typography variant='b4'>Kn.</Typography>
            </div>
          </div>

          {/* Select Ship Condition */}
          <div className='flex items-center'>
            <Typography variant='b4' weight='medium' className='w-[30%]'>
              Load Cond.
            </Typography>
            <div className='w-[35%]'>
              <Select
                value={loadCondition || 'full_load'}
                onValueChange={setLoadCondition}
              >
                <SelectTrigger className='h-5 md:text-[0.5625rem] md:leading-[0.8125rem]'>
                  <SelectValue placeholder='Select Ship Condition'>
                    {loadCondition === 'full_load' && 'Full Load'}
                    {loadCondition === 'ballast' && 'Ballast'}
                  </SelectValue>
                </SelectTrigger>
                <SelectContent className='md:text-[0.5625rem] md:leading-[0.8125rem]'>
                  <SelectItem value='full_load'>Full Load</SelectItem>
                  <SelectItem value='ballast'>Ballast</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>
        <Separator className='mt-3' />
      </div>
      <div className='mt-2.5 px-2.5'>
        <Typography weight='semibold' className='mb-1.5' variant='b2'>
          Route Planner
        </Typography>
        <div className='ml-1.5 space-y-2.5'>
          <section className='flex flex-col gap-2.5'>
            {!locations.some((loc) => loc.type === 'from') && (
              <div className='flex items-center gap-2'>
                <div
                  className='flex h-7 w-7 cursor-pointer items-center justify-center rounded-full border-[1.5px] border-typo-normal-white text-typo-normal-white'
                  onClick={() => setLocationTypeToAdd('from')}
                >
                  <Plus size={16} color='#ffffff' />
                </div>
                <Typography
                  className='text-typo-normal-white'
                  variant='b3'
                  weight='semibold'
                >
                  Add From
                </Typography>
              </div>
            )}

            {locations.map((location, index) => (
              <div key={index} className='relative flex flex-col gap-2'>
                <div className='relative flex items-center'>
                  <div className='flex h-7 w-7 items-center justify-center rounded-full border-[1.5px] border-typo-normal-white text-typo-normal-white'>
                    <Typography
                      className='text-typo-normal-white'
                      weight='medium'
                      variant='b2'
                    >
                      {location.type === 'from' ? 'F' : 'D'}
                    </Typography>
                  </div>
                  <div className='ml-2 flex w-[calc(100%-3rem)] items-center justify-between'>
                    <Typography
                      className='w-full truncate text-typo-normal-white'
                      variant='b3'
                      weight='semibold'
                    >
                      {location.name}
                    </Typography>
                    <IconButton
                      variant='danger'
                      onClick={() => handleRemoveLocation(index)}
                      Icon={X}
                      IconClassName='text-[12px]'
                      className='h-5 w-5'
                      size='small'
                    />
                  </div>
                </div>

                {locations.some((loc) => loc.type === 'from') &&
                  locations.some((loc) => loc.type === 'destination') &&
                  index === 0 && (
                    <div className='flex-start flex items-center gap-1'>
                      <div className='relative flex h-fit w-7 flex-col items-center'>
                        <div className='h-1.5 w-1.5 -translate-x-[0.4px] rounded-full bg-typo-normal-white'></div>
                        <div className='h-16 w-[1.5px] bg-typo-normal-white'></div>
                        <ChevronDown
                          className='absolute -bottom-2.5 text-typo-normal-white'
                          strokeWidth={1.5}
                        />
                      </div>
                      <div className='flex w-[calc(100%-52px)] flex-col gap-1'>
                        <div className='flex flex-col'>
                          <Typography
                            className='text-typo-normal-white'
                            variant='b3'
                            weight='medium'
                          >
                            Distance
                          </Typography>
                          <Typography
                            className='text-typo-normal-secondary'
                            variant='b5'
                            weight='medium'
                          >
                            {routeCondition === 'normal'
                              ? optimalDistance
                                ? `${optimalDistance} km`
                                : 'Not Available'
                              : safestDistance
                                ? `${safestDistance} km`
                                : 'Not Available'}
                          </Typography>
                        </div>
                        <Separator />
                        <div className='flex flex-col'>
                          <Typography
                            className='text-typo-normal-white'
                            variant='b3'
                            weight='medium'
                          >
                            Duration
                          </Typography>
                          <Typography
                            className='text-typo-normal-secondary'
                            variant='b5'
                            weight='medium'
                          >
                            {routeCondition === 'normal'
                              ? optimalDuration
                                ? `${optimalDuration.toFixed(2)} hr`
                                : 'Not Available'
                              : safestDuration
                                ? `${safestDuration.toFixed(2)} hr`
                                : 'Not Available'}
                          </Typography>
                        </div>
                      </div>
                    </div>
                  )}
              </div>
            ))}

            {locations.some((loc) => loc.type === 'from') &&
              !locations.some((loc) => loc.type === 'destination') && (
                <div className='flex items-center gap-2'>
                  <div
                    className='flex h-7 w-7 cursor-pointer items-center justify-center rounded-full border-[1.5px] border-typo-normal-white text-typo-normal-white'
                    onClick={() => setLocationTypeToAdd('destination')}
                  >
                    <Plus size={16} color='#ffffff' />
                  </div>
                  <Typography
                    className='text-typo-normal-white'
                    variant='b3'
                    weight='semibold'
                  >
                    Add Destination
                  </Typography>
                </div>
              )}

            {/* Optimal and Safest Route Buttons */}
            {locations.some((loc) => loc.type === 'from') &&
              locations.some((loc) => loc.type === 'destination') && (
                <div className='mt-1 flex justify-center gap-2'>
                  <Button
                    variant='success'
                    appearance='dark'
                    className='rounded-sm'
                    size='small'
                    onClick={handleNormalClick}
                  >
                    Normal
                  </Button>
                  <Button
                    appearance='dark'
                    className='rounded-sm'
                    size='small'
                    onClick={handleSafestClick}
                  >
                    Safest
                  </Button>
                </div>
              )}
          </section>
        </div>
      </div>
    </section>
  );
}
