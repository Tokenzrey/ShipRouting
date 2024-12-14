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
import { useRouteStore } from '@/lib/GlobalState/state';
import Button from './buttons/Button';

export function NavMain() {
  // Access the global state
  const {
    locations,
    distance,
    duration,
    shipSpeed,
    loadCondition,
    setLocationTypeToAdd,
    removeLocation,
    setDistance,
    setDuration,
    setLoadCondition,
    setShipSpeed,
  } = useRouteStore();

  const calculateDistanceAndDuration = () => {
    const fromLocation = locations.find((loc) => loc.type === 'from');
    const destinationLocation = locations.find(
      (loc) => loc.type === 'destination',
    );

    if (fromLocation && destinationLocation) {
      // Replace these with your actual calculation logic
      setDistance(13436.5); // Dummy distance
      setDuration(126); // Dummy duration in hours
    } else {
      setDistance(0);
      setDuration(0);
    }
  };

  // useEffect to trigger calculation on locations change
  React.useEffect(() => {
    calculateDistanceAndDuration();
  }, [locations]);

  return (
    <section className='my-4'>
      <div className='px-4'>
        <Typography weight='semibold' className='mb-1.5' variant='b2'>
          SHIP PARTICULAR
        </Typography>
        <div className='ml-1.5 space-y-1'>
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
        <Separator className='mt-3' />
      </div>
      <div className='mt-5 px-4'>
        <Typography weight='semibold' className='mb-1.5' variant='b2'>
          SHIP OPERATIONAL DATA
        </Typography>
        <div className='ml-1.5 space-y-4'>
          {/* Input Ship Speed */}
          <div className='flex items-center'>
            <Typography weight='medium' className='w-[45%]'>
              Ship Speed
            </Typography>
            <div className='flex w-[55%] items-center gap-2'>
              <Input
                type='number'
                value={shipSpeed || ''}
                onChange={(e) => setShipSpeed(Number(e.target.value))}
                className='h-7 w-1/2 rounded border-0 px-2 ring-0'
                placeholder='Enter speed'
                min='0'
              />
              <Typography>Kn.</Typography>
            </div>
          </div>

          {/* Select Ship Condition */}
          <div className='flex items-center'>
            <Typography weight='medium' className='w-[45%]'>
              Load Cond.
            </Typography>
            <div className='w-[55%]'>
              <Select
                value={loadCondition || 'full_load'}
                onValueChange={setLoadCondition}
              >
                <SelectTrigger className='h-7'>
                  <SelectValue placeholder='Select Ship Condition'>
                    {loadCondition === 'full_load' && 'Full Load'}
                    {loadCondition === 'ballast' && 'Ballast'}
                  </SelectValue>
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value='full_load'>Full Load</SelectItem>
                  <SelectItem value='ballast'>Ballast</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>
        <Separator className='mt-3' />
      </div>
      <div className='mt-5 px-4'>
        <Typography weight='semibold' className='mb-1.5' variant='b2'>
          Route Planner
        </Typography>
        <div className='ml-1.5 space-y-4'>
          <section className='flex flex-col gap-4'>
            {!locations.some((loc) => loc.type === 'from') && (
              <div className='flex items-center gap-3'>
                <div
                  className='flex h-10 w-10 cursor-pointer items-center justify-center rounded-full border-2 border-typo-normal-white text-typo-normal-white'
                  onClick={() => setLocationTypeToAdd('from')}
                >
                  <Plus size={28} color='#ffffff' />
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
              <div key={index} className='relative flex flex-col gap-3'>
                <div className='relative flex items-center'>
                  <div className='flex h-10 w-10 items-center justify-center rounded-full border-2 border-typo-normal-white text-typo-normal-white'>
                    <Typography
                      className='text-typo-normal-white'
                      weight='medium'
                      variant='b2'
                    >
                      {location.type === 'from' ? 'F' : 'D'}
                    </Typography>
                  </div>
                  <div className='ml-3 flex w-[calc(100%-3rem)] items-center justify-between'>
                    <Typography
                      className='w-min text-typo-normal-white'
                      variant='b3'
                      weight='semibold'
                    >
                      {location.name}
                    </Typography>
                    <IconButton
                      variant='danger'
                      onClick={() => removeLocation(index)}
                      Icon={X}
                      size='small'
                    />
                  </div>
                </div>

                {locations.some((loc) => loc.type === 'from') &&
                  locations.some((loc) => loc.type === 'destination') &&
                  index === 0 && (
                    <div className='flex items-center gap-3'>
                      <div className='relative flex h-fit w-10 flex-col items-center'>
                        <div className='h-3 w-3 rounded-full bg-typo-normal-white'></div>
                        <div className='h-20 w-[2.5px] bg-typo-normal-white'></div>
                        <ChevronDown
                          className='absolute -bottom-2.5 text-typo-normal-white'
                          strokeWidth={2.5}
                        />
                      </div>
                      <div className='flex w-[calc(100%-52px)] flex-col gap-2'>
                        <div className='flex flex-col'>
                          <Typography
                            className='text-typo-normal-white'
                            variant='b3'
                            font='futura'
                            weight='medium'
                          >
                            Distance
                          </Typography>
                          <Typography
                            className='text-typo-normal-secondary'
                            variant='b5'
                            font='futura'
                            weight='medium'
                          >
                            {distance ? `${distance} m` : 'Not Available'}
                          </Typography>
                        </div>
                        <Separator />
                        <div className='flex flex-col'>
                          <Typography
                            className='text-typo-normal-white'
                            variant='b3'
                            font='futura'
                            weight='medium'
                          >
                            Duration
                          </Typography>
                          <Typography
                            className='text-typo-normal-secondary'
                            variant='b5'
                            font='futura'
                            weight='medium'
                          >
                            {duration ? `${duration} hr` : 'Not Available'}
                          </Typography>
                        </div>
                      </div>
                    </div>
                  )}
              </div>
            ))}

            {locations.some((loc) => loc.type === 'from') &&
              !locations.some((loc) => loc.type === 'destination') && (
                <div className='flex items-center gap-3'>
                  <div
                    className='flex h-10 w-10 cursor-pointer items-center justify-center rounded-full border-2 border-typo-normal-white text-typo-normal-white'
                    onClick={() => setLocationTypeToAdd('destination')}
                  >
                    <Plus size={28} color='#ffffff' />
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
                <div className='mt-2 flex justify-center gap-3'>
                  <Button
                    variant='success'
                    appearance='dark'
                    className='rounded-md'
                  >
                    Normal
                  </Button>
                  <Button appearance='dark' className='rounded-md'>
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
