'use client';

import { ChevronDown, Route } from 'lucide-react';
import { useState } from 'react';
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
import Button from './buttons/Button';

export function NavMain({}) {
  // State untuk menyimpan nilai input
  const [shipSpeed, setShipSpeed] = useState<number>(0);
  const [shipCondition, setShipCondition] = useState<string>('full_load');

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
                value={shipSpeed}
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
              <Select value={shipCondition} onValueChange={setShipCondition}>
                <SelectTrigger className='h-7'>
                  <SelectValue placeholder='Select Ship Condition'>
                    {shipCondition === 'full_load' && 'Full Load'}
                    {shipCondition === 'ballast' && 'Ballast'}
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
        <div className='ml-1.5 space-y-1'>
          <section className='flex flex-col gap-4'>
            <div className='flex items-center gap-3'>
              <div className='flex h-10 w-10 items-center justify-center rounded-full border-2 border-typo-normal-white text-typo-normal-white'>
                <Typography
                  className='text-typo-normal-white'
                  weight='medium'
                  variant='b2'
                >
                  D
                </Typography>
              </div>
              <Typography
                className='text-typo-normal-white'
                variant='b3'
                weight='semibold'
              >
                Jayapura
              </Typography>
            </div>
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
                    13436.5 m
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
                    5d. 6hr.
                  </Typography>
                </div>
              </div>
            </div>
            <div className='flex items-center gap-3'>
              <div className='flex h-10 w-10 items-center justify-center rounded-full border-2 border-typo-normal-white text-typo-normal-white'>
                <Typography
                  className='text-typo-normal-white'
                  weight='medium'
                  variant='b2'
                >
                  A
                </Typography>
              </div>
              <Typography
                className='text-typo-normal-white'
                variant='b3'
                weight='semibold'
              >
                Lombok, NTB
              </Typography>
            </div>
            <div className='flex justify-center gap-3 mt-2'>
              <Button
                variant='success'
                appearance='dark'
                className='rounded-md'
              >
                Optimal
              </Button>
              <Button appearance='dark' className='rounded-md'>
                Safest
              </Button>
            </div>
          </section>
        </div>
      </div>
    </section>
  );
}
