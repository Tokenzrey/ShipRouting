'use client';
import React from 'react';
import dynamic from 'next/dynamic';
import Typography from '@/components/Typography';
import { useRouteStore } from '@/lib/GlobalState/state';

// Memuat komponen Gauge secara dinamis
const GaugeComponent = dynamic(() => import('react-gauge-component'), {
  ssr: false,
});
interface EnviromentalSeekerProps {
  datasets: {
    enviromental: {
      label: string;
      data: number[];
      yAxisID: string;
      color: string;
    }[];
    ship_motion: {
      label: string;
      data: number[];
      yAxisID: string;
      color: string;
    }[];
  };
}
export default function EnviromentalSeeker({
  datasets,
}: EnviromentalSeekerProps) {
  const currentAnimationIndex = useRouteStore(
    (state) => state.currentAnimationIndex,
  );

  const [seekerData, setSeekerData] = React.useState({
    waveHeight: 0,
    wavePeriod: 0,
    waveHeading: 0,
    roll: 0,
    heave: 0,
    pitch: 0,
  });

  const maxValues = {
    waveHeight: Math.max(...datasets.enviromental[0].data, 5),
    wavePeriod: Math.max(...datasets.enviromental[1].data, 15),
    waveHeading: 360, // Heading is always 0-360
  };

  React.useEffect(() => {
    // Periksa apakah datasets memiliki data
    if (
      datasets.enviromental?.length > 0 &&
      datasets.ship_motion?.length > 0 &&
      currentAnimationIndex !== undefined
    ) {
      const updatedSeekerData = {
        waveHeight:
          datasets.enviromental[0]?.data[currentAnimationIndex || 0] || 0,
        wavePeriod:
          datasets.enviromental[1]?.data[currentAnimationIndex || 0] || 0,
        waveHeading:
          datasets.enviromental[2]?.data[currentAnimationIndex || 0] || 0,
        roll: datasets.ship_motion[0]?.data[currentAnimationIndex || 0] || 0,
        heave: datasets.ship_motion[1]?.data[currentAnimationIndex || 0] || 0,
        pitch: datasets.ship_motion[2]?.data[currentAnimationIndex || 0] || 0,
      };

      setSeekerData(updatedSeekerData);
    }
    console.log(currentAnimationIndex);
  }, [currentAnimationIndex, datasets]);

  // Konfigurasi gauge
  const gaugeConfig = [
    {
      label: 'Wave Height',
      key: 'waveHeight',
      unit: 'm',
      labelStyle: { fontSize: '40px', fill: '#ffffff' },
      arc: [
        { percentage: 0.25, color: '#EC8F5E' },
        { percentage: 0.5, color: '#F3B664' },
        { percentage: 0.75, color: '#F1EB90' },
        { percentage: 1, color: '#9FBB73' },
      ],
    },
    {
      label: 'Wave Period',
      key: 'wavePeriod',
      unit: 's',
      labelStyle: { fontSize: '50px', fill: '#ffffff' },
      arc: [
        { percentage: 0.33, color: '#453C67' },
        { percentage: 0.66, color: '#6D67E4' },
        { percentage: 1, color: '#46C2CB' },
      ],
    },
    {
      label: 'Wave Heading',
      key: 'waveHeading',
      unit: 'deg.',
      labelStyle: { fontSize: '80px', fill: '#ffffff' },
      arc: [
        { percentage: 0.25, color: '#EA4228' },
        { percentage: 0.5, color: '#F58B19' },
        { percentage: 0.75, color: '#F5CD19' },
        { percentage: 1, color: '#5BE12C' },
      ],
    },
  ];

  // Konfigurasi metrik pergerakan kapal
  const motionMetrics = [
    {
      label: 'Roll',
      key: 'roll',
      unit: 'deg.',
      color: 'bg-warning-normal',
      textColor: 'text-warning-normal',
    },
    {
      label: 'Heave',
      key: 'heave',
      unit: 'm',
      color: 'bg-info-normal',
      textColor: 'text-info-normal',
    },
    {
      label: 'Pitch',
      key: 'pitch',
      unit: 'deg.',
      color: 'bg-success-light',
      textColor: 'text-success-light',
    },
  ];

  return (
    <div className='flex justify-between'>
      {/* Gauge untuk data lingkungan */}
      <section className='flex w-full items-center gap-4'>
        {gaugeConfig.map(({ label, key, unit, arc, labelStyle }) => {
          const value = Number(seekerData[key as keyof typeof seekerData]) || 0; // Konversi ke number
          return (
            <div key={key} className='flex items-center'>
              <GaugeComponent
                arc={{
                  subArcs: arc.map(({ percentage, color }) => ({
                    limit:
                      percentage * maxValues[key as keyof typeof maxValues],
                    color,
                  })),
                }}
                type='radial'
                value={value}
                maxValue={maxValues[key as keyof typeof maxValues]}
                className='w-[90px]'
                labels={{
                  valueLabel: {
                    style: labelStyle,
                    formatTextValue: (val) => `${val.toFixed(1)} ${unit}`,
                  },
                  tickLabels: {
                    type: 'inner',
                    defaultTickValueConfig: {
                      style: { fontSize: '6px', fill: '#ffffff' },
                    },
                  },
                }}
              />
              <div>
                <Typography
                  weight='semibold'
                  className='text-xs text-typo-normal-white md:text-xs'
                >
                  {label}
                </Typography>
                <Typography className='text-xs text-typo-normal-white md:text-xs'>
                  {value.toFixed(1)}
                  <span>{unit}</span>
                </Typography>
              </div>
            </div>
          );
        })}
      </section>

      {/* Metrik pergerakan kapal */}
      <section className='flex w-full items-center justify-end gap-3 pr-4'>
        {motionMetrics.map(({ label, key, unit, color, textColor }) => {
          const value = Number(seekerData[key as keyof typeof seekerData]) || 0; // Konversi ke number
          return (
            <div
              key={key}
              className='relative flex h-14 w-[100px] items-center gap-3 rounded-md border-[1.5px] border-typo-normal-white px-3 py-2'
            >
              <div className={`h-full w-1 ${color}`}></div>
              <div className='flex flex-col'>
                <Typography
                  variant='b4'
                  weight='semibold'
                  className={textColor}
                >
                  {label}
                </Typography>
                <Typography variant='t3' weight='medium' className={textColor}>
                  {value.toFixed(1)} {unit}
                </Typography>
              </div>
            </div>
          );
        })}
      </section>
    </div>
  );
}
