'use client';
import React from 'react';
import dynamic from 'next/dynamic';
import Typography from '@/components/Typography';

// Memuat komponen Gauge secara dinamis untuk mendukung rendering di sisi klien
const GaugeComponent = dynamic(() => import('react-gauge-component'), {
  ssr: false, // Menonaktifkan server-side rendering untuk komponen ini
});

interface EnviromentalSeekerProps {
  data: {
    waveHeight: number; // Tinggi gelombang dalam meter
    wavePeriod: number; // Periode gelombang dalam detik
    waveHeading: number; // Arah gelombang dalam derajat
    roll: number; // Kemiringan kapal (roll) dalam derajat
    heave: number; // Pergerakan vertikal kapal (heave) dalam meter
    pitch: number; // Kemiringan kapal depan-belakang (pitch) dalam derajat
  };
  maxValues: {
    waveHeight: number; // Nilai maksimum tinggi gelombang
    wavePeriod: number; // Nilai maksimum periode gelombang
    waveHeading: number; // Nilai maksimum arah gelombang
  };
}

export default function EnviromentalSeeker({
  data,
  maxValues,
}: EnviromentalSeekerProps) {
  // Konfigurasi gauge untuk menampilkan metrik lingkungan
  const gaugeConfig = [
    {
      label: 'Wave Height', // Label untuk tinggi gelombang
      key: 'waveHeight', // Properti data yang direferensikan
      unit: 'm', // Satuan tinggi gelombang
      labelStyle: { fontSize: '40px', fill: '#ffffff' }, // Gaya label nilai
      arc: [
        // Konfigurasi warna berdasarkan persentase
        { percentage: 0.25, color: '#EC8F5E' },
        { percentage: 0.5, color: '#F3B664' },
        { percentage: 0.75, color: '#F1EB90' },
        { percentage: 1, color: '#9FBB73' },
      ],
    },
    {
      label: 'Wave Period', // Label untuk periode gelombang
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
      label: 'Wave Heading', // Label untuk arah gelombang
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

  // Konfigurasi untuk metrik pergerakan kapal
  const motionMetrics = [
    {
      label: 'Roll', // Label untuk roll
      key: 'roll',
      unit: 'deg.',
      color: 'bg-warning-normal', // Warna latar belakang
      textColor: 'text-warning-normal', // Warna teks
    },
    {
      label: 'Heave', // Label untuk heave
      key: 'heave',
      unit: 'm',
      color: 'bg-info-normal',
      textColor: 'text-info-normal',
    },
    {
      label: 'Pitch', // Label untuk pitch
      key: 'pitch',
      unit: 'deg.',
      color: 'bg-success-light',
      textColor: 'text-success-light',
    },
  ];

  return (
    <div className='flex justify-between'>
      {/* Bagian gauge untuk menampilkan data gelombang */}
      <section className='flex w-full items-center gap-4'>
        {gaugeConfig.map(({ label, key, unit, arc, labelStyle }) => (
          <div key={key} className='flex items-center'>
            <GaugeComponent
              arc={{
                subArcs: arc.map(({ percentage, color }) => ({
                  limit:
                    percentage *
                    maxValues[
                      key as keyof EnviromentalSeekerProps['maxValues']
                    ],
                  color,
                })),
              }}
              type='radial'
              value={data[key as keyof EnviromentalSeekerProps['data']]}
              maxValue={
                maxValues[key as keyof EnviromentalSeekerProps['maxValues']]
              }
              className='w-[90px]'
              labels={{
                valueLabel: {
                  style: labelStyle,
                  formatTextValue: (value) => `${value.toFixed(1)} ${unit}`,
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
                {data[key as keyof EnviromentalSeekerProps['data']]}
                <span>{unit}</span>
              </Typography>
            </div>
          </div>
        ))}
      </section>

      {/* Bagian metrik pergerakan kapal */}
      <section className='flex w-full items-center justify-end gap-3 pr-4'>
        {motionMetrics.map(({ label, key, unit, color, textColor }) => (
          <div
            key={key}
            className='relative flex h-14 w-[100px] items-center gap-3 rounded-md border-[1.5px] border-typo-normal-white px-3 py-2'
          >
            <div className={`h-full w-1 ${color}`}></div>
            <div className='flex flex-col'>
              <Typography variant='b4' weight='semibold' className={textColor}>
                {label}
              </Typography>
              <Typography variant='t3' weight='medium' className={textColor}>
                {data[key as keyof EnviromentalSeekerProps['data']].toFixed(1)}{' '}
                {unit}
              </Typography>
            </div>
          </div>
        ))}
      </section>
    </div>
  );
}
