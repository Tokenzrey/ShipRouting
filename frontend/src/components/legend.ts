// src/utils/legend.ts

import {
  getColorForValue,
  OverlayType,
} from '@/app/dashboard/components/OverlayHandler';
export interface LegendStep {
  color: string;
  from: number;
  to: number;
}

/**
 * Membuat array interval legend, berjumlah `numSteps`.
 * @param minValue nilai minimum wave
 * @param maxValue nilai maksimum wave
 * @param numSteps jumlah interval
 * @param overlayType tipe overlay, misal 'htsgwsfc' | 'perpwsfc' | 'none'
 * @param getColorFn fungsi untuk menghitung warna berdasarkan value
 * @returns array LegendStep
 */
export function getLegendSteps(
  minValue: number,
  maxValue: number,
  numSteps: number,
  overlayType: OverlayType,
  getColorFn: (
    value: number,
    minVal: number,
    maxVal: number,
    overlayType: OverlayType,
  ) => string,
): LegendStep[] {
  const steps: LegendStep[] = [];
  const range = maxValue - minValue;

  if (range <= 0) {
    // Semua nilai sama
    const singleColor = getColorFn(minValue, minValue, maxValue, overlayType);
    return [
      {
        color: singleColor,
        from: minValue,
        to: maxValue,
      },
    ];
  }

  for (let i = 0; i < numSteps; i++) {
    const stepMin = minValue + (range * i) / numSteps;
    const stepMax = minValue + (range * (i + 1)) / numSteps;
    const midpoint = (stepMin + stepMax) / 2;
    const color = getColorFn(midpoint, minValue, maxValue, overlayType);

    steps.push({
      color,
      from: stepMin,
      to: stepMax,
    });
  }

  return steps;
}
