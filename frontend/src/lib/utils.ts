import { fromLonLat } from 'ol/proj';
import { CONFIG } from '@/constant/map';

export interface GridCenter {
  lon: number;
  lat: number;
}

export const findNearestGridCenter = (
  clickedLon: number,
  clickedLat: number,
): GridCenter => {
  const [minLon, minLat, maxLon, maxLat] = CONFIG.INDONESIA_EXTENT;
  const gridSpacing = 1 / 111.32;

  const lonOffset = clickedLon - minLon;
  const latOffset = clickedLat - minLat;

  const gridLon =
    minLon +
    Math.floor(lonOffset / gridSpacing) * gridSpacing +
    gridSpacing / 2;
  const gridLat =
    minLat +
    Math.floor(latOffset / gridSpacing) * gridSpacing +
    gridSpacing / 2;

  return {
    lon: Math.max(minLon, Math.min(gridLon, maxLon)),
    lat: Math.max(minLat, Math.min(gridLat, maxLat)),
  };
};

export const fromLonLatExtent = (
  extent: [number, number, number, number],
): [number, number, number, number] => {
  const [minLon, minLat, maxLon, maxLat] = extent;
  const bottomLeft = fromLonLat([minLon, minLat]) as [number, number];
  const topRight = fromLonLat([maxLon, maxLat]) as [number, number];
  return [bottomLeft[0], bottomLeft[1], topRight[0], topRight[1]];
};
