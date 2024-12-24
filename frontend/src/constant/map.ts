export const MAPTILER_API_KEY = 'hKcj2yBSfnwPvNZKwL8F';

export const CONFIG = {
  MAP_CENTER: [117.92, -2.56] as [number, number],
  MAP_ZOOM: 5,
  MIN_ZOOM: 4,
  MAX_ZOOM: 14,
  INDONESIA_EXTENT: [92.0, -11, 141.0, 9.5] as [number, number, number, number],
  GEOJSON_URL: '/contents/eez.json',
  STYLE_URL: `https://api.maptiler.com/maps/9c6239d7-c933-48b0-bb76-bb4aa3d0dde9/style.json?key=${MAPTILER_API_KEY}`,
};
