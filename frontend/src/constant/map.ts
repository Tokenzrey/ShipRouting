export const MAPTILER_API_KEY = 'hKcj2yBSfnwPvNZKwL8F';

export const CONFIG = {
  MAP_CENTER: [117.92, -2.56] as [number, number],
  MAP_ZOOM: 5,
  MIN_ZOOM: 4,
  MAX_ZOOM: 14,
  INDONESIA_EXTENT: [92.0, -11, 141.0, 9.5] as [number, number, number, number],
  GEOJSON_URL: '/contents/eez.json',
  STYLE_URL_ALL: `https://api.maptiler.com/maps/9c6239d7-c933-48b0-bb76-bb4aa3d0dde9/style.json?key=${MAPTILER_API_KEY}`,
  STYLE_URL_BOUNDARY: `https://api.maptiler.com/maps/c8b12fa4-8ea7-479e-a8ab-0ab7547a9e9c/style.json?key=${MAPTILER_API_KEY}`,
  STYLE_URL_SHIPPINGLANE: `https://api.maptiler.com/maps/85324a98-483e-454f-b8a9-632802f21dcb/style.json?key=${MAPTILER_API_KEY}`,
  STYLE_URL_VANILLA: `https://api.maptiler.com/maps/c33ac5bd-217e-4977-bab6-56198acbb22b/style.json?key=${MAPTILER_API_KEY}`,

};
