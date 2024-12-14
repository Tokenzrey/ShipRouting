import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface Location {
  type: 'from' | 'destination';
  name: string;
  longitude: number; // Longitude koordinat
  latitude: number; // Latitude koordinat
}

interface RouteStore {
  locations: Location[]; // List of locations (from and destination)
  distance: number | null; // Distance in kilometers
  duration: number | null; // Duration in hours
  loadCondition: string; // Load condition (e.g., "Light", "Medium", "Heavy")
  shipSpeed: number; // Ship speed in knots
  optimalRoute: any | null; // Replace `any` with specific type if available
  safestRoute: any | null;
  locationTypeToAdd: 'from' | 'destination' | null;

  // Actions for locations
  addLocation: (location: Location) => void;
  removeLocation: (index: number) => void;

  // Actions for route details
  setDistance: (distance: number) => void;
  setDuration: (duration: number) => void;
  setLoadCondition: (loadCondition: string) => void;
  setShipSpeed: (speed: number) => void;
  setOptimalRoute: (route: any) => void; // Replace `any` with specific type
  setSafestRoute: (route: any) => void; // Replace `any` with specific type
  setLocationTypeToAdd: (type: 'from' | 'destination' | null) => void;
}

export const useRouteStore = create(
  persist<RouteStore>(
    (set) => ({
      // Initial state
      locations: [],
      distance: null,
      duration: null,
      loadCondition: 'full_load', // Default to "Full Load"
      shipSpeed: 10, // Default to 10 knots
      optimalRoute: null,
      safestRoute: null,
      locationTypeToAdd: null,

      // Location actions
      addLocation: (location) =>
        set((state) => ({
          locations: [...state.locations, location].sort((a, b) =>
            a.type === 'from' ? -1 : b.type === 'from' ? 1 : 0,
          ),
        })),
      removeLocation: (index) =>
        set((state) => ({
          locations: state.locations.filter((_, i) => i !== index),
        })),
      setLocationTypeToAdd: (type) =>
        set((state) => {
          if (
            type === 'from' &&
            !state.locations.some((loc: Location) => loc.type === 'from')
          ) {
            return { locationTypeToAdd: 'from' };
          } else if (
            type === 'destination' &&
            !state.locations.some((loc: Location) => loc.type === 'destination')
          ) {
            return { locationTypeToAdd: 'destination' };
          }
          return { locationTypeToAdd: null };
        }),

      // Route details actions
      setDistance: (distance) => set({ distance }),
      setDuration: (duration) => set({ duration }),
      setLoadCondition: (loadCondition) => set({ loadCondition }),
      setShipSpeed: (speed) => set({ shipSpeed: speed }),
      setOptimalRoute: (route) => set({ optimalRoute: route }),
      setSafestRoute: (route) => set({ safestRoute: route }),
    }),
    {
      name: 'route-store', // Key for localStorage
      partialize: (state) => ({
        locations: state.locations,
        distance: state.distance,
        duration: state.duration,
        loadCondition: state.loadCondition,
        shipSpeed: state.shipSpeed,
        locationTypeToAdd: state.locationTypeToAdd,
        // Add placeholders for other fields to satisfy RouteStore type
        optimalRoute: null,
        safestRoute: null,
        addLocation: () => undefined,
        removeLocation: () => undefined,
        setDistance: () => undefined,
        setDuration: () => undefined,
        setLoadCondition: () => undefined,
        setShipSpeed: () => undefined,
        setOptimalRoute: () => undefined,
        setSafestRoute: () => undefined,
        setLocationTypeToAdd: () => undefined,
      }),
    },
  ),
);
