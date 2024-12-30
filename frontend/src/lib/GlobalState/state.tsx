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
  optimalDistance: number | null; // Distance for the optimal route in kilometers
  safestDistance: number | null; // Distance for the safest route in kilometers
  optimalDuration: number | null; // Duration for the optimal route in hours
  safestDuration: number | null; // Duration for the safest route in hours
  loadCondition: string; // Load condition (e.g., "Light", "Medium", "Heavy")
  shipSpeed: number; // Ship speed in knots
  optimalRoute: any | null; // Replace `any` with specific type if available
  safestRoute: any | null; // Replace `any` with specific type if available
  locationTypeToAdd: 'from' | 'destination' | null;

  // Actions for locations
  addLocation: (location: Location) => void;
  removeLocation: (index: number) => void;

  // Actions for route details
  setOptimalDistance: (distance: number | null) => void;
  setSafestDistance: (distance: number | null) => void;
  setOptimalDuration: (duration: number | null) => void;
  setSafestDuration: (duration: number | null) => void;
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
      optimalDistance: null,
      safestDistance: null,
      optimalDuration: null,
      safestDuration: null,
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
      setOptimalDistance: (distance) => set({ optimalDistance: distance }),
      setSafestDistance: (distance) => set({ safestDistance: distance }),
      setOptimalDuration: (duration) => set({ optimalDuration: duration }),
      setSafestDuration: (duration) => set({ safestDuration: duration }),
      setLoadCondition: (loadCondition) => set({ loadCondition }),
      setShipSpeed: (speed) => set({ shipSpeed: speed }),
      setOptimalRoute: (route) => set({ optimalRoute: route }),
      setSafestRoute: (route) => set({ safestRoute: route }),
    }),
    {
      name: 'route-store', // Key for localStorage
      partialize: (state) => ({
        locations: state.locations,
        optimalDistance: state.optimalDistance,
        safestDistance: state.safestDistance,
        optimalDuration: state.optimalDuration,
        safestDuration: state.safestDuration,
        loadCondition: state.loadCondition,
        shipSpeed: state.shipSpeed,
        locationTypeToAdd: state.locationTypeToAdd,
        // Add placeholders for other fields to satisfy RouteStore type
        optimalRoute: null,
        safestRoute: null,
        addLocation: () => undefined,
        removeLocation: () => undefined,
        setOptimalDistance: () => undefined,
        setSafestDistance: () => undefined,
        setOptimalDuration: () => undefined,
        setSafestDuration: () => undefined,
        setLoadCondition: () => undefined,
        setShipSpeed: () => undefined,
        setOptimalRoute: () => undefined,
        setSafestRoute: () => undefined,
        setLocationTypeToAdd: () => undefined,
      }),
    },
  ),
);
