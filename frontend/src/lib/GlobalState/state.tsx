import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface Location {
  type: 'from' | 'destination';
  name: string;
  longitude: number; // Longitude koordinat
  latitude: number; // Latitude koordinat
}

export interface PathPoint {
  Heave: number;
  Pitch: number;
  Roll: number;
  coordinates: [number, number];
  dirpwsfc: number;
  htsgwsfc: number;
  node_id: string;
  perpwsfc: number;
  rel_heading: number;
}

export interface BlockedEdge {
  source_coords: [number, number]; // Koordinat awal edge
  target_coords: [number, number]; // Koordinat akhir edge
  isBlocked: boolean; // Apakah edge diblokir
}

export interface FinalPath {
  path: PathPoint[];
  distance: number;
}

export interface Keyframes {
  partial_path: PathPoint[];
  final_path: FinalPath;
  all_edges: BlockedEdge[];
}

export interface Expansion {
  coordinates: [number, number]; // [lon, lat]
}

interface RouteStore {
  locations: Location[]; // List of locations (from and destination)
  optimalDistance: number | null; // Distance for the optimal route in kilometers
  safestDistance: number | null; // Distance for the safest route in kilometers
  optimalDuration: number | null; // Duration for the optimal route in hours
  safestDuration: number | null; // Duration for the safest route in hours

  // New State for Keyframes
  optimalKeyframes: Keyframes | null;
  safestKeyframes: Keyframes | null;

  finalPath: FinalPath;
  expansions: any[];
  partialPath: PathPoint[];
  blockedEdges: BlockedEdge[];
  isCalculating: Boolean;
  routeSelected: string;
  loadCondition: string; // Load condition (e.g., "Light", "Medium", "Heavy")
  animationState: string | null;
  currentAnimationIndex: number | null;
  shipSpeed: number; // Ship speed in knots
  optimalRoute: PathPoint[];
  safestRoute: PathPoint[];
  locationTypeToAdd: 'from' | 'destination' | null;
  activeRoute: 'safest' | 'optimal' | null;

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
  setOptimalRoute: (route: PathPoint[]) => void;
  setSafestRoute: (route: PathPoint[]) => void;
  setLocationTypeToAdd: (type: 'from' | 'destination' | null) => void;
  setAnimationState: (animationState: string) => void;
  setCurrentAnimationIndex: (currentAnimationIndex: number) => void;
  setActiveRoute: (activeRoute: 'safest' | 'optimal') => void;
  setFinalPath: (path: PathPoint[], distance: number) => void;
  setPartialPath: (path: PathPoint[]) => void;
  addExpansions: (expansions: Expansion[]) => void;
  setExpansions: (expansions: Expansion[]) => void;
  setBlockedEdges: (edges: BlockedEdge[]) => void;
  addBlockedEdge: (edge: BlockedEdge) => void;
  setIsCalculating: (isCaluclating: Boolean) => void;
  setRouteSelected: (routeSelected: string) => void;

  // New Setters for Keyframes
  setOptimalKeyframes: (keyframes: Keyframes) => void;
  setSafestKeyframes: (keyframes: Keyframes) => void;
  resetKeyframes: () => void;

  clearRoutes: () => void;
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
      optimalRoute: [],
      safestRoute: [],
      partialPath: [],
      finalPath: {
        path: [],
        distance: 0.0,
      },
      expansions: [],
      blockedEdges: [],
      isCalculating: false,
      routeSelected: 'None',
      locationTypeToAdd: null,
      animationState: 'idle',
      currentAnimationIndex: 0,
      activeRoute: 'optimal',
      optimalKeyframes: null,
      safestKeyframes: null,

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
      setFinalPath: (path, distance) => set({ finalPath: { path, distance } }),
      setCurrentAnimationIndex: (index) =>
        set({ currentAnimationIndex: index }),
      setAnimationState: (state) => set({ animationState: state }),
      setActiveRoute: (route: 'optimal' | 'safest') =>
        set({ activeRoute: route }),
      setPartialPath: (path) => set({ partialPath: path }),
      setExpansions: (expansions) => set({ expansions }),
      addExpansions: (expansion) =>
        set((state) => ({ expansions: [...state.expansions, ...expansion] })),
      setBlockedEdges: (edges) => set({ blockedEdges: edges }),
      addBlockedEdge: (edge) =>
        set((state) => ({ blockedEdges: [...state.blockedEdges, edge] })),
      setRouteSelected: (routeSelected) =>
        set({ routeSelected: routeSelected }),
      setIsCalculating: (isClaculating) =>
        set({ isCalculating: isClaculating }),
      setOptimalKeyframes: (keyframes) => set({ optimalKeyframes: keyframes }),
      setSafestKeyframes: (keyframes) => set({ safestKeyframes: keyframes }),
      resetKeyframes: () =>
        set({ optimalKeyframes: null, safestKeyframes: null }),
      clearRoutes: () =>
        set({
          optimalRoute: [],
          safestRoute: [],
          partialPath: [],
          expansions: [],
        }),
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
        animationState: state.animationState,
        currentAnimationIndex: state.currentAnimationIndex,
        activeRoute: state.activeRoute,
        partialPath: state.partialPath,
        expansions: state.expansions,
        finalPath: state.finalPath,
        blockedEdges: state.blockedEdges,
        routeSelected: state.routeSelected,
        isCalculating: state.isCalculating,
        optimalKeyframes: state.optimalKeyframes,
        safestKeyframes: state.safestKeyframes,
        // Add placeholders for other fields to satisfy RouteStore type
        optimalRoute: state.optimalRoute,
        safestRoute: state.safestRoute,
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
        setCurrentAnimationIndex: () => undefined,
        setAnimationState: () => undefined,
        setActiveRoute: () => undefined,
        setPartialPath: () => undefined,
        setFinalPath: () => undefined,
        addExpansions: () => undefined,
        setExpansions: () => undefined,
        setBlockedEdges: () => undefined,
        addBlockedEdge: () => undefined,
        setRouteSelected: () => undefined,
        setIsCalculating: () => undefined,
        setOptimalKeyframes: () => undefined,
        setSafestKeyframes: () => undefined,
        resetKeyframes: () => undefined,
        clearRoutes: () => undefined,
      }),
    },
  ),
);
