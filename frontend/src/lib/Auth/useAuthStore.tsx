import { createSelectorHooks } from 'auto-zustand-selectors-hook';
import { produce } from 'immer';
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

/**
 * AuthStoreType defines the structure and functionality of the authentication store.
 * @typedef {Object} AuthStoreType
 * @property {string | null} username - The username of the currently authenticated user, or null if not authenticated.
 * @property {boolean} isAuthenticated - Indicates if the user is authenticated.
 * @property {boolean} isLoading - Indicates if authentication state is currently loading.
 * @property {(username: string) => void} login - Function to log in the user.
 * @property {() => void} logout - Function to log out the user.
 * @property {() => void} stopLoading - Function to stop loading state.
 */
type AuthStoreType = {
  username: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (username: string) => void;
  logout: () => void;
  stopLoading: () => void;
};

// Base auth store with Zustand, using immer for immutability and zustand-persist for persistence.
const useAuthStoreBase = create<AuthStoreType>()(
  persist(
    (set) => ({
      username: null, // Initial username state is null
      isAuthenticated: false, // User is not authenticated by default
      isLoading: true, // Set loading state to true initially

      /**
       * Login updates the state with username and sets isAuthenticated to true.
       * @param {string} username - The username of the user.
       */
      login: (username: string) => {
        set(
          produce((state: AuthStoreType) => {
            state.username = username; // Set username
            state.isAuthenticated = true; // Set authenticated status to true
            state.isLoading = false; // Stop loading after login
          })
        );
      },

      /**
       * Logout clears the username and sets isAuthenticated to false.
       */
      logout: () => {
        set(
          produce((state: AuthStoreType) => {
            state.username = null; // Clear username
            state.isAuthenticated = false; // Set authenticated status to false
          })
        );
      },

      /**
       * stopLoading sets the loading state to false.
       */
      stopLoading: () => {
        set(
          produce((state: AuthStoreType) => {
            state.isLoading = false; // Stop loading
          })
        );
      },
    }),
    { name: '@shipRouting/auth', getStorage: () => localStorage } // Persisting in localStorage with custom key
  )
);

// Use createSelectorHooks for easy selector usage within components.
const useAuthStore = createSelectorHooks(useAuthStoreBase);

export default useAuthStore;
