import { useEffect, useState } from 'react';
import { usePathname, useRouter, useSearchParams } from 'next/navigation';
import * as React from 'react';

import LoadingPage from '@/components/Loading';
import { showToast } from '@/components/Toast';
import useAuthStore from './useAuthStore';

export const USER_ROUTE = '/';
const LOGIN_ROUTE = '/login';

/**
 * Higher-order component (HOC) to handle authentication and public route access.
 * @param Component - The component to wrap.
 * @param routeType - 'auth' for protected routes or 'public' for publicly accessible routes.
 */
export default function withAuth<T>(
  Component: React.ComponentType<T>,
  routeType: 'auth' | 'public',
) {
  function ComponentWithAuth(props: T) {
    const [isMounted, setIsMounted] = useState(false); // To track client-side rendering
    const router = useRouter();
    const pathname = usePathname();
    const redirect = useSearchParams().get('redirect') || pathname;

    const isAuthenticated = useAuthStore.useIsAuthenticated();
    const isLoading = useAuthStore.useIsLoading();
    const logout = useAuthStore.useLogout();
    const stopLoading = useAuthStore.useStopLoading();
    const username = useAuthStore.useUsername();

    /**
     * Check authentication status based on store state.
     */
    const checkAuth = React.useCallback(() => {
      // Hanya mengecek apakah user telah terautentikasi atau tidak
      if (!isAuthenticated || !username) {
        logout();
      }
      stopLoading();
    }, [isAuthenticated, logout, stopLoading, username]);

    React.useEffect(() => {
      checkAuth();
      window.addEventListener('focus', checkAuth);
      return () => window.removeEventListener('focus', checkAuth);
    }, [checkAuth]);

    /**
     * Set isMounted to true after component mounts
     */
    useEffect(() => {
      setIsMounted(true); // This will only happen after the component is mounted on the client
    }, []);

    /**
     * Handle redirection based on authentication status and route type.
     */
    React.useEffect(() => {
      if (!isLoading) {
        if (routeType === 'auth' && !isAuthenticated) {
          // Redirect to login page if the route is protected and the user is not authenticated
          showToast('Please login to continue', 'Access denied', 'ERROR');
          router.push(`${LOGIN_ROUTE}?redirect=${pathname}`);
        } else if (routeType === 'public' && isAuthenticated) {
          // If the user is already authenticated, no need to login again, just stay on the page
          router.replace(redirect);
        }
      }
    }, [isAuthenticated, isLoading, redirect, router, routeType]);

    /**
     * Render the loading page or the component after mounting
     */
    if (!isMounted || isLoading) {
      return <LoadingPage />;
    }

    // Render the wrapped component
    return <Component {...(props as T)} username={username} />;
  }

  return ComponentWithAuth;
}
