'use client';

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import React from 'react';

import LoadingPage from '@/components/Loading';
import { Toaster } from '@/components/ui/toaster';

const queryClient = new QueryClient();

export default function Providers({ children }: { children: React.ReactNode }) {
  return (
    <QueryClientProvider client={queryClient}>
      <Toaster />
      <React.Suspense fallback={<LoadingPage />}>{children}</React.Suspense>
    </QueryClientProvider>
  );
}
