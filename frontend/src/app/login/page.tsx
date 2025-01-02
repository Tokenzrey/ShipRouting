import * as React from 'react';

import { Ship } from 'lucide-react';
import LoginForm from '@/app/login/components/LoginForm';
import Typography from '@/components/Typography';

export default function LoginPage() {
  return (
    <main
      id='login'
      className='relative m-0 flex min-h-screen items-center justify-center gap-4 p-2 lg:flex-row lg:px-8 lg:py-2'
    >
      {/* Video Background */}
      <video
        autoPlay
        loop
        muted
        className='absolute left-0 top-0 h-full w-full object-cover'
        style={{ zIndex: -1 }}
      >
        <source src='/videos/background.mp4' type='video/mp4' />
        Your browser does not support the video tag.
      </video>

      {/* Overlay Content */}
      <div
        className='relative flex h-full w-full items-center justify-center overflow-hidden rounded-[20px] lg:w-3/4 lg:rounded-[40px]'
        id='form'
      >
        <div className='relative z-20 flex h-full w-full flex-col items-center justify-evenly gap-6 rounded-[40px] px-8 py-4 lg:justify-center lg:px-0 lg:max-w-lg'>
          <div className='flex flex-col items-center justify-center gap-2.5'>
            <Ship size={100} color='#ffffff' strokeWidth={2.25} />
            <Typography
              as='h1'
              variant='h1'
              weight='bold'
              className='text-typo-normal-black text-3xl text-typo-normal-white'
            >
              SafeVoyager
            </Typography>
            <Typography
              as='h1'
              variant='t1'
              className='text-typo-normal-black text-center text-xl text-typo-normal-white'
              weight='light'
            >
              Securing Your Journey, Preserving Your Assets
            </Typography>
          </div>
          <LoginForm />
        </div>
      </div>
    </main>
  );
}
