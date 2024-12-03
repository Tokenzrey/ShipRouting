'use client';
import { NextPageContext } from 'next';
import NextImage from '@/components/NextImage';
import Link from 'next/link';

interface ErrorProps {
  statusCode: number | null | undefined;
}

function Error({ statusCode }: ErrorProps) {
  return (
    <main className='flex min-h-screen flex-col items-center justify-center bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 p-12'>
      <div className='text-center'>
        <NextImage
          src='/global-error.png'
          alt='Error'
          width={300}
          height={300}
          priority
        />
        <h1 className='mt-8 text-4xl font-bold text-white'>
          {statusCode === 404 ? 'Page Not Found' : 'An Error Occurred'}
        </h1>
        <p className='text-xl text-white'>
          {statusCode
            ? `An error ${statusCode} occurred on server`
            : 'An error occurred on client'}
        </p>
      </div>

      <Link
        href='/'
        className='mt-6 inline-block rounded-lg bg-white px-6 py-3 text-sm font-semibold text-indigo-500 transition-colors hover:bg-gray-100'
      >
        Go back home
      </Link>
    </main>
  );
}

Error.getInitialProps = ({ res, err }: NextPageContext): ErrorProps => {
  const statusCode = res ? res.statusCode : err ? err.statusCode : null;
  return { statusCode };
};

export default Error;
