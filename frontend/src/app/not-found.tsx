import NextImage from '@/components/NextImage';
import Link from 'next/link';

export default function NotFoundPage() {
  return (
    <main className='flex min-h-screen flex-col items-center justify-center bg-gradient-to-b from-gray-200 via-gray-300 to-gray-400 p-12'>
      <div className='flex flex-col items-center'>
        <NextImage
          src='/not-found.png'
          alt='Not Found'
          width={300}
          height={300}
        />
        <h1 className='mt-8 text-center text-4xl font-bold text-gray-800'>
          Oops!
        </h1>
        <p className='text-center text-lg text-gray-600 '>
          We can't seem to find the page you're looking for.
        </p>
        <p className='mt-2 text-center text-gray-500 '>Error code: 404</p>
      </div>

      <Link
        href='/'
        className='mt-6 inline-block rounded-lg bg-blue-500 px-6 py-3 text-sm font-semibold text-white transition-colors hover:bg-blue-600'
      >
        Go back home
      </Link>
    </main>
  );
}
