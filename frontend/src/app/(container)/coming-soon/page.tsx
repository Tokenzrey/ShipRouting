import NextImage from '@/components/NextImage';

export default function ComingSoonPage() {
  return (
    <main className='flex min-h-screen flex-col items-center justify-center bg-gradient-to-b from-cyan-500 to-blue-700 p-24'>
      <div className='text-center'>
        <NextImage
          src='/coming-soon.png'
          alt='Coming Soon'
          width={250}
          height={250}
          className='relative left-1/2 -translate-x-1/2'
        />
        <h1 className='mt-8 text-4xl font-bold text-white'>Coming Soon!</h1>
        <p className='text-xl text-white'>
          We're working hard to finish the development of this site. Stay tuned!
        </p>
      </div>

      <div className='mt-10'>
        <a
          href='/'
          className='inline-block rounded-lg bg-white px-6 py-3 text-sm font-semibold text-blue-700 transition-colors hover:bg-blue-50'
          target='_blank'
          rel='noopener noreferrer'
        >
          Visit Our Homepage
        </a>
      </div>
    </main>
  );
}
