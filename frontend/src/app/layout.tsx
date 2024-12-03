import './globals.css';
import 'aos/dist/aos.css';

import { cn } from '@/lib/cn';
import { Futura, poppins } from '@/lib/font';

import Providers from '@/app/providers';

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang='en'>
      <body className={cn(poppins.variable, Futura.variable)}>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
