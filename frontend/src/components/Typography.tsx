import * as React from 'react';

import { cn } from '@/lib/cn';

export enum TypographyVariant {
  'd1',
  'd2',
  'd3',
  'h1',
  'h2',
  'h3',
  'h4',
  't1',
  't2',
  't3',
  'b1',
  'b2',
  'b3',
  'b4',
  'b5',
  'l1',
  'l2',
  'l3',
}

enum FontVariant {
  'poppins',
  'futura',
}

enum FontWeight {
  'thin',
  'extralight',
  'light',
  'regular',
  'medium',
  'semibold',
  'bold',
  'extrabold',
  'black',
}

type TypographyProps<T extends React.ElementType> = {
  as?: T;
  className?: string;
  variant?: keyof typeof TypographyVariant;
  weight?: keyof typeof FontWeight;
  font?: keyof typeof FontVariant;
  children: React.ReactNode;
};

export default function Typography<T extends React.ElementType>({
  as,
  children,
  weight = 'regular',
  className,
  font = 'poppins',
  variant = 'b4',
  ...props
}: TypographyProps<T> &
  Omit<React.ComponentProps<T>, keyof TypographyProps<T>>) {
  const Component = as || 'p';
  return (
    <Component
      className={cn(
        'text-black',
        // *=============== Font Type ==================
        [
          font === 'poppins' && 'font-poppins',
          font === 'futura' && 'font-futura',
        ],
        // *=============== Font Weight ==================
        [
          weight === 'thin' && 'font-thin',
          weight === 'extralight' && 'font-extralight',
          weight === 'light' && 'font-light',
          weight === 'regular' && 'font-normal',
          weight === 'medium' && 'font-medium',
          weight === 'semibold' && 'font-semibold',
          weight === 'bold' && 'font-bold',
          weight === 'black' && 'font-black',
        ],
        // *=============== Font Variants ==================
        [
          variant === 'd1' && ['md:text-[4.6875rem] md:leading-[5rem]'],
          variant === 'd2' && ['md:text-[3.3125rem] md:leading-[3.6875rem]'],
          variant === 'd3' && ['md:text-[2.5rem] md:leading-[2.75rem]'],
          variant === 'h1' && ['md:text-[2rem] md:leading-[2.25rem]'],
          variant === 'h2' && ['md:text-[1.5rem] md:leading-[1.75rem]'],
          variant === 'h3' && ['md:text-[1.1875rem] md:leading-[1.4375rem]'],
          variant === 'h4' && ['md:text-[1rem] md:leading-[1.25rem]'],
          variant === 't1' && ['md:text-[0.9375rem] md:leading-[1.1875rem]'],
          variant === 't2' && ['md:text-[0.6875rem] md:leading-[0.9375rem]'],
          variant === 't3' && ['md:text-[0.5rem] md:leading-[0.75rem]'],
          variant === 'b1' && ['md:text-[0.8125rem] md:leading-[1.0625rem]'],
          variant === 'b2' && ['md:text-[0.75rem] md:leading-[1rem]'],
          variant === 'b3' && ['md:text-[0.6875rem] md:leading-[0.9375rem]'],
          variant === 'b4' && ['md:text-[0.5625rem] md:leading-[0.8125rem]'],
          variant === 'b5' && ['md:text-[0.5rem] md:leading-[0.75rem]'],
          variant === 'l1' && ['md:text-[0.75rem] md:leading-[1rem]'],
          variant === 'l2' && ['md:text-[0.6875rem] md:leading-[0.9375rem]'],
          variant === 'l3' && ['md:text-[0.5625rem] md:leading-[0.8125rem]'],
        ],
        'text-typo-normal-white',
        className,
      )}
      {...props}
    >
      {children}
    </Component>
  );
}
