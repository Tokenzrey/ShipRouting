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
          variant === 'd1' && ['md:text-[112px] md:leading-[120px]'],
          variant === 'd2' && ['md:text-[80px] md:leading-[88px]'],
          variant === 'd3' && ['md:text-[60px] md:leading-[66px]'],
          variant === 'h1' && ['md:text-[48px] md:leading-[54px]'],
          variant === 'h2' && ['md:text-[36px] md:leading-[42px]'],
          variant === 'h3' && ['md:text-[28px] md:leading-[34px]'],
          variant === 'h4' && ['md:text-[24px] md:leading-[30px]'],
          variant === 't1' && ['md:text-[22px] md:leading-[28px]'],
          variant === 't2' && ['md:text-[16px] md:leading-[22px]'],
          variant === 't3' && ['md:text-[12px] md:leading-[18px]'],
          variant === 'b1' && ['md:text-[20px] md:leading-[26px]'],
          variant === 'b2' && ['md:text-[18px] md:leading-[24px]'],
          variant === 'b3' && ['md:text-[16px] md:leading-[22px]'],
          variant === 'b4' && ['md:text-[14px] md:leading-[20px]'],
          variant === 'b5' && ['md:text-[12px] md:leading-[18px]'],
          variant === 'l1' && ['md:text-[18px] md:leading-[24px]'],
          variant === 'l2' && ['md:text-[16px] md:leading-[22px]'],
          variant === 'l3' && ['md:text-[14px] md:leading-[20px]'],
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
