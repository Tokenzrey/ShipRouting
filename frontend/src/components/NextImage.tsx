'use client';

import Image, { ImageProps } from 'next/image';
import * as React from 'react';

import { cn } from '@/lib/cn';

type NextImageProps = {
  useSkeleton?: boolean;
  imgClassName?: string;
  serverStaticImg?: boolean;
  blurClassName?: string;
  alt: string;
  width: string | number;
  onErrorSrc?: string;
  priority?: boolean;
} & (
  | { width: string | number; height: string | number }
  | { layout: 'fill'; width?: string | number; height?: string | number }
) &
  ImageProps;

export default function NextImage({
  useSkeleton = false,
  serverStaticImg = false,
  src,
  width,
  height,
  alt,
  className,
  imgClassName,
  blurClassName,
  onErrorSrc,
  priority = false,
  ...rest
}: NextImageProps) {
  const [status, setStatus] = React.useState(
    useSkeleton ? 'loading' : 'complete',
  );
  const widthIsSet = className?.includes('w-') ?? false;

  return (
    <figure
      style={!widthIsSet ? { width: `${width}px` } : undefined}
      className={className}
    >
      <Image
        className={cn(
          imgClassName,
          status === 'loading' && cn('animate-pulse bg-red-50', blurClassName),
        )}
        src={serverStaticImg ? src : '/images' + src}
        width={width}
        height={height}
        alt={alt || 'Image'}
        priority={priority}
        onLoad={() => setStatus('complete')}
        onError={(e) => {
          if (onErrorSrc) {
            e.currentTarget.src = onErrorSrc;
          }
        }}
        {...rest}
      />
    </figure>
  );
}
