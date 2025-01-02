import * as React from 'react';

import Typography from '@/components/Typography';
import { cn } from '@/lib/cn';

export default function ErrorMessage({
  children,
  className,
  errorTextClassName,
}: {
  children: string | React.ReactNode;
  className?: string;
  errorTextClassName?: string;
}) {
  return (
    <div className={cn('flex space-x-1', className)}>
      <Typography
        as='p'
        weight='medium'
        variant='b4'
        className={cn(
          'text-danger-normal text-sm !leading-tight',
          errorTextClassName,
        )}
      >
        {children}
      </Typography>
    </div>
  );
}
