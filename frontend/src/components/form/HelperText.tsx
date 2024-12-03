import * as React from 'react';

import Typography from '@/components/Typography';
import { cn } from '@/lib/cn';

export default function HelperText({
  children,
  className,
  helperTextClassName,
}: {
  children: React.ReactNode;
  className?: string;
  helperTextClassName?: string;
}) {
  return (
    <div className={cn('flex space-x-1', className)}>
      <Typography
        as='p'
        weight='medium'
        variant='b3'
        className={cn(
          'text-grey-purple-800 text-base !leading-tight',
          helperTextClassName,
        )}
      >
        {children}
      </Typography>
    </div>
  );
}
