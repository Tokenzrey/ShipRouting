import { ReactNode } from 'react';

import Typography from '@/components/Typography';
import { cn } from '@/lib/cn';

export default function LabelText({
  children,
  labelTextClasname,
  required,
}: {
  children: ReactNode;
  labelTextClasname?: string;
  required?: boolean;
}) {
  return (
    <label>
      <Typography
        as='p'
        variant='b3'
        weight='medium'
        className={cn('text-typo-normal-white text-base', labelTextClasname)}
      >
        {children} {required && <span className='text-danger-normal'>*</span>}
      </Typography>
    </label>
  );
}
