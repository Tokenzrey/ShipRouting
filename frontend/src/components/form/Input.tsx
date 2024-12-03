import type { IconType } from '@icons-pack/react-simple-icons';
import { Eye, EyeOff, LucideIcon } from 'lucide-react';
import { useState } from 'react';
import { get, RegisterOptions, useFormContext } from 'react-hook-form';

import ErrorMessage from '@/components/form/ErrorMessage';
import HelperText from '@/components/form/HelperText';
import LabelText from '@/components/form/LabelText';
import Typography from '@/components/Typography';
import { cn } from '@/lib/cn';

export type InputProps = {
  id: string;
  label?: string;
  helperText?: React.ReactNode;
  helperTextClassName?: string;
  hideError?: boolean;
  validation?: RegisterOptions;
  prefix?: string;
  suffix?: string;
  prefixClassName?: string;
  suffixClassName?: string;
  rightIcon?: LucideIcon | IconType;
  leftIcon?: LucideIcon | IconType;
  rightIconClassName?: string;
  leftIconClassName?: string;
} & React.ComponentPropsWithoutRef<'input'>;

export default function Input({
  id,
  label,
  helperText,
  hideError = false,
  validation,
  prefix,
  suffix,
  prefixClassName,
  suffixClassName,
  className,
  type = 'text',
  readOnly = false,
  rightIcon: RightIcon,
  leftIcon: LeftIcon,
  rightIconClassName,
  leftIconClassName,
  helperTextClassName,
  ...rest
}: InputProps) {
  const {
    register,
    formState: { errors },
  } = useFormContext();

  const [showPassword, setShowPassword] = useState(false);
  const error = get(errors, id);

  return (
    <div className='w-full space-y-2'>
      {label && (
        <LabelText required={validation?.required ? true : false}>
          {label}
        </LabelText>
      )}

      <div className='relative flex w-full gap-0'>
        <div
          className={cn(
            'pointer-events-none absolute h-full w-full rounded-md ring-inset',
          )}
        />

        {prefix && (
          <Typography
            variant='l3'
            className={cn(
              'bg-grey-purple-100 text-grey-purple-800 flex items-center rounded-l-md border-r px-3 text-sm font-semibold',
              prefixClassName,
            )}
          >
            {prefix}
          </Typography>
        )}

        <div
          className={cn(
            'relative w-full rounded-xl',
            prefix && 'rounded-l-md',
            suffix && 'rounded-r-md',
          )}
        >
          {LeftIcon && (
            <div
              className={cn(
                'absolute left-0 top-0 h-full',
                'flex items-center justify-center pl-2.5',
                'text-grey-purple-700 text-lg md:text-xl',
                leftIconClassName,
              )}
            >
              <LeftIcon />
            </div>
          )}

          <input
            {...register(id, validation)}
            type={
              type === 'password' ? (showPassword ? 'text' : 'password') : type
            }
            id={id}
            name={id}
            readOnly={readOnly}
            disabled={readOnly}
            className={cn(
              'caret-info-active h-full w-full rounded-md px-3 py-2.5',
              LeftIcon && 'pl-9',
              RightIcon && 'pr-9',
              'focus:outline-info-active focus:outline-1 focus:ring-inset',
              'bg-grey-purple-100 text-sm',
              'hover:ring-info-active hover:ring-1 hover:ring-inset',
              'placeholder:font-helvetica placeholder:text-grey-purple-800 placeholder:text-sm placeholder:font-semibold',
              'text-typo-normal-main',
              readOnly && 'cursor-not-allowed',
              error &&
                'bg-danger-light ring-danger-normal placeholder:text-grey-purple-800 border-none ring-2 ring-inset',
              prefix && 'rounded-l-none rounded-r-md',
              suffix && 'rounded-l-md rounded-r-none',
              prefix && suffix && 'rounded-none',
              className,
            )}
            aria-describedby={id}
            {...rest}
          />

          {RightIcon && type !== 'password' && (
            <div
              className={cn(
                'absolute bottom-0 right-0 h-full',
                'flex items-center justify-center pr-2.5',
                'text-grey-purple-700 text-lg md:text-xl',
                rightIconClassName,
              )}
            >
              <RightIcon />
            </div>
          )}

          {type === 'password' && (
            <div
              className={cn(
                'absolute bottom-0 right-0 h-full',
                'flex items-center justify-center pr-3',
                'text-grey-purple-800 text-lg md:text-xl',
                rightIconClassName,
              )}
              onClick={() => setShowPassword(!showPassword)}
            >
              {showPassword ? <EyeOff /> : <Eye />}
            </div>
          )}
        </div>
        {suffix && (
          <Typography
            variant='l3'
            className={cn(
              'bg-grey-purple-100 text-grey-purple-800 flex w-min items-center rounded-r-md border-l px-3 text-sm font-semibold',
              suffixClassName,
            )}
          >
            {suffix}
          </Typography>
        )}
      </div>

      {!hideError && error && <ErrorMessage>{error.message}</ErrorMessage>}
      {helperText && (
        <HelperText helperTextClassName={helperTextClassName}>
          {helperText}
        </HelperText>
      )}
    </div>
  );
}
