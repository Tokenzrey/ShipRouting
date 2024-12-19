import type { IconType } from '@icons-pack/react-simple-icons';
import { LoaderCircle, LucideIcon } from 'lucide-react';
import * as React from 'react';

import { cn } from '@/lib/cn';

const ButtonVariant = [
  'info',
  'danger',
  'warning',
  'success',
  'disabled',
  'outlined',
  'blue',
] as const;

const ButtonAppearance = ['light', 'normal', 'dark', 'outlined'] as const;

const ButtonSize = ['small', 'medium', 'large'] as const;
type ButtonProps = {
  isLoading?: boolean;
  variant?: (typeof ButtonVariant)[number];
  size?: (typeof ButtonSize)[number];
  appearance?: (typeof ButtonAppearance)[number];
  leftIcon?: LucideIcon | IconType;
  rightIcon?: LucideIcon | IconType;
  leftIconClassName?: string;
  rightIconClassName?: string;
  loadingClassName?: string;
} & React.ComponentPropsWithRef<'button'>;

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      children,
      className,
      disabled: buttonDisabled,
      isLoading,
      variant = 'info',
      size = 'medium',
      appearance = 'normal',
      leftIcon: LeftIcon,
      rightIcon: RightIcon,
      leftIconClassName,
      rightIconClassName,
      loadingClassName,
      ...rest
    },
    ref,
  ) => {
    const disabled = isLoading || buttonDisabled;

    return (
      <button
        ref={ref}
        type='button'
        disabled={disabled}
        className={cn(
          'inline-flex items-center justify-center rounded-[10px] font-medium',
          'focus:outline-none focus-visible:ring',
          'shadow-sm',
          'transition-colors duration-75',
          //#region  //*=========== Size ===========
          [
            size === 'large' && [
              'md:px-[26px] md:py-3',
              'text-base md:text-lg',
            ],
            size === 'medium' && [
              'md:px-[22px] md:py-2',
              'text-sm md:text-base',
            ],
            size === 'small' && ['md:px-[16px] md:py-[0.3rem]', 'text-[8px] md:text-xs'],
          ],
          //#endregion  //*======== Size ===========
          //#region  //*=========== Variants ===========
          [
            variant === 'info' && [
              appearance === 'light' && 'bg-info-light text-info-dark',
              appearance === 'normal' &&
                'bg-info-normal text-typo-normal-white',
              appearance === 'dark' && 'bg-info-dark text-typo-normal-white',
              appearance === 'outlined' && [
                'text-info-normal',
                'border-info-normal border-2',
                !isLoading &&
                  'hover:bg-info-normal hover:text-typo-normal-white',
                !isLoading &&
                  'active:border-info-active active:bg-info-active active:text-typo-normal-white',
                'focus-visible:ring-info-light',
              ],

              appearance !== 'outlined' && [
                'hover:bg-info-hover hover:text-typo-normal-white',
                'active:bg-info-active',
                'disabled:bg-info-active',
                'focus-visible:ring-info-light',
              ],
            ],
            variant === 'success' && [
              appearance === 'light' &&
                'bg-success-light text-typo-colorful-green',
              appearance === 'normal' &&
                'bg-success-normal text-typo-normal-white',
              appearance === 'dark' && 'bg-success-dark text-typo-normal-white',
              appearance === 'outlined' && [
                'text-success-normal',
                'border-success-normal border-2',
                !isLoading &&
                  'hover:bg-success-normal hover:text-typo-normal-white',
                !isLoading &&
                  'active:border-success-active active:bg-success-active active:text-typo-normal-white',
                'focus-visible:ring-success-light',
              ],

              appearance !== 'outlined' && [
                'hover:bg-success-hover hover:text-typo-normal-white',
                'active:bg-success-active',
                'disabled:bg-success-active',
                'focus-visible:ring-success-light',
              ],
            ],
            variant === 'danger' && [
              appearance === 'light' &&
                'bg-danger-light text-typo-colorful-red',
              appearance === 'normal' &&
                'bg-danger-normal text-typo-normal-white',
              appearance === 'dark' && 'bg-danger-dark text-typo-normal-white',
              appearance === 'outlined' && [
                'text-danger-normal',
                'border-danger-normal border-2',
                !isLoading &&
                  'hover:bg-danger-normal hover:text-typo-normal-white',
                !isLoading &&
                  'active:border-danger-active active:bg-danger-active active:text-typo-normal-white',
                'focus-visible:ring-danger-light',
              ],

              appearance !== 'outlined' && [
                'hover:bg-danger-hover hover:text-typo-normal-white',
                'active:bg-danger-active',
                'disabled:bg-danger-active',
                'focus-visible:ring-danger-light',
              ],
            ],
            variant === 'warning' && [
              appearance === 'light' && 'bg-warning-light text-warning-dark',
              appearance === 'normal' && 'bg-warning-normal text-yellow-900',
              appearance === 'dark' && 'bg-warning-dark text-typo-normal-white',
              appearance === 'outlined' && [
                'text-warning-normal',
                'border-warning-normal border-2',
                !isLoading &&
                  'hover:bg-warning-normal hover:text-typo-normal-white',
                !isLoading &&
                  'active:border-warning-active active:bg-warning-active active:text-typo-normal-white',
                'focus-visible:ring-danger-light',
              ],

              appearance !== 'outlined' && [
                'hover:bg-warning-hover hover:text-yellow-900',
                'active:bg-warning-active',
                'disabled:bg-warning-active',
                'focus-visible:ring-warning-light',
              ],
            ],
            variant === 'blue' && [
              appearance === 'light' && 'bg-blue-100 text-blue-900',
              appearance === 'normal' && 'text-typo-normal-white bg-blue-500',
              appearance === 'dark' && 'text-typo-normal-white bg-blue-900',
              appearance === 'outlined' && [
                'text-blue-500',
                'border-2 border-blue-500',
                'hover:text-typo-normal-white hover:bg-blue-700',
                'active:text-typo-normal-white active:border-blue-600 active:bg-blue-600',
                'focus-visible:ring-danger-light',
              ],

              appearance !== 'outlined' && [
                'hover:text-typo-normal-white hover:bg-blue-700',
                'active:bg-blue-600',
                'disabled:bg-blue-600',
                'focus-visible:ring-warning-light',
              ],
            ],
            variant === 'outlined' && [
              'border-grey-purple-900 text-grey-purple-900 border-2',
              !isLoading &&
                'hover:bg-grey-purple-900 hover:text-typo-normal-white',
              !isLoading && 'active:bg-grey-purple-800',
              'disabled:bg-grey-purple-900',
              'focus-visible:ring-purple-100',
            ],
            variant === 'disabled' && [
              'bg-grey-purple-700 text-grey-purple-900',
              'disabled:bg-grey-purple-700',
              'focus-visible:ring-purple-100',
            ],
          ],
          //#endregion  //*======== Variants ===========
          'disabled:cursor-not-allowed',
          isLoading &&
            'relative text-transparent transition-none hover:text-transparent disabled:cursor-wait',
          className,
        )}
        {...rest}
      >
        {isLoading && (
          <div
            className={cn(
              'absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2',
              (variant === 'info' ||
                variant === 'success' ||
                variant === 'danger' ||
                variant === 'warning' ||
                variant === 'blue') && [
                appearance !== 'outlined' && 'text-typo-normal-white',
                appearance === 'outlined' && `text-${variant}-normal`,
              ],
              variant === 'outlined' && 'text-typo-normal-white',
              loadingClassName,
            )}
          >
            <LoaderCircle size={18} className='animate-spin' />
          </div>
        )}
        {LeftIcon && (
          <div
            className={cn([
              size === 'large' && 'mr-3',
              size === 'medium' && 'mr-2',
              size === 'small' && 'mr-1',
            ])}
          >
            <LeftIcon
              size='1em'
              className={cn('text-2xl', leftIconClassName)}
            />
          </div>
        )}
        {children}
        {RightIcon && (
          <div
            className={cn([
              size === 'large' && 'ml-3',
              size === 'medium' && 'ml-2',
              size === 'small' && 'ml-1',
            ])}
          >
            <RightIcon
              size='1em'
              className={cn('text-2xl', rightIconClassName)}
            />
          </div>
        )}
      </button>
    );
  },
);

export default Button;
