import type { IconType } from '@icons-pack/react-simple-icons';
import { LucideIcon } from 'lucide-react';
import * as React from 'react';

import UnstyledLink from '@/components/links/UnstyledLink';
import { cn } from '@/lib/cn';

const ButtonLinkVariant = [
  'info',
  'danger',
  'warning',
  'success',
  'disabled',
  'outlined',
  'blue',
] as const;

const ButtonLinkAppearance = ['light', 'normal', 'dark', 'outlined'] as const;

const ButtonLinkSize = ['small', 'medium', 'large'] as const;

type ButtonLinkProps = {
  variant?: (typeof ButtonLinkVariant)[number];
  size?: (typeof ButtonLinkSize)[number];
  appearance?: (typeof ButtonLinkAppearance)[number];
  href: string;
  openNewTab?: boolean;
  leftIcon?: LucideIcon | IconType;
  rightIcon?: LucideIcon | IconType;
  leftIconClassName?: string;
  rightIconClassName?: string;
} & React.ComponentPropsWithRef<'a'>;

const ButtonLink = React.forwardRef<HTMLAnchorElement, ButtonLinkProps>(
  (
    {
      children,
      className,
      href,
      openNewTab,
      variant = 'info',
      size = 'medium',
      appearance = 'normal',
      leftIcon: LeftIcon,
      rightIcon: RightIcon,
      leftIconClassName,
      rightIconClassName,
      ...rest
    },
    ref,
  ) => {
    return (
      <UnstyledLink
        ref={ref}
        type='button'
        href={href}
        openNewTab={openNewTab}
        className={cn(
          'inline-flex items-center justify-center rounded-[10px] font-medium',
          'focus:outline-none focus-visible:ring',
          'shadow-sm',
          'transition-colors duration-75',
          //#region  //*=========== Size ===========
          [
            size === 'large' && ['px-[26px] py-3', 'text-base md:text-lg'],
            size === 'medium' && ['px-[22px] py-2', 'text-sm md:text-base'],
            size === 'small' && ['px-[18px] py-1', 'text-xs md:text-sm'],
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
                'hover:bg-info-normal hover:text-typo-normal-white',
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
                'hover:bg-success-normal hover:text-typo-normal-white',
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
                'hover:bg-danger-normal hover:text-typo-normal-white',
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
                'hover:bg-warning-normal hover:text-typo-normal-white',
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
              'hover:bg-grey-purple-900 hover:text-typo-normal-white',
              'active:bg-grey-purple-800',
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
          className,
        )}
        {...rest}
      >
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
      </UnstyledLink>
    );
  },
);

export default ButtonLink;
