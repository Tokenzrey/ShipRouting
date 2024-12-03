import * as React from 'react';

import UnstyledLink from '@/components/links/UnstyledLink';
import { cn } from '@/lib/cn';

const PrimaryLinkVariant = [
  'info',
  'danger',
  'warning',
  'success',
  'disabled',
] as const;

const PrimaryLinkAppearance = ['light', 'normal', 'dark'] as const;

const PrimaryLinkSize = ['small', 'medium', 'large'] as const;

type primaryLinkProps = {
  href: string;
  variant?: (typeof PrimaryLinkVariant)[number];
  size?: (typeof PrimaryLinkSize)[number];
  appearance?: (typeof PrimaryLinkAppearance)[number];
  underline?: boolean;
  children: React.ReactNode;
  openNewTab?: boolean;
  className?: string;
};

const PrimaryLink = React.forwardRef<HTMLAnchorElement, primaryLinkProps>(
  (
    {
      className,
      children,
      href,
      variant = 'info',
      size = 'medium',
      appearance = 'normal',
      openNewTab = false,
      underline = true,
      ...rest
    },
    ref,
  ) => {
    return (
      <UnstyledLink
        ref={ref}
        href={href}
        openNewTab={openNewTab}
        {...rest}
        className={cn(
          //#region  //*=========== Size ===========
          [
            size === 'large' && 'text-lg',
            size === 'medium' && 'text-base',
            size === 'small' && 'text-sm',
          ],
          //#endregion  //*=========== Size ===========

          //#region  //*=========== Variants ===========
          [
            variant === 'info' && [
              appearance === 'light' && 'text-info-light',
              appearance === 'normal' && 'text-info-normal',
              appearance === 'dark' && 'text-info-dark',
              'hover:text-info-hover',
              'active:text-info-active',
              'disabled:text-info-active',
            ],
            variant === 'success' && [
              appearance === 'light' && 'text-success-light',
              appearance === 'normal' && 'text-success-normal',
              appearance === 'dark' && 'text-success-dark',
              'hover:text-success-hover',
              'active:text-success-active',
              'disabled:text-success-active',
            ],
            variant === 'danger' && [
              appearance === 'light' && 'text-danger-light',
              appearance === 'normal' && 'text-danger-normal',
              appearance === 'dark' && 'text-danger-dark',
              'hover:text-danger-hover',
              'active:text-danger-active',
              'disabled:text-danger-active',
            ],
            variant === 'warning' && [
              appearance === 'light' && 'text-warning-light',
              appearance === 'normal' && 'text-warning-normal',
              appearance === 'dark' && 'text-warning-dark',
              'hover:text-warning-hover',
              'active:text-warning-active',
              'disabled:text-warning-active',
            ],
            variant === 'disabled' && [
              'text-grey-purple-900',
              'disabled:text-grey-purple-700',
            ],
          ],
          //#endregion  //*======== Variants ===========
          underline && 'underline underline-offset-4',
          className,
        )}
      >
        {children}
      </UnstyledLink>
    );
  },
);

export default PrimaryLink;
