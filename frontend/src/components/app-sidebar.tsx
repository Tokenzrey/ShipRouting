'use client';

import * as React from 'react';
import { useRouter } from 'next/navigation';
import { LogOut, CalendarIcon } from 'lucide-react';

import { NavMain } from '@/components/nav-main';
import {
  Sidebar,
  SidebarContent,
  SidebarHeader,
  SidebarFooter,
  SidebarTrigger,
} from '@/components/ui/sidebar';
import { showToast } from '@/components/Toast';
import useAuthStore from '@/lib/Auth/useAuthStore';
import { NavUser } from './nav-user';
import { Separator } from './ui/separator';
import IconButton from './buttons/IconButton';
import Typography from './Typography';
import { getFormattedDate } from '@/lib/getDate';

// This is sample data.
const data = {
  user: {
    name: 'Steven Caramoy',
  },
};

export function AppSidebar({ ...props }: React.ComponentProps<typeof Sidebar>) {
  const router = useRouter();
  const { logout } = useAuthStore(); // Make sure to destructure correctly if it's exported as part of an object
  const [currentDate, setCurrentDate] = React.useState(getFormattedDate());

  React.useEffect(() => {
    const updateDate = () => {
      const newDate = getFormattedDate();
      setCurrentDate(newDate);
    };

    // Interval untuk memeriksa perubahan hari
    const interval = setInterval(() => {
      updateDate();
    }, 60000); // Perbarui setiap menit

    return () => clearInterval(interval); // Bersihkan interval saat komponen unmount
  }, []);

  const handleLogout = () => {
    logout();
    showToast('Logout success!', 'You have successfully logged out', 'SUCCESS');
    router.replace('/'); // Navigate to the root or login page after logout
  };

  return (
    <Sidebar {...props}>
      <SidebarHeader>
        <NavUser user={data.user} />
        <SidebarTrigger className='absolute -right-8 top-2 bg-oceanBlue text-typo-normal-white hover:text-oceanBlue' />
      </SidebarHeader>
      <Separator />
      <SidebarContent className='SidebarContent'>
        <NavMain />
      </SidebarContent>
      <Separator />
      <SidebarFooter className='flex flex-row justify-between px-4 py-3'>
        <div className='flex items-center gap-1 rounded-sm border-2 border-typo-normal-white px-2 py-1'>
          <CalendarIcon
            className='text-typo-normal-white'
            size={14}
            strokeWidth={2.5}
          />
          <Typography
            className='text-typo-normal-white'
            weight='medium'
            variant='b5'
          >
            {currentDate}
          </Typography>
        </div>
        <IconButton
          size='small'
          variant='danger'
          Icon={LogOut}
          appearance='light'
          IconClassName='w-5 h-5 p-0.5'
          onClick={handleLogout}
        />
      </SidebarFooter>
    </Sidebar>
  );
}
