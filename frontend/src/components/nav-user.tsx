import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { SidebarMenu, SidebarMenuItem } from '@/components/ui/sidebar';
import Typography from './Typography';

export function NavUser({
  user,
}: {
  user: {
    name: string;
  };
}) {
  return (
    <SidebarMenu>
      <SidebarMenuItem className='flex items-center justify-between px-[0.125rem] py-[0.25rem]'>
        <div className='flex items-center gap-2'>
          <Avatar className='h-8 w-8 rounded-md'>
            <AvatarImage src='/images/avatar.jpeg' alt={user.name} />
            <AvatarFallback className='rounded-md'>CN</AvatarFallback>
          </Avatar>
          <div className='grid flex-1 text-left text-sm leading-tight'>
            <Typography
              className='truncate font-semibold text-typo-normal-white'
              variant='b2'
            >
              {user.name}
            </Typography>
            <Typography
              className='truncate text-xs text-typo-normal-secondary'
              variant='b3'
            >
              Captain
            </Typography>
          </div>
        </div>
      </SidebarMenuItem>
    </SidebarMenu>
  );
}
