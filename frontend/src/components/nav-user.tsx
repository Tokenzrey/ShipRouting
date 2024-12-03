import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import {
  SidebarMenu,
  SidebarMenuItem,
} from '@/components/ui/sidebar';

export function NavUser({
  user,
}: {
  user: {
    name: string;
  };
}) {

  return (
    <SidebarMenu>
      <SidebarMenuItem className='flex items-center justify-between px-2 py-3'>
        <div className='flex items-center gap-2'>
          <Avatar className='h-8 w-8 rounded-lg'>
            <AvatarImage src='/images/avatar.jpeg' alt={user.name} />
            <AvatarFallback className='rounded-lg'>CN</AvatarFallback>
          </Avatar>
          <div className='grid flex-1 text-left text-sm leading-tight'>
            <span className='truncate font-semibold text-typo-normal-white'>
              {user.name}
            </span>
            <span className='truncate text-xs text-typo-normal-secondary'>
              Captain
            </span>
          </div>
        </div>
      </SidebarMenuItem>
    </SidebarMenu>
  );
}
