import { AppSidebar } from '@/components/app-sidebar';
import { SidebarProvider  } from '@/components/ui/sidebar';

export default function Sidebar() {
  return (
    <SidebarProvider>
      <AppSidebar variant='floating' />
    </SidebarProvider>
  );
}
