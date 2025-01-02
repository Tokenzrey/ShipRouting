import { toast } from '@/components/ui/use-toast';

export function showToast(
  title: string,
  description: string,
  variant: 'INFO' | 'SUCCESS' | 'ERROR',
) {
  toast({
    title,
    description,
    variant,
  });
}
