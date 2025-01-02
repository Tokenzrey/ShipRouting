import { Skeleton } from "./ui/skeleton";
export default function LoadingPage() {
  return (
    <main className='flex min-h-screen w-full flex-col items-center justify-between p-24'>
      <Skeleton className="h-full w-full" />
    </main>
  );
}
