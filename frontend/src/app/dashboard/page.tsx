'use client';
import Sidebar from '@/components/sidebar';
import withAuth from '@/lib/Auth/withAuth';
import DataVisualizationdPage from './containers/dataVisualitization';
import Map from './containers/map';

export default withAuth(DashboardPage, 'auth');
function DashboardPage() {
  return (
    <main>
      <Sidebar />
      <div className='relative min-h-screen w-full bg-oceanBlue'>
        {/* map */}
        <section className='relative'>
          <Map />
        </section>

        {/* chart data */}
        <section>
          <DataVisualizationdPage />
        </section>
      </div>
    </main>
  );
}
