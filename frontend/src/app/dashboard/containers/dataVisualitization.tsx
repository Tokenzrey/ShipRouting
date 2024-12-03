import React, { useState, useEffect, useRef } from 'react';
import Typography from '@/components/Typography';
import LineChart from '../components/lineChart';
import {
  Route,
  ChevronDown,
  BetweenHorizontalStart,
  AlignJustify,
  RotateCw,
  Play,
  Pause,
} from 'lucide-react';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@radix-ui/react-select';
import EnviromentalSeeker from '../components/enviromentalSeeker';
import { cn } from '@/lib/cn';
import { datasets } from '@/contents/dataVisual';

type RouteType = 'normal_route' | 'safest_route';
type DataType = 'enviromental' | 'ship_motion';

export default function DataVisualizationPage() {
  // State untuk mengelola data yang dipilih
  const [selectedRoute, setSelectedRoute] = useState<RouteType>('normal_route');
  const [selectedData, setSelectedData] = useState<DataType>('enviromental');
  const [currentIndex, setCurrentIndex] = useState(0); // Indeks data saat ini
  const [isPlaying, setIsPlaying] = useState(false); // Status animasi
  const [isExpanded, setIsExpanded] = useState(false); // Status ekspansi panel

  const intervalRef = useRef<NodeJS.Timeout | null>(null); // Referensi interval animasi

  // Dataset saat ini berdasarkan rute dan tipe data
  const currentDataset = datasets[selectedRoute][selectedData];
  const currentDatasetEnviromental = datasets[selectedRoute];

  // Data untuk EnviromentalSeeker
  const seekerData = {
    waveHeight: currentDatasetEnviromental.enviromental[0].data[currentIndex],
    wavePeriod: currentDatasetEnviromental.enviromental[1].data[currentIndex],
    waveHeading: currentDatasetEnviromental.enviromental[2].data[currentIndex],
    roll: currentDatasetEnviromental.ship_motion[0].data[currentIndex],
    heave: currentDatasetEnviromental.ship_motion[1].data[currentIndex],
    pitch: currentDatasetEnviromental.ship_motion[2].data[currentIndex],
  };

  // Nilai maksimum untuk normalisasi grafik
  const maxValues = {
    waveHeight: 5,
    wavePeriod: 15,
    waveHeading: 360,
  };

  // Fungsi untuk memulai animasi
  const playAnimation = () => {
    setIsPlaying(false);
    setCurrentIndex(0);
    setIsPlaying(true);
    if (intervalRef.current) clearInterval(intervalRef.current);

    intervalRef.current = setInterval(() => {
      setCurrentIndex((prevIndex) => {
        if (prevIndex < currentDataset[0].data.length - 1) {
          return prevIndex + 1;
        } else {
          clearInterval(intervalRef.current!);
          setIsPlaying(false);
          return prevIndex;
        }
      });
    }, 500); // Interval 0.5 detik
  };

  // Fungsi untuk menjeda animasi
  const pauseAnimation = () => {
    setIsPlaying(false);
    if (intervalRef.current) clearInterval(intervalRef.current);
  };

  // Fungsi untuk mereset animasi
  const resetAnimation = () => {
    setIsPlaying(false);
    if (intervalRef.current) clearInterval(intervalRef.current);
    setCurrentIndex(0);
  };

  // Fungsi untuk mengubah posisi panel
  const togglePosition = () => setIsExpanded((prev) => !prev);

  // Cleanup saat komponen tidak lagi digunakan
  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  return (
    <div
      className={cn(
        'fixed bottom-0 flex w-full flex-col bg-aquaBlue transition-[max-height] duration-500 ease-in',
        !isExpanded ? 'max-h-0' : 'max-h-[500px]',
      )}
    >
      {/* Tombol kontrol */}
      <div className='absolute -top-8 right-1/2 flex translate-x-1/2 gap-2 rounded-md bg-typo-normal-white px-2'>
        <button onClick={resetAnimation}>
          <RotateCw color='#1f2937' size={20} />
        </button>
        <button className='bg-gray-800 px-3 py-2' onClick={togglePosition}>
          <AlignJustify color='#ffffff' size={20} />
        </button>
        {isPlaying ? (
          <button onClick={pauseAnimation}>
            <Pause color='#1f2937' size={20} />
          </button>
        ) : (
          <button onClick={playAnimation}>
            <Play color='#1f2937' size={20} />
          </button>
        )}
      </div>

      {/* Panel utama */}
      <section className='relative'>
        <div className='absolute -top-24 right-5 flex w-28 flex-col gap-2 rounded-lg bg-aquaBlue p-2'>
          {/* Dropdown tipe rute */}
          <div className='flex'>
            <div className='w-[20%]'>
              <Route size={10} color='#ffffff' />
            </div>
            <div className='flex w-[80%] flex-col'>
              <Typography className='text-[10px] text-typo-normal-white md:text-[10px]'>
                Route's Type
              </Typography>
              <Select
                value={selectedRoute}
                onValueChange={(value) => setSelectedRoute(value as RouteType)}
              >
                <SelectTrigger className='flex items-center gap-1.5 rounded-lg border-0 bg-transparent text-left outline-0'>
                  <SelectValue>
                    <Typography className='text-left text-[8px] text-typo-normal-white md:text-[8px]'>
                      {selectedRoute === 'normal_route'
                        ? 'Normal Route'
                        : 'Safest Route'}
                    </Typography>
                  </SelectValue>
                  <ChevronDown color='#ffffff' width={10} height={10} />
                </SelectTrigger>
                <SelectContent className='rounded-sm bg-white p-1.5 shadow-lg'>
                  <SelectItem
                    value='normal_route'
                    className='rounded px-1 py-1.5 hover:bg-gray-200'
                  >
                    <Typography className='text-[8px] text-gray-800 md:text-[8px]'>
                      Normal Route
                    </Typography>
                  </SelectItem>
                  <SelectItem
                    value='safest_route'
                    className='rounded px-1 py-1.5 hover:bg-gray-200'
                  >
                    <Typography className='text-[8px] text-gray-800 md:text-[8px]'>
                      Safest Route
                    </Typography>
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          {/* Dropdown tipe data */}
          <div className='flex'>
            <div className='w-[20%]'>
              <BetweenHorizontalStart size={10} color='#ffffff' />
            </div>
            <div className='flex w-[80%] flex-col'>
              <Typography className='text-[10px] text-typo-normal-white md:text-[10px]'>
                Data's Type
              </Typography>
              <Select
                value={selectedData}
                onValueChange={(value) => setSelectedData(value as DataType)}
              >
                <SelectTrigger className='flex items-center gap-1.5 rounded-lg border-0 bg-transparent text-left outline-0'>
                  <SelectValue>
                    <Typography className='text-left text-[8px] text-typo-normal-white md:text-[8px]'>
                      {selectedData === 'enviromental'
                        ? 'Enviromental'
                        : 'Ship Motion'}
                    </Typography>
                  </SelectValue>
                  <ChevronDown color='#ffffff' width={10} height={10} />
                </SelectTrigger>
                <SelectContent className='rounded-sm bg-white p-1.5 shadow-lg'>
                  <SelectItem
                    value='enviromental'
                    className='rounded px-1 py-1.5 hover:bg-gray-200'
                  >
                    <Typography className='text-[8px] text-gray-800 md:text-[8px]'>
                      Enviromental
                    </Typography>
                  </SelectItem>
                  <SelectItem
                    value='ship_motion'
                    className='rounded px-1 py-1.5 hover:bg-gray-200'
                  >
                    <Typography className='text-[8px] text-gray-800 md:text-[8px]'>
                      Ship Motion
                    </Typography>
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>

        {/* Komponen LineChart */}
        <div className='relative'>
          <LineChart
            dataset={currentDataset.map((item) => ({
              ...item,
              highlightIndex: currentIndex,
            }))}
          />
          <div
            className='absolute top-0 h-full w-[2px] bg-red-500'
            style={{
              left: `${
                1.9 +
                (currentIndex / (currentDataset[0].data.length - 1)) * 93.5
              }%`,
              transition: 'left 0.5s linear',
            }}
          ></div>
        </div>

        {/* Komponen EnviromentalSeeker */}
        <EnviromentalSeeker data={seekerData} maxValues={maxValues} />
      </section>
    </div>
  );
}
