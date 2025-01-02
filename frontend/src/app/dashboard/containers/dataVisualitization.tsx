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
import { useRouteStore } from '@/lib/GlobalState/state';

// Fungsi throttle dengan tipe parameter yang eksplisit
function throttle<T extends (...args: any[]) => void>(
  func: T,
  limit: number,
): (...args: Parameters<T>) => void {
  let lastFunc: ReturnType<typeof setTimeout> | null = null;
  let lastRan: number | null = null;

  return function (this: any, ...args: Parameters<T>) {
    const context = this;
    const now = Date.now();

    if (lastRan && now - lastRan < limit) {
      if (lastFunc) clearTimeout(lastFunc);

      lastFunc = setTimeout(
        () => {
          lastRan = Date.now();
          func.apply(context, args);
        },
        limit - (now - lastRan),
      );
    } else {
      lastRan = now;
      func.apply(context, args);
    }
  };
}

type RouteType = 'normal_route' | 'safest_route';
type DataType = 'enviromental' | 'ship_motion';

export default function DataVisualizationPage() {
  const [selectedRoute, setSelectedRoute] = useState<RouteType>('normal_route');
  const [selectedData, setSelectedData] = useState<DataType>('enviromental');
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const [height, setHeight] = useState(185); // Initial height in pixels

  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const isDraggingRef = useRef(false);
  const startYRef = useRef(0);
  const startHeightRef = useRef(0);

  const optimalRoute = useRouteStore((state) => state.optimalRoute);
  const safestRoute = useRouteStore((state) => state.safestRoute);
  const currentAnimationIndex = useRouteStore(
    (state) => state.currentAnimationIndex,
  );
  const { setCurrentAnimationIndex, setAnimationState, setActiveRoute } =
    useRouteStore();

  const datasets = {
    normal_route: {
      enviromental: [
        {
          label: 'Wave Height (m)',
          data: optimalRoute.map((p) => p.htsgwsfc),
          yAxisID: 'y1',
          color: '#ff0000',
        },
        {
          label: 'Wave Period (s)',
          data: optimalRoute.map((p) => p.perpwsfc),
          yAxisID: 'y2',
          color: '#00ff00',
        },
        {
          label: 'Wave Heading (deg)',
          data: optimalRoute.map((p) => p.dirpwfsfc),
          yAxisID: 'y3',
          color: '#ffff00',
        },
      ],
      ship_motion: [
        {
          label: 'Roll (deg)',
          data: optimalRoute.map((p) => p.Roll),
          yAxisID: 'y1',
          color: '#ff0000',
        },
        {
          label: 'Heave (m)',
          data: optimalRoute.map((p) => p.Heave),
          yAxisID: 'y2',
          color: '#00ff00',
        },
        {
          label: 'Pitch (deg)',
          data: optimalRoute.map((p) => p.Pitch),
          yAxisID: 'y3',
          color: '#ffff00',
        },
      ],
    },
    safest_route: {
      enviromental: [
        {
          label: 'Wave Height (m)',
          data: safestRoute.map((p) => p.htsgwsfc),
          yAxisID: 'y1',
          color: '#ff0000',
        },
        {
          label: 'Wave Period (s)',
          data: safestRoute.map((p) => p.perpwsfc),
          yAxisID: 'y2',
          color: '#00ff00',
        },
        {
          label: 'Wave Heading (deg)',
          data: safestRoute.map((p) => p.dirpwfsfc),
          yAxisID: 'y3',
          color: '#ffff00',
        },
      ],
      ship_motion: [
        {
          label: 'Roll (deg)',
          data: safestRoute.map((p) => p.Roll),
          yAxisID: 'y1',
          color: '#ff5722',
        },
        {
          label: 'Heave (m)',
          data: safestRoute.map((p) => p.Heave),
          yAxisID: 'y2',
          color: '#00ff00',
        },
        {
          label: 'Pitch (deg)',
          data: safestRoute.map((p) => p.Pitch),
          yAxisID: 'y3',
          color: '#ffff00',
        },
      ],
    },
  };

  const currentDataset = datasets[selectedRoute][selectedData];
  const totalDataPoints = currentDataset[0]?.data.length || 1;

  const handleMouseDown = (e: React.MouseEvent) => {
    if (!isExpanded) return;
    isDraggingRef.current = true;
    startYRef.current = e.clientY;
    startHeightRef.current = height;
    document.addEventListener('mousemove', throttledMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (!isDraggingRef.current || !containerRef.current) return;

    const deltaY = startYRef.current - e.clientY;
    const newHeight = Math.min(
      Math.max(185, startHeightRef.current + deltaY),
      window.innerHeight * 0.95,
    );

    setHeight(newHeight);
  };

  const throttledMouseMove = throttle(handleMouseMove, 200);

  const handleMouseUp = () => {
    isDraggingRef.current = false;
    document.removeEventListener('mousemove', throttledMouseMove);
    document.removeEventListener('mouseup', handleMouseUp);
  };

  const playAnimation = () => {
    setAnimationState('playing');
    setIsPlaying(true);

    if (intervalRef.current) clearInterval(intervalRef.current);

    intervalRef.current = setInterval(() => {
      const currentState = useRouteStore.getState();
      const nextIndex = (currentState.currentAnimationIndex || 0) + 1;

      if (nextIndex < totalDataPoints) {
        setCurrentAnimationIndex(nextIndex);
      } else {
        clearInterval(intervalRef.current!);
        setAnimationState('idle');
        setIsPlaying(false);
      }
    }, 500);
  };

  const pauseAnimation = () => {
    // Gunakan setter function daripada setState langsung
    setAnimationState('paused');
    setIsPlaying(false);
    if (intervalRef.current) clearInterval(intervalRef.current);
  };

  const resetAnimation = () => {
    // Gunakan setter function daripada setState langsung
    setAnimationState('idle');
    setCurrentAnimationIndex(0);
    setIsPlaying(false);
    if (intervalRef.current) clearInterval(intervalRef.current);
    setCurrentIndex(0);
  };

  useEffect(() => {
    setCurrentIndex(currentAnimationIndex || 0); // Sync local index with global state
  }, [currentAnimationIndex]);

  useEffect(() => {
    setActiveRoute(selectedRoute === 'normal_route' ? 'optimal' : 'safest'); // Sync local index with global state
  }, [selectedRoute]);

  const togglePosition = () => {
    setIsExpanded((prev) => !prev);
    if (!isExpanded) {
      setHeight(185); // Reset to minimum height when expanding
    }
  };

  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      document.removeEventListener('mousemove', throttledMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, []);

  return (
    <div
      ref={containerRef}
      className={cn(
        'fixed bottom-0 z-[2] flex w-full flex-col bg-aquaBlue transition-all duration-200 ease-in',
        !isExpanded ? 'max-h-0' : '',
      )}
      style={{ height: isExpanded ? `${height}px` : 0 }}
    >
      {/* Resize Handle */}
      {isExpanded && (
        <div
          className='absolute -top-3 left-0 z-10 flex h-4 w-full cursor-ns-resize justify-center hover:opacity-70'
          onMouseDown={handleMouseDown}
        ></div>
      )}

      {/* Control Buttons */}
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

      {/* Rest of the component remains the same */}
      <section className='relative'>
        <div className='absolute -top-24 right-5 flex w-28 flex-col gap-2 rounded-lg bg-aquaBlue p-2'>
          {/* Route Type Dropdown */}
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

          {/* Data Type Dropdown */}
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

        {/* Line Chart */}
        <div className='relative'>
          <LineChart
            dataset={currentDataset.map((item) => ({
              ...item,
              highlightIndex: currentIndex,
            }))}
            height={height - 94.5}
          />
          <div
            className='absolute top-0 h-full w-[2px] bg-red-500'
            style={{
              left: `${1.9 + (currentIndex / (totalDataPoints - 1)) * 93.5}%`,
              transition: 'left 0.5s linear',
            }}
          ></div>
        </div>

        {/* Environmental Seeker */}
        <EnviromentalSeeker
          datasets={
            datasets[selectedRoute] || { enviromental: [], ship_motion: [] }
          }
        />
      </section>
    </div>
  );
}
