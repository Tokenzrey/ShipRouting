import React from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ChartOptions,
  ChartData,
} from 'chart.js';

// Registrasi komponen untuk Chart.js agar dapat digunakan
ChartJS.register(
  CategoryScale, // Untuk skala kategori pada sumbu x
  LinearScale, // Untuk skala linear pada sumbu y
  PointElement, // Untuk elemen titik dalam grafik
  LineElement, // Untuk elemen garis dalam grafik
  Title, // Untuk menambahkan judul grafik
  Tooltip, // Untuk tooltip pada grafik
  Legend, // Untuk legenda pada grafik
);

// Definisi tipe properti untuk komponen LineChart
interface LineChartProps {
  height?: number; // Tinggi grafik (default: 90)
  dataset: Array<{
    label: string; // Label dataset (contoh: suhu, tekanan, dll.)
    data: number[]; // Data array untuk setiap dataset
    color: string; // Warna garis dataset
    yAxisID: string; // ID sumbu y untuk dataset
  }>;
}

// Komponen LineChart
export default function LineChart({ height = 90, dataset }: LineChartProps) {
  // Konfigurasi data untuk grafik
  const data: ChartData<'line'> = {
    labels: [
      'arrival',
      '10 nm',
      '20 nm',
      '30 nm',
      '40 nm',
      '50 nm',
      '60 nm',
      'Depature',
    ], // Label sumbu x
    datasets: dataset.map((item) => ({
      label: item.label, // Label dataset
      data: item.data, // Data untuk dataset
      borderColor: item.color, // Warna garis dataset
      backgroundColor: item.color, // Warna latar belakang titik
      borderWidth: 1.5, // Ketebalan garis
      pointRadius: 1.75, // Ukuran titik
      tension: 0, // Ketegangan garis (0 untuk garis lurus)
      yAxisID: item.yAxisID, // ID sumbu y
    })),
  };

  // Konfigurasi opsi untuk grafik
  const options: ChartOptions<'line'> = {
    responsive: true, // Grafik responsif terhadap ukuran layar
    maintainAspectRatio: false, // Tidak mempertahankan rasio aspek
    interaction: {
      mode: 'index', // Interaksi tooltip berdasarkan indeks
      intersect: false, // Tooltip tidak membutuhkan interseksi langsung
    },
    plugins: {
      legend: {
        display: false, // Tidak menampilkan legenda
      },
      tooltip: {
        enabled: true, // Tooltip diaktifkan
        mode: 'index', // Mode tooltip per indeks
        intersect: false, // Tooltip tidak memerlukan interseksi langsung
      },
    },
    scales: {
      x: {
        position: 'top', // Sumbu x di posisi atas
        grid: { color: '#7E7474' }, // Warna grid sumbu x
        ticks: { color: '#ffffff' }, // Warna teks sumbu x
      },
      // Mengatur skala sumbu y berdasarkan dataset
      ...Object.fromEntries(
        dataset.map(({ yAxisID, data }) => [
          yAxisID,
          {
            type: 'linear',
            display: true,
            position: yAxisID === 'y1' || yAxisID === 'y4' ? 'left' : 'right', // Posisi sumbu y
            grid: {
              color: '#7E7474',
              drawOnChartArea: yAxisID === 'y1' || yAxisID === 'y4', // Gambar grid hanya untuk y1 atau y4
            },
            ticks: {
              color: '#ffffff', // Warna teks sumbu y
              min: Math.min(...data) - 1, // Nilai minimum sumbu y
              max: Math.max(...data) + 1, // Nilai maksimum sumbu y
            },
          },
        ]),
      ),
    },
  };

  // Render komponen
  return (
    <div className='w-full bg-gray-800 px-1 py-2' style={{ height: height }}>
      <Line data={data} options={options} />
    </div>
  );
}
