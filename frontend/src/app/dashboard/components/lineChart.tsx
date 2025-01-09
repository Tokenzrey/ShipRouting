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
  height?: number; // Tinggi grafik (default: 300)
  dataset: Array<{
    label: string; // Label dataset (contoh: suhu, tekanan, dll.)
    data: number[]; // Data array untuk setiap dataset
    color: string; // Warna garis dataset
    yAxisID: string; // ID sumbu y untuk dataset
  }>;
  stepPercentage?: number; // Interval persentase antar label (default: 5%)
}

// Fungsi utilitas untuk menghasilkan label dinamis
const generateLabels = (
  length: number,
  stepPercentage: number = 5,
): string[] => {
  if (length < 2) {
    return ['arrival', 'Departure'];
  }

  const labels: string[] = ['arrival'];
  const numSteps = Math.floor(100 / stepPercentage);
  const stepSize = Math.floor(length / numSteps);

  for (let i = 1; i < numSteps; i++) {
    const index = i * stepSize;
    if (index < length - 1) {
      // Contoh label: "5%", "10%", ..., "95%"
      labels.push(`${i * stepPercentage}%`);
    }
  }

  labels.push('Departure');
  return labels;
};

// Komponen LineChart
const LineChart: React.FC<LineChartProps> = ({
  height = 300,
  dataset,
  stepPercentage = 5,
}) => {
  // Validasi bahwa semua dataset memiliki panjang data yang sama
  const dataLength = dataset.length > 0 ? dataset[0].data.length : 0;
  const allSameLength = dataset.every(
    (item) => item.data.length === dataLength,
  );

  if (!allSameLength) {
    console.error('Semua dataset harus memiliki panjang data yang sama.');
    return null;
  }

  // Menghasilkan label dinamis berdasarkan panjang dataset dan interval 5%
  const labels = generateLabels(dataLength, stepPercentage);

  // Konfigurasi data untuk grafik
  const data: ChartData<'line'> = {
    labels: labels, // Label sumbu x yang dihasilkan secara dinamis
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
};

export default LineChart;
