import React from 'react';

// Definisi tipe untuk properti (props) komponen Popup
interface PopupProps {
  placeName?: string; // Nama tempat yang ditampilkan di popup
  latitude?: number; // Nilai latitude koordinat
  longitude?: number; // Nilai longitude koordinat
  waveData?: {
    dirpwsfc: number;
    htsgwsfc: number;
    perpwsfc: number;
  };
  loading?: boolean; // Status loading, true jika data sedang dimuat
  onClose?: () => void; // Fungsi callback untuk menutup popup
}

// Komponen Popup untuk menampilkan informasi lokasi
const Popup: React.FC<PopupProps> = ({
  placeName, // Nama tempat yang akan ditampilkan
  latitude, // Koordinat latitude
  longitude, // Koordinat longitude
  waveData,
  loading = false, // Status loading, default-nya false
  onClose, // Callback untuk menutup popup
}) => {
  console.log(waveData);
  return (
    <div className='w-44 rounded bg-white p-2 shadow-md'>
      {/* Tombol untuk menutup popup */}
      <button
        onClick={(e) => {
          e.stopPropagation(); // Mencegah event klik memengaruhi elemen lain
          onClose?.(); // Memanggil fungsi onClose jika ada
        }}
        className='float-right cursor-pointer border-none bg-transparent text-base'
      >
        &times; {/* Ikon "X" untuk menutup */}
      </button>

      {/* Kondisi jika status loading true */}
      {loading ? (
        <div>
          <strong>Loading...</strong> {/* Menampilkan teks loading */}
          {latitude !== undefined && longitude !== undefined && (
            <>
              <br />
              <span>Lat: {latitude.toFixed(6)}</span>{' '}
              {/* Menampilkan latitude */}
              <br />
              <span>Lon: {longitude.toFixed(6)}</span>{' '}
              {/* Menampilkan longitude */}
            </>
          )}
        </div>
      ) : (
        // Kondisi jika status loading false
        <div>
          <strong>{placeName || 'Tidak ada nama'}</strong> {/* Nama tempat */}
          <br />
          <span>Lat: {latitude?.toFixed(6)}</span> {/* Menampilkan latitude */}
          <br />
          <span>Lon: {longitude?.toFixed(6)}</span>
          {waveData && (
            <>
              <br />
              <strong>Wave Data:</strong>
              <br />
              <span>Dirpwsfc: {waveData.dirpwsfc.toFixed(6)}</span>
              <br />
              <span>Htsgwsfc: {waveData.htsgwsfc.toFixed(6)}</span>
              <br />
              <span>Perpwsfc: {waveData.perpwsfc.toFixed(6)}</span>
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default Popup;
