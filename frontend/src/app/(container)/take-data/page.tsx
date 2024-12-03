'use client';
import React, { useEffect, useState } from 'react';
const DATASET_URL =
  'http://nomads.ncep.noaa.gov:80/dods/wave/gfswave/20241201/gfswave.global.0p16_00z';

interface VariableData {
  variable: string;
  data: any;
  dimensions?: any;
}

const Page: React.FC = () => {
  const [variables, setVariables] = useState<string[]>([
    'htsgwsfc',
    'dirpwsfc',
    'windsfc',
  ]);
  const [dataset, setDataset] = useState<VariableData[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchData = async (variable: string) => {
    try {
      // First, fetch the DDS (Dataset Descriptor Structure)
      const ddsResponse = await fetch(`${DATASET_URL}.dds`);
      const ddsText = await ddsResponse.text();

      // Fetch the DAS (Dataset Attribute Structure)
      const dasResponse = await fetch(`${DATASET_URL}.das`);
      const dasText = await dasResponse.text();

      // Fetch the actual data for the specific variable
      const dataResponse = await fetch(`${DATASET_URL}.dods?${variable}`);

      if (!dataResponse.ok) {
        throw new Error(
          `Failed to fetch ${variable}: ${dataResponse.statusText}`,
        );
      }

      // Convert response to ArrayBuffer
      const buffer = await dataResponse.arrayBuffer();

      // You might need to use a specific DODS/OPeNDAP parsing library here
      // This is a simplified example and might need more robust parsing
      return {
        variable,
        data: buffer,
        dimensions: {
          dds: ddsText,
          das: dasText,
        },
      };
    } catch (err) {
      console.error(err);
      setError(`Error fetching variable ${variable}`);
      return { variable, data: null };
    }
  };

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      setError(null);

      try {
        const results = await Promise.all(
          variables.map((variable) => fetchData(variable)),
        );
        setDataset(results);
      } catch (err) {
        console.error(err);
        setError('Error loading dataset');
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [variables]);

  return (
    <div className='p-4'>
      <h1 className='mb-4 text-xl font-bold'>Wave Data Visualization</h1>
      {loading && <p>Loading data...</p>}
      {error && <p className='text-red-500'>{error}</p>}
      <div className='space-y-4'>
        {dataset.map((variableData) => (
          <div key={variableData.variable} className='rounded-lg border p-4'>
            <h2 className='text-lg font-semibold'>{variableData.variable}</h2>
            {variableData.dimensions && (
              <div>
                <h3 className='font-medium'>DDS:</h3>
                <pre className='max-h-48 overflow-auto rounded bg-gray-100 p-2 text-sm'>
                  {variableData.dimensions.dds}
                </pre>
                <h3 className='font-medium'>DAS:</h3>
                <pre className='max-h-48 overflow-auto rounded bg-gray-100 p-2 text-sm'>
                  {variableData.dimensions.das}
                </pre>
              </div>
            )}
            <pre className='max-h-96 overflow-auto rounded bg-gray-100 p-2 text-sm'>
              {variableData.data
                ? `Data buffer length: ${variableData.data.byteLength} bytes`
                : 'No data retrieved'}
            </pre>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Page;
