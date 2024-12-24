// Bilinear Interpolation Function
  const bilinearInterpolation = useCallback(
    (
      data: number[][],
      x: number,
      y: number,
      xFrac: number,
      yFrac: number,
    ): number => {
      // Ensure indices are within bounds
      if (y >= data.length - 1 || x >= data[0].length - 1 || y < 0 || x < 0) {
        return 0;
      }

      const val00 = data[y][x];
      const val10 = data[y][x + 1];
      const val01 = data[y + 1][x];
      const val11 = data[y + 1][x + 1];

      const val0 = val00 * (1 - xFrac) + val10 * xFrac;
      const val1 = val01 * (1 - xFrac) + val11 * xFrac;

      return val0 * (1 - yFrac) + val1 * yFrac;
    },
    [],
  );