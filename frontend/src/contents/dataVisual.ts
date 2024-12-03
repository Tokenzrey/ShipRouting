export const datasets = {
  normal_route: {
    enviromental: [
      {
        label: 'Wave Height (m)',
        data: [1.5, 1.8, 1.2, 2.0, 1.6, 2.1, 1.9, 1.7],
        color: '#ff0000',
        yAxisID: 'y1',
      },
      {
        label: 'Wave Period (s)',
        data: [4, 5, 3, 6, 4, 5, 4, 3],
        color: '#00ff00',
        yAxisID: 'y2',
      },
      {
        label: 'Wave Heading (deg)',
        data: [45, 60, 90, 120, 80, 100, 70, 90],
        color: '#ffff00',
        yAxisID: 'y3',
      },
    ],
    ship_motion: [
      {
        label: 'Roll (deg)',
        data: [10, 15, 12, 8, 13, 9, 10, 14],
        color: '#ff0000',
        yAxisID: 'y4',
      },
      {
        label: 'Heave (m)',
        data: [1, 1.2, 1.1, 1.3, 1.0, 1.4, 1.2, 1.3],
        color: '#00ff00',
        yAxisID: 'y5',
      },
      {
        label: 'Pitch (deg)',
        data: [3, 5, 4, 6, 4, 5, 3, 4],
        color: '#ffff00',
        yAxisID: 'y6',
      },
    ],
  },
  safest_route: {
    enviromental: [
      {
        label: 'Wave Height (m)',
        data: [1.2, 1.4, 1.1, 1.8, 1.3, 1.5, 1.6, 1.9],
        color: '#ff0000',
        yAxisID: 'y1',
      },
      {
        label: 'Wave Period (s)',
        data: [3, 4, 5, 3, 6, 5, 4, 3],
        color: '#00ff00',
        yAxisID: 'y2',
      },
      {
        label: 'Wave Heading (deg)',
        data: [60, 90, 100, 120, 80, 70, 110, 100],
        color: '#ffff00',
        yAxisID: 'y3',
      },
    ],
    ship_motion: [
      {
        label: 'Roll (deg)',
        data: [8, 10, 12, 9, 13, 10, 11, 14],
        color: '#ff0000',
        yAxisID: 'y4',
      },
      {
        label: 'Heave (m)',
        data: [1.1, 1.3, 1.0, 1.4, 1.2, 1.1, 1.3, 1.2],
        color: '#00ff00',
        yAxisID: 'y5',
      },
      {
        label: 'Pitch (deg)',
        data: [4, 3, 5, 4, 6, 4, 5, 3],
        color: '#ffff00',
        yAxisID: 'y6',
      },
    ],
  },
};
