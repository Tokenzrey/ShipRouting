export const getFormattedDate = () => {
  const now = new Date();
  const formatter = new Intl.DateTimeFormat('en-US', {
    month: 'long',
    day: 'numeric',
    year: 'numeric',
  });

  // Format tanggal menjadi "November 29th, 2024"
  const [month, day, year] = formatter.format(now).split(' ');
  const dayWithSuffix = `${day}${getDaySuffix(parseInt(day))}`;
  return `${month} ${dayWithSuffix}, ${year}`;
};

const getDaySuffix = (day: number) => {
  if (day >= 11 && day <= 13) return 'th'; // Handle special cases: 11th, 12th, 13th
  switch (day % 10) {
    case 1:
      return 'st';
    case 2:
      return 'nd';
    case 3:
      return 'rd';
    default:
      return 'th';
  }
};
