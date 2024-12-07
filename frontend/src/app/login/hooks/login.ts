import { useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';

import { showToast } from '@/components/Toast';
import useAuthStore from '@/lib/Auth/useAuthStore';
import { LoginFormRequest } from '@/types/login';
import { dummyPassword, dummyUsername } from '@/constant/user';

export const useLogin = () => {
  const [isPending, setIsPending] = useState(false);
  const router = useRouter();
  const searchParams = useSearchParams();
  const { login } = useAuthStore();

  const handleLogin = async (data: LoginFormRequest) => {
    const { username, password } = data;
    setIsPending(true); // Set isPending menjadi true saat proses login dimulai
    try {
      if (username === dummyUsername && password === dummyPassword) {
        // Set data user di store dengan status login berhasil
        login( dummyUsername );

        showToast(
          'Login Success',
          'You have successfully logged in.',
          'SUCCESS',
        );

        // Redirect setelah login berhasil
        const redirect = searchParams.get('redirect') || '/';
        router.push(redirect);

        return { username: dummyUsername }; // Kembali sebagai LoginFormResponse jika dibutuhkan
      } else {
        throw new Error('Invalid credentials. Please try again.');
      }
    } catch (err) {
      // Menangani error selama proses login
      showToast(
        'Login Failed',
        'Login failed. Please try again.',
        'ERROR',
      );
    } finally {
      setIsPending(false); // Set isPending menjadi false setelah login selesai
    }
  };

  return { handleLogin, isPending };
};
