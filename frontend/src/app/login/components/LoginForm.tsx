'use client';
import * as React from 'react';
import { FormProvider, useForm } from 'react-hook-form';

import { useLogin } from '@/app/login/hooks/login';
import Button from '@/components/buttons/Button';
import { UserRoundCheck } from 'lucide-react';
import Input from '@/components/form/Input';
import { LoginFormRequest } from '@/types/login';

export default function LoginForm() {
  const methods = useForm<LoginFormRequest>({
    mode: 'onTouched',
  });

  const { handleSubmit } = methods;
  const { handleLogin, isPending } = useLogin();

  const onSubmit = (data: LoginFormRequest) => {
    handleLogin(data);
  };

  return (
    <FormProvider {...methods}>
      <form
        onSubmit={handleSubmit(onSubmit)}
        className='glass mx-auto flex w-full flex-col items-center justify-center gap-3 rounded-xl p-6 md:w-[73%] lg:w-full'
      >
        <div className='w-full space-y-2'>
          <Input
            id='username'
            label='Username'
            className='w-full'
            placeholder='Masukkan username'
            defaultValue='Admin'
            validation={{
              required: 'Username tidak boleh kosong!',
            }}
          />

          <Input
            label='Password'
            id='password'
            type='password'
            className='w-full'
            placeholder='Masukkan Password'
            defaultValue='123'
            validation={{
              required: 'Password tidak boleh kosong!',
            }}
          />
        </div>
        <Button
          type='submit'
          variant='info'
          appearance='light'
          className='mt-4 w-full py-1.5'
          isLoading={isPending}
          rightIcon={UserRoundCheck}
        >
          LOGIN
        </Button>
      </form>
    </FormProvider>
  );
}
