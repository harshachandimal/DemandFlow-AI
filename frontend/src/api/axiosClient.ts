import axios, { AxiosResponse, AxiosError } from 'axios';

const axiosClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api',
  headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
  timeout: 30_000,
});

// Response error interceptor — normalise error messages
axiosClient.interceptors.response.use(
  (res: AxiosResponse) => res,
  (err: AxiosError<any>) => {
    const message =
      err.response?.data?.error ||
      err.response?.data?.message ||
      err.message ||
      'An unexpected error occurred.';
    return Promise.reject(new Error(message));
  },
);

export default axiosClient;
