import { Loader2 } from 'lucide-react';
import Card from './Card';

interface LoadingStateProps {
  message?: string;
  minHeight?: string;
}

export default function LoadingState({ message = 'Loading...', minHeight = '200px' }: LoadingStateProps) {
  return (
    <Card 
      className="flex flex-col items-center justify-center text-slate-400" 
      style={{ minHeight }}
    >
      <Loader2 size={32} className="animate-spin text-cyan-500 mb-4" />
      <p className="text-sm font-medium">{message}</p>
    </Card>
  );
}
