import { AlertCircle } from 'lucide-react';
import Card from './Card';

interface ErrorStateProps {
  message: string;
  title?: string;
}

export default function ErrorState({ message, title = 'Error' }: ErrorStateProps) {
  return (
    <Card className="bg-red-500/5 border-red-500/20 text-center p-8">
      <div className="flex justify-center mb-3">
        <div className="p-3 bg-red-500/10 rounded-full text-red-400">
          <AlertCircle size={24} />
        </div>
      </div>
      <h3 className="text-red-400 font-semibold mb-2">{title}</h3>
      <p className="text-red-400/80 text-sm max-w-md mx-auto">{message}</p>
    </Card>
  );
}
