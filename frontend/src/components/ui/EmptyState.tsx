import React from 'react';
import Card from './Card';

interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  description: string;
  action?: React.ReactNode;
}

export default function EmptyState({ icon, title, description, action }: EmptyStateProps) {
  return (
    <Card className="flex flex-col items-center justify-center text-center p-12 border-dashed border-slate-700/50">
      {icon && (
        <div className="mb-4 text-slate-500">
          {icon}
        </div>
      )}
      <h3 className="text-lg font-semibold text-slate-200 mb-2">{title}</h3>
      <p className="text-sm text-slate-400 max-w-sm mx-auto mb-6">
        {description}
      </p>
      {action && (
        <div>
          {action}
        </div>
      )}
    </Card>
  );
}
