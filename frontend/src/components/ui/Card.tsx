import React from 'react';

interface CardProps {
  children: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
  padding?: string;
}

export default function Card({ children, className = '', style = {}, padding = '1.5rem' }: CardProps) {
  return (
    <div
      className={`glass-card ${className}`}
      style={{ padding, ...style }}
    >
      {children}
    </div>
  );
}
