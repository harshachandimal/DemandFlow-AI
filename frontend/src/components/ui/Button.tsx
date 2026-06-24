import React from 'react';
import { Loader2 } from 'lucide-react';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
  loading?: boolean;
  icon?: React.ReactNode;
}

export default function Button({ 
  children, 
  variant = 'primary', 
  loading = false, 
  icon,
  className = '',
  disabled,
  ...props 
}: ButtonProps) {
  
  const baseStyles = "inline-flex items-center justify-center gap-2 rounded-[0.625rem] font-semibold transition-all duration-200 cursor-pointer";
  
  const variants = {
    primary: "btn-primary",
    secondary: "bg-slate-800/50 hover:bg-slate-700/50 text-slate-200 border border-slate-700/50 shadow-sm",
    ghost: "bg-transparent hover:bg-slate-800/40 text-slate-400 hover:text-slate-200",
    danger: "bg-red-500/10 hover:bg-red-500/20 text-red-500 border border-red-500/20"
  };

  const styleClass = variants[variant];
  const paddingClass = "px-5 py-[0.65rem] text-sm";
  const disabledClass = (disabled || loading) ? "opacity-50 cursor-not-allowed transform-none" : "";

  return (
    <button 
      className={`${baseStyles} ${styleClass} ${paddingClass} ${disabledClass} ${className}`}
      disabled={disabled || loading}
      {...props}
    >
      {loading ? (
        <Loader2 size={16} className="animate-spin" />
      ) : icon ? (
        icon
      ) : null}
      {children}
    </button>
  );
}
