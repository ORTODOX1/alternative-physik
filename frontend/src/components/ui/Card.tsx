import { ReactNode } from 'react';

interface CardProps {
  children: ReactNode;
  className?: string;
  title?: string;
  subtitle?: string;
  accent?: 'blue' | 'green' | 'amber' | 'red' | 'purple';
  hover?: boolean;
}

const accentColors = {
  blue: 'border-blue-500/30 hover:border-blue-500/50',
  green: 'border-emerald-500/30 hover:border-emerald-500/50',
  amber: 'border-amber-500/30 hover:border-amber-500/50',
  red: 'border-red-500/30 hover:border-red-500/50',
  purple: 'border-purple-500/30 hover:border-purple-500/50',
};

const accentBg = {
  blue: 'bg-blue-500/10',
  green: 'bg-emerald-500/10',
  amber: 'bg-amber-500/10',
  red: 'bg-red-500/10',
  purple: 'bg-purple-500/10',
};

export default function Card({ children, className = '', title, subtitle, accent, hover = false }: CardProps) {
  return (
    <div
      className={`
        rounded-xl border bg-[#1a1a2e] border-[#2a2a40]
        ${accent ? accentColors[accent] : ''}
        ${hover ? 'transition-all hover:bg-[#22223a] hover:translate-y-[-1px]' : ''}
        ${className}
      `}
    >
      {(title || subtitle) && (
        <div className={`px-5 py-4 border-b border-[#2a2a40] ${accent ? accentBg[accent] : ''}`}>
          {title && <h3 className="text-base font-semibold text-white">{title}</h3>}
          {subtitle && <p className="text-sm text-gray-400 mt-0.5">{subtitle}</p>}
        </div>
      )}
      <div className="p-5">{children}</div>
    </div>
  );
}
