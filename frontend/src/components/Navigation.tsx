'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

const NAV_ITEMS = [
  { href: '/', label: 'Обзор', icon: '◈' },
  { href: '/physics', label: 'Физика', icon: '⚛' },
  { href: '/materials', label: 'Материалы', icon: '◆' },
  { href: '/experiments', label: 'Эксперименты', icon: '◉' },
  { href: '/simulator', label: 'Симулятор', icon: '▶' },
  { href: '/tsc', label: 'TSC', icon: '✦' },
];

export default function Navigation() {
  const pathname = usePathname();

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 border-b border-[#2a2a40] bg-[#0a0a0f]/90 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <div className="flex items-center justify-between h-14">
          <Link href="/" className="flex items-center gap-2 text-white font-semibold text-lg">
            <span className="text-blue-400 text-xl">⚡</span>
            <span className="hidden sm:inline">LENR Simulator</span>
            <span className="sm:hidden">LENR</span>
          </Link>

          <div className="flex items-center gap-1">
            {NAV_ITEMS.map(({ href, label, icon }) => {
              const isActive = pathname === href;
              return (
                <Link
                  key={href}
                  href={href}
                  className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
                    isActive
                      ? 'bg-blue-500/15 text-blue-400 border border-blue-500/30'
                      : 'text-gray-400 hover:text-white hover:bg-white/5'
                  }`}
                >
                  <span className="mr-1.5">{icon}</span>
                  <span className="hidden md:inline">{label}</span>
                </Link>
              );
            })}
          </div>
        </div>
      </div>
    </nav>
  );
}
