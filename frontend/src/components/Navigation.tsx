'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

const NAV_ITEMS = [
  { href: '/', label: 'Overview', icon: '◈' },
  { href: '/physics', label: 'Models', icon: '⚛' },
  { href: '/materials', label: 'Materials', icon: '◆' },
  { href: '/experiments', label: 'Data', icon: '◉' },
  { href: '/simulator', label: 'Simulator', icon: '▶' },
  { href: '/tsc', label: 'TSC', icon: '✦' },
  { href: '/3d', label: '3D Lab', icon: '🔮' },
];

export default function Navigation() {
  const pathname = usePathname();

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 border-b border-[#2a2a40] bg-[#0a0a0f]/90 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <div className="flex items-center justify-between h-14">
          <Link href="/" className="flex items-center gap-2 text-white font-semibold text-lg">
            <span className="text-blue-400 text-xl">⚡</span>
            <span className="hidden sm:inline">Nuclear Sim</span>
            <span className="sm:hidden">NSim</span>
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
            <a
              href="https://github.com/ORTODOX1/Syniz"
              target="_blank"
              rel="noopener noreferrer"
              className="ml-2 px-3 py-1.5 rounded-md text-sm font-medium text-gray-400 hover:text-white hover:bg-white/5 transition-all"
            >
              <span className="mr-1">↗</span>
              <span className="hidden lg:inline">GitHub</span>
            </a>
          </div>
        </div>
      </div>
    </nav>
  );
}
