import type { Metadata } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import Navigation from '@/components/Navigation';
import './globals.css';

const geistSans = Geist({ variable: '--font-geist-sans', subsets: ['latin'] });
const geistMono = Geist_Mono({ variable: '--font-geist-mono', subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'LENR Simulator — Alternative Physics ML Platform',
  description: 'ML-платформа для симуляции ядерных процессов с альтернативными физическими допущениями. 3 режима физики: Maxwell, Кулон, Черепанов.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ru">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
        <Navigation />
        <main className="pt-14 min-h-screen">{children}</main>
      </body>
    </html>
  );
}
