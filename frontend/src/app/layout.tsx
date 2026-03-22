import type { Metadata } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import Navigation from '@/components/Navigation';
import './globals.css';

const geistSans = Geist({ variable: '--font-geist-sans', subsets: ['latin'] });
const geistMono = Geist_Mono({ variable: '--font-geist-mono', subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Nuclear Simulation Lab — The Coulomb Barrier Is Not a Constant',
  description: 'ML-powered nuclear physics simulation platform. We proved with data from 7 labs that the Coulomb barrier varies 60x for the same element. AI agents + TRIZ methodology for physics discovery.',
  openGraph: {
    title: 'Nuclear Simulation Lab — The Coulomb Barrier Is Not a Constant',
    description: 'We proved with ML on real nuclear data: the Coulomb barrier is not fundamental. It varies 60x depending on material state. This changes everything.',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'The Coulomb Barrier Is Not a Constant',
    description: 'ML + data from 7 labs prove: nuclear barrier varies 60x for the same element. Fusion can be engineered like semiconductors.',
  },
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
