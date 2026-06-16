// app/layout.js
import './globals.css';
import { Inter } from 'next/font/google';
import { cn } from '@/lib/utils.js'; // Explicit .js for utility
import { Sidebar } from '@/components/layout/Sidebar.jsx'; // Explicit .jsx
import { TopBar } from '@/components/layout/TopBar.jsx';   // Explicit .jsx
import { Toaster } from '@/components/ui/toaster.jsx';     // Explicit .jsx

const inter = Inter({ subsets: ['latin'] });

export const metadata = {
  title: 'EV Battery Health Dashboard',
  description: 'Monitor and predict the health of your EV fleet batteries.',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" className="dark">
      <body className={cn(
        'min-h-screen font-sans antialiased bg-slate-base text-foreground',
        inter.className
      )}>
        <div className="flex min-h-screen bg-slate-base text-foreground">
          <Sidebar />
          <div className="flex-1 flex flex-col">
            <TopBar />
            <main className="flex-1 p-6 lg:p-8 overflow-auto">
              <div className="max-w-[1440px] mx-auto grid gap-6">
                {children}
              </div>
            </main>
          </div>
        </div>
        <Toaster />
      </body>
    </html>
  );
}