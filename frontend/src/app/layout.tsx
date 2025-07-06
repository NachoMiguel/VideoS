import React from 'react'
import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { Toaster } from '@/components/ui/toaster'
import { ReactPlugin } from '@stagewise-plugins/react'
import { StagewiseToolbar } from '@stagewise/toolbar-next'
import ErrorBoundary from '@/components/ErrorBoundary'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'AI Video Slicer',
  description: 'AI-powered video editing with face recognition and script-driven assembly',
  keywords: ['AI', 'video editing', 'face recognition', 'automatic editing', 'video slicer'],
  authors: [{ name: 'AI Video Slicer Team' }],
  viewport: 'width=device-width, initial-scale=1',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${inter.className} min-h-screen bg-background antialiased`}>
        <ErrorBoundary>
          <main className="relative flex min-h-screen flex-col">
            {children}
          </main>
          <Toaster />
          <StagewiseToolbar config={{ plugins: [ReactPlugin] }} />
        </ErrorBoundary>
      </body>
    </html>
  )
} 