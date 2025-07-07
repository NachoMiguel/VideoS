'use client'

import React from 'react'

interface ClientLayoutWrapperProps {
  children: React.ReactNode
}

export default function ClientLayoutWrapper({ children }: ClientLayoutWrapperProps) {
  return (
    <main className="relative flex min-h-screen flex-col">
      {children}
    </main>
  )
}
