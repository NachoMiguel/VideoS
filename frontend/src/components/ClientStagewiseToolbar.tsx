'use client'

import { ReactPlugin } from '@stagewise-plugins/react'
import { StagewiseToolbar } from '@stagewise/toolbar-next'

export default function ClientStagewiseToolbar() {
  // Only render in development mode
  if (process.env.NODE_ENV !== 'development') {
    return null
  }

  return <StagewiseToolbar config={{ plugins: [ReactPlugin] }} />
} 