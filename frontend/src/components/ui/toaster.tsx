'use client'

import { useEffect, useState } from 'react'
import { ReactPlugin } from '@stagewise-plugins/react'
import { StagewiseToolbar } from '@stagewise/toolbar-next'

export default function ClientStagewiseToolbar() {
  const [shouldRender, setShouldRender] = useState(false)
  const [hasError, setHasError] = useState(false)

  useEffect(() => {
    // Only render in development mode
    if (process.env.NODE_ENV !== 'development') {
      return
    }

    // Delay rendering until after hydration to avoid SSR/client mismatches
    // and ensure React internals are fully initialized
    const timer = setTimeout(() => {
      setShouldRender(true)
    }, 100)

    return () => clearTimeout(timer)
  }, [])

  // Don't render if not in development or if there was an error
  if (process.env.NODE_ENV !== 'development' || !shouldRender || hasError) {
    return null
  }

  // Error boundary for the StagewiseToolbar to prevent crashes
  try {
    return (
      <div style={{ position: 'fixed', zIndex: 9999 }}>
        <StagewiseToolbar 
          config={{ plugins: [ReactPlugin] }}
        />
      </div>
    )
  } catch (error) {
    // Fallback if StagewiseToolbar throws during render
    console.warn('StagewiseToolbar failed to render (non-critical development tool):', error)
    setHasError(true)
    return null
  }
}