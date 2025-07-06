// Import Sentry configuration
const { withSentryConfig } = require('@sentry/nextjs')

/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    instrumentationHook: true,
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8000/api/:path*',
      },
      {
        source: '/ws',
        destination: 'http://localhost:8000/ws',
      },
    ]
  },
  // Enable WebSocket support
  webpack: (config, { dev, isServer }) => {
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        net: false,
        tls: false,
      }
    }
    return config
  },
}

// Sentry configuration
const sentryConfig = {
  // Disable source maps upload in development
  silent: process.env.NODE_ENV === 'development',
  
  // Upload source maps to Sentry
  widenClientFileUpload: true,
  
  // Enable auto-discovery of performance monitoring
  transpileClientSDK: true,
  
  // Tunnel through Next.js to avoid ad-blockers
  tunnelRoute: '/monitoring',
  
  // Hide source maps from generated client bundles
  hideSourceMaps: true,
  
  // Disable telemetry collection
  disableLogger: true,
}

module.exports = withSentryConfig(nextConfig, sentryConfig) 