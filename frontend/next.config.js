/** @type {import('next').NextConfig} */
const nextConfig = {
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
  // ADD THESE TIMEOUT CONFIGURATIONS:
  experimental: {
    proxyTimeout: 180000, // 3 minutes in milliseconds
  },
  serverRuntimeConfig: {
    // Server-side timeout settings
    timeout: 180000
  }
}

module.exports = nextConfig 