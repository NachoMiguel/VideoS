import * as Sentry from '@sentry/nextjs'

Sentry.init({
  dsn: process.env.NEXT_PUBLIC_SENTRY_DSN,
  environment: process.env.NEXT_PUBLIC_SENTRY_ENVIRONMENT || 'development',
  release: process.env.NEXT_PUBLIC_SENTRY_RELEASE,
  
  // Error Monitoring
  sampleRate: parseFloat(process.env.NEXT_PUBLIC_SENTRY_SAMPLE_RATE || '1.0'),
  
  // Performance Monitoring
  tracesSampleRate: parseFloat(process.env.NEXT_PUBLIC_SENTRY_TRACES_SAMPLE_RATE || '0.1'),
  
  // Disable in development unless explicitly enabled
  enabled: process.env.NODE_ENV === 'production' || process.env.NEXT_PUBLIC_SENTRY_ENABLED === 'true',
}) 