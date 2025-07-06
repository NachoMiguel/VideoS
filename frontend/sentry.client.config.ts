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
  
  // Additional configuration
  beforeSend(event, hint) {
    // Filter out development errors we don't want to track
    if (process.env.NODE_ENV === 'development') {
      console.log('Sentry Event:', event)
    }
    return event
  },
  
  integrations: [
    Sentry.replayIntegration({
      // Capture 10% of all sessions for replay
      sessionSampleRate: 0.1,
      // Capture 100% of sessions with errors for replay
      errorSampleRate: 1.0,
    }),
  ],
}) 