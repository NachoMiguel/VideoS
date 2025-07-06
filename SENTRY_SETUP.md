# Sentry Integration Setup Guide

## ✅ Complete Integration Status

Your AI Video Slicer application now has complete Sentry integration:

### Created Resources:
- **Team:** AI Video Slicer (`ai-video-slicer`)
- **Frontend Project:** AI Video Slicer Frontend (`ai-video-slicer-frontend`)
- **Backend Project:** AI Video Slicer Backend (`ai-video-slicer-backend`)

### Integration Components:
- ✅ Sentry SDK dependencies added to both frontend and backend
- ✅ Configuration added to backend settings
- ✅ Frontend instrumentation files created
- ✅ Backend initialization added to main.py
- ✅ Next.js config updated for Sentry

## 🔧 Required Environment Variables

### Backend (.env file in /backend directory):
```env
# Sentry Configuration
SENTRY_DSN=https://4e208570f0fe9abb426c1d94e9f8b3b4@o4509612362432512.ingest.us.sentry.io/4509619488686080
SENTRY_ENVIRONMENT=development
SENTRY_ENABLED=true
SENTRY_SAMPLE_RATE=1.0
SENTRY_TRACES_SAMPLE_RATE=0.1

# Your existing API keys...
OPENAI_API_KEY=your_openai_api_key_here
ELEVENLABS_API_KEY_1=your_elevenlabs_api_key_1_here
YOUTUBE_API_KEY=your_youtube_api_key_here
# ... etc
```

### Frontend (.env.local file in /frontend directory):
```env
# Frontend Sentry Configuration
NEXT_PUBLIC_SENTRY_DSN=https://bf6c70cab6cee6ed21a40426305ede02@o4509612362432512.ingest.us.sentry.io/4509619488423936
NEXT_PUBLIC_SENTRY_ENVIRONMENT=development
NEXT_PUBLIC_SENTRY_ENABLED=true
NEXT_PUBLIC_SENTRY_SAMPLE_RATE=1.0
NEXT_PUBLIC_SENTRY_TRACES_SAMPLE_RATE=0.1

NODE_ENV=development
```

## 🚀 Installation Steps

### 1. Install Dependencies:
```bash
# Backend
cd backend
pip install -r requirements.txt

# Frontend
cd frontend
npm install
```

### 2. Create Environment Files:
Create the `.env` files with the variables shown above.

### 3. Test the Integration:
```bash
# Start backend (should show "Sentry initialized" message)
cd backend
python main.py

# Start frontend
cd frontend
npm run dev
```

## 📊 What You Get

### Error Monitoring:
- **Automatic Error Capture:** All uncaught exceptions in both frontend and backend
- **Performance Monitoring:** Track API response times, database queries, and page loads
- **Real-time Alerts:** Get notified when errors occur in production

### Features Available:
- **Error Tracking:** See stack traces, error frequency, and user impact
- **Performance Monitoring:** Track slow API endpoints and frontend performance
- **Release Tracking:** Monitor deployments and their impact on error rates
- **User Context:** See which users are affected by errors
- **Session Replay:** Watch user sessions to understand error context

### Development vs Production:
- **Development:** Errors logged to console + sent to Sentry (if enabled)
- **Production:** Errors automatically sent to Sentry for monitoring

## 🔍 Viewing Your Data

1. **Visit Sentry Dashboard:** https://bastion-wo.sentry.io
2. **Select Project:**
   - Frontend errors: "AI Video Slicer Frontend"
   - Backend errors: "AI Video Slicer Backend"
3. **Monitor Issues:** Real-time error tracking and performance metrics

## 🛠️ Advanced Configuration

### Custom Error Tracking:
Add custom error tracking in your code:

```python
# Backend (Python)
import sentry_sdk
sentry_sdk.capture_exception(Exception("Custom error"))
sentry_sdk.capture_message("Custom info message")
```

```javascript
// Frontend (React)
import * as Sentry from '@sentry/nextjs'
Sentry.captureException(new Error('Custom error'))
Sentry.captureMessage('Custom info message')
```

### Performance Monitoring:
Track custom performance metrics:

```python
# Backend
with sentry_sdk.start_transaction(op="video_processing", name="process_video"):
    # Your video processing code
    pass
```

```javascript
// Frontend
const transaction = Sentry.startTransaction({
  op: 'video_upload',
  name: 'Upload Video File'
})
// Your upload code
transaction.finish()
```

## 🎯 Next Steps

1. **Set up environment variables** as shown above
2. **Install dependencies** using the commands provided
3. **Test the integration** by starting both servers
4. **Generate a test error** to verify Sentry is working
5. **Check the Sentry dashboard** for error reports

Your AI Video Slicer app now has enterprise-grade error monitoring and performance tracking! 