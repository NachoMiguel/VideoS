# 🔍 Sentry Integration Guide - AI Video Slicer

## ✅ Integration Status

Your AI Video Slicer application now has **complete Sentry integration** with:

### 🎯 Created Sentry Resources:
- **Organization:** `bastion-wo`
- **Team:** `AI Video Slicer` (`ai-video-slicer`)
- **Frontend Project:** `AI Video Slicer Frontend` 
- **Backend Project:** `AI Video Slicer Backend`

### 🛠️ Integrated Components:
- ✅ **Backend FastAPI** - Error tracking, performance monitoring, request context
- ✅ **Frontend Next.js** - Error tracking, user sessions, component error boundaries
- ✅ **Custom Exception Handling** - Automatic error capture with context
- ✅ **Performance Monitoring** - Transaction tracking for key operations
- ✅ **User Context** - Session-based user tracking
- ✅ **WebSocket Monitoring** - Real-time error tracking

---

## 🚀 Quick Setup (5 minutes)

### 1. Install Dependencies
```bash
# Backend (if not already installed)
cd backend
pip install -r requirements.txt

# Frontend (if not already installed)  
cd frontend
npm install
```

### 2. Environment Variables

#### Backend (.env file in `/backend` directory):
```env
# Sentry Configuration
SENTRY_DSN=https://4e208570f0fe9abb426c1d94e9f8b3b4@o4509612362432512.ingest.us.sentry.io/4509619488686080
SENTRY_ENVIRONMENT=development
SENTRY_ENABLED=true
SENTRY_SAMPLE_RATE=1.0
SENTRY_TRACES_SAMPLE_RATE=0.1

# Your existing API keys
OPENAI_API_KEY=your_openai_key_here
ELEVENLABS_API_KEY_1=your_elevenlabs_key_here
YOUTUBE_API_KEY=your_youtube_key_here
```

#### Frontend (.env.local file in `/frontend` directory):
```env
# Frontend Sentry Configuration
NEXT_PUBLIC_SENTRY_DSN=https://bf6c70cab6cee6ed21a40426305ede02@o4509612362432512.ingest.us.sentry.io/4509619488423936
NEXT_PUBLIC_SENTRY_ENVIRONMENT=development
NEXT_PUBLIC_SENTRY_ENABLED=true
NEXT_PUBLIC_SENTRY_SAMPLE_RATE=1.0
NEXT_PUBLIC_SENTRY_TRACES_SAMPLE_RATE=0.1

NODE_ENV=development
```

### 3. Test the Integration
```bash
# Start backend (should show "Sentry initialized" message)
cd backend
python main.py

# Start frontend
cd frontend
npm run dev
```

---

## 📊 What You Get

### 🔥 Automatic Error Tracking
- **All exceptions** automatically captured with full context
- **Stack traces** with source code context
- **User sessions** tracked across requests
- **Request context** (URL, headers, IP) attached to errors
- **Custom error codes** for easy categorization

### 📈 Performance Monitoring
- **API response times** tracked automatically
- **Video processing** performance metrics
- **OpenAI API calls** latency monitoring
- **Database queries** and external API calls tracked
- **Frontend page loads** and user interactions

### 👥 User Context
- **Session-based tracking** - each video processing session tracked
- **User flow** across frontend and backend
- **File upload context** (filename, size, type)
- **Processing context** (video metadata, progress)

### 🎯 Smart Error Categorization
- **Service-based tagging** (OpenAI, Video Processing, etc.)
- **Operation-based tagging** (upload, processing, script generation)
- **Error type classification** (API errors, validation errors, etc.)
- **Custom error codes** for quick identification

---

## 🔍 Viewing Your Data

### Dashboard Access
1. **Visit:** https://bastion-wo.sentry.io
2. **Projects:**
   - **Frontend errors:** "AI Video Slicer Frontend"
   - **Backend errors:** "AI Video Slicer Backend"

### Key Features
- **Real-time error alerts**
- **Performance dashboards**
- **User session replays**
- **Release tracking**
- **Error trends and analytics**

---

## 🛡️ What's Being Monitored

### Backend Services
- **OpenAI Service** - Script generation, character extraction
- **Video Processor** - Frame processing, scene analysis
- **Face Detection** - Face recognition pipeline
- **WebSocket** - Real-time communication
- **API Endpoints** - All REST endpoints

### Frontend Components
- **VideoProcessor** - Upload and processing flow
- **WebSocket handling** - Real-time updates
- **Error Boundary** - Component-level error catching
- **User interactions** - Button clicks, form submissions

### Critical Operations
- **Video upload** - File validation and processing
- **Script generation** - OpenAI API calls
- **Scene analysis** - Video processing pipeline
- **Character detection** - Face recognition
- **Audio generation** - ElevenLabs integration

---

## 🎛️ Advanced Configuration

### Development vs Production
```python
# Backend - Adjust sampling rates
SENTRY_SAMPLE_RATE=1.0        # 100% in dev, 0.1 in prod
SENTRY_TRACES_SAMPLE_RATE=0.1 # 10% performance monitoring
```

### Custom Error Tracking
```python
# Backend - Add custom context
import sentry_sdk

sentry_sdk.set_context("video_processing", {
    "video_id": video_id,
    "processing_stage": "character_extraction",
    "duration": processing_time
})

# Capture custom messages
sentry_sdk.capture_message("Processing milestone reached", level="info")
```

```javascript
// Frontend - Custom error tracking
import * as Sentry from '@sentry/nextjs'

Sentry.captureException(error, {
  contexts: {
    user_action: {
      action: 'video_upload',
      filename: file.name,
      timestamp: new Date().toISOString()
    }
  }
})
```

### Performance Monitoring
```python
# Backend - Custom transactions
with sentry_sdk.start_transaction(op="video_processing", name="full_pipeline"):
    # Your processing code
    with sentry_sdk.start_span(op="face_detection", description="Detect faces"):
        # Face detection code
        pass
```

---

## 🚨 Error Examples You'll See

### Common Errors Tracked
- **Video upload failures** - Invalid format, size limits
- **OpenAI API errors** - Rate limits, invalid keys
- **Face detection failures** - No faces found, processing errors
- **WebSocket disconnections** - Network issues, server errors
- **Memory issues** - Large video processing failures

### Error Context Example
```json
{
  "error": "OpenAI API rate limit exceeded",
  "contexts": {
    "openai_operation": {
      "operation": "generate_script",
      "transcript_length": 5000,
      "model": "gpt-3.5-turbo"
    },
    "user": {
      "id": "session_123",
      "ip": "192.168.1.1"
    }
  },
  "tags": {
    "service": "openai",
    "operation": "script_generation",
    "error_type": "rate_limit"
  }
}
```

---

## 🎯 Best Practices

### 🔄 Regular Monitoring
- Check dashboard daily for new errors
- Monitor performance trends weekly
- Review error patterns monthly

### 🏷️ Error Categorization
- Use tags to categorize errors by service
- Set up alerts for critical errors
- Track error trends over time

### 📈 Performance Optimization
- Monitor slow API calls
- Track video processing performance
- Optimize based on Sentry insights

### 🔐 Privacy & Security
- Sensitive data is automatically scrubbed
- API keys are not captured in errors
- User data is anonymized by session ID

---

## 🚀 Production Deployment

### Environment Variables for Production
```env
# Backend Production
SENTRY_ENVIRONMENT=production
SENTRY_ENABLED=true
SENTRY_SAMPLE_RATE=0.1          # Reduced sampling
SENTRY_TRACES_SAMPLE_RATE=0.01  # Minimal performance monitoring

# Frontend Production
NEXT_PUBLIC_SENTRY_ENVIRONMENT=production
NEXT_PUBLIC_SENTRY_ENABLED=true
NEXT_PUBLIC_SENTRY_SAMPLE_RATE=0.1
NEXT_PUBLIC_SENTRY_TRACES_SAMPLE_RATE=0.01
```

### Release Tracking
Add release information to track deployments:
```env
SENTRY_RELEASE=ai-video-slicer@1.0.0
NEXT_PUBLIC_SENTRY_RELEASE=ai-video-slicer@1.0.0
```

---

## 💡 Next Steps

1. **✅ Set up environment variables** (required)
2. **✅ Test error reporting** - Generate a test error
3. **✅ Check dashboard** - Verify errors appear in Sentry
4. **✅ Set up alerts** - Get notified of production issues
5. **✅ Monitor performance** - Track response times and bottlenecks

## 🎉 You're All Set!

Your AI Video Slicer now has enterprise-grade error monitoring and performance tracking. Every error is automatically captured with full context, helping you debug issues faster and provide better user experiences.

**Dashboard:** https://bastion-wo.sentry.io 