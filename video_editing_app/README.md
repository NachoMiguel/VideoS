# Video Editing App

Standalone video editing application for scene analysis and video assembly.

## Setup

### Backend
```bash
cd video_editing_app
pip install fastapi uvicorn opencv-python moviepy numpy
python main.py
```

### Frontend
```bash
cd video_editing_app
npm install
npm run dev
```

## Usage

1. Upload 2-3 video files (MP4, AVI, MOV, MKV)
2. Upload audio file (MP3, WAV, OGG)
3. Click "Start Editing"
4. Download final video when complete

## Features

- Scene analysis with face detection
- Character detection in scenes
- Video assembly with audio synchronization
- Modern Notion-style UI
- Real-time progress tracking 