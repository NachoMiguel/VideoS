'use client'

import { useState, useEffect } from 'react'
import { VideoUpload } from '@/components/VideoUpload'
import { AudioUpload } from '@/components/AudioUpload'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Upload, Play, Download, CheckCircle } from 'lucide-react'

console.log('🎯 Page component rendered')

export default function Home() {
  console.log('🏠 Home component initialized')
  // Remove sessionId state
  const [videos, setVideos] = useState<File[]>([])
  const [audioFile, setAudioFile] = useState<File | null>(null)
  const [status, setStatus] = useState<'idle' | 'uploading' | 'ready' | 'processing' | 'completed' | 'error'>('idle')
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState<string>('')

  // Add state change logging
  useEffect(() => {
    console.log('📊 State changed:', { videosCount: videos.length, audioFile: !!audioFile, status })
  }, [videos, audioFile, status])

  // Remove session creation useEffect

  // Remove session check
  // if (!sessionId) {
  //   return <div>Creating session...</div>
  // }

  const handleVideosUploaded = (uploadedVideos: File[]) => {
    console.log('🎬 handleVideosUploaded called with:', uploadedVideos.length, 'videos')
    setVideos(uploadedVideos)
    console.log('📹 Videos state updated')
    
    if (audioFile) {
      console.log('✅ Both videos and audio ready, setting status to ready')
      setStatus('ready')
    } else {
      console.log('⏳ Waiting for audio upload...')
    }
  }

  const handleAudioUploaded = (uploadedAudio: File) => {
    console.log('🎵 handleAudioUploaded called with:', uploadedAudio.name)
    setAudioFile(uploadedAudio)
    console.log('🎵 Audio state updated')
    
    if (videos.length > 0) {
      console.log('✅ Both videos and audio ready, setting status to ready')
      setStatus('ready')
    } else {
      console.log('⏳ Waiting for video upload...')
    }
  }

  // Add status logging
  console.log('📊 Current status:', status, 'Videos:', videos.length, 'Audio:', !!audioFile)

  const startEditing = async () => {
    if (videos.length === 0 || !audioFile) return

    setStatus('processing')
    setProgress(0)
    setError('')

    try {
      // Remove session_id from URL
      const response = await fetch('http://localhost:8001/start-editing', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      })

      if (!response.ok) throw new Error('Failed to start editing')

      // Poll for status without session_id
      const pollStatus = async () => {
        const statusResponse = await fetch('http://localhost:8001/status')
        const statusData = await statusResponse.json()

        setProgress(statusData.progress)

        if (statusData.status === 'completed') {
          setStatus('completed')
        } else if (statusData.status === 'error') {
          setStatus('error')
          setError(statusData.error)
        } else {
          setTimeout(pollStatus, 1000)
        }
      }

      pollStatus()

    } catch (err) {
      setStatus('error')
      setError(err instanceof Error ? err.message : 'Unknown error')
    }
  }

  const downloadVideo = () => {
    window.open('http://localhost:8001/download', '_blank')
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-slate-900 mb-2">Video Editor</h1>
          <p className="text-slate-600">Upload videos and audio to create your final compilation</p>
        </div>

        {/* Main Content */}
        <div className="max-w-4xl mx-auto space-y-6">
          {/* Upload Section */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <VideoUpload 
              onVideosUploaded={handleVideosUploaded}
              // Remove: onSessionCreated={setSessionId}
            />
            <AudioUpload 
              onAudioUploaded={handleAudioUploaded}
            />
          </div>

          {/* Status Section */}
          {/* Always show status for debugging */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                Debug Status: {status} | Videos: {videos.length} | Audio: {audioFile ? 'Yes' : 'No'}
                {/* Remove: | Session: {sessionId || 'None'} */}
                </CardTitle>
              </CardHeader>
              <CardContent>
                {status === 'ready' && (
                  <div className="space-y-4">
                    <p className="text-green-600">Ready to start editing!</p>
                    <Button onClick={startEditing} className="w-full">
                      <Play className="h-4 w-4 mr-2" />
                      Start Editing
                    </Button>
                  </div>
                )}

              {status === 'idle' && (
                <div className="space-y-4">
                  <p className="text-yellow-600">Waiting for uploads...</p>
                  <p className="text-sm text-gray-500">
                    Videos: {videos.length} | Audio: {audioFile ? 'Uploaded' : 'Not uploaded'}
                  </p>
                </div>
                )}

              {status === 'uploading' && (
                  <div className="space-y-4">
                  <p className="text-blue-600">Uploading...</p>
                  <Progress value={progress} />
                  </div>
                )}
              {status === 'processing' && (
                <div className="space-y-4">
                  <div className="flex items-center gap-2">
                    <Play className="h-5 w-5 text-blue-500 animate-pulse" />
                    <span className="font-medium">Processing Video</span>
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Processing...</span>
                      <span>{progress}%</span>
                    </div>
                    <Progress value={progress} />
                  </div>
                  </div>
                )}
              </CardContent>
            </Card>
        </div>
      </div>
    </div>
  )
}
