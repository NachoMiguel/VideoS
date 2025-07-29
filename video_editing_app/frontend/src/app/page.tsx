'use client'

import { useState, useEffect, useCallback } from 'react'
import { VideoUpload } from '@/components/VideoUpload'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Upload, Play, Download, CheckCircle, FileText } from 'lucide-react'

console.log('🎯 Page component rendered')

interface ScriptData {
  characters: string
  scriptContent: string
  audioDuration: number
}

export default function Home() {
  console.log('🏠 Home component initialized')
  const [videos, setVideos] = useState<File[]>([])
  const [scriptData, setScriptData] = useState<ScriptData | null>(null)
  const [status, setStatus] = useState<'idle' | 'uploading' | 'ready' | 'processing' | 'completed' | 'error'>('idle')
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState<string>('')

  // Add state change logging
  useEffect(() => {
    console.log('📊 State changed:', { videosCount: videos.length, scriptData: !!scriptData, status })
  }, [videos, scriptData, status])

  // CRITICAL FIX: Auto-set ready status when both videos and script data are available
  useEffect(() => {
    if (videos.length > 0 && scriptData) {
      console.log('✅ Auto-setting status to ready - both videos and script data available')
      setStatus('ready')
    }
  }, [videos, scriptData])

  // CRITICAL FIX: Use useCallback to stabilize the function and prevent infinite loops
  const handleVideosUploaded = useCallback((uploadedVideos: File[]) => {
    console.log('🎬 handleVideosUploaded called with:', uploadedVideos.length, 'videos')
    setVideos(uploadedVideos)
    console.log('📹 Videos state updated')
    
    // Status will be set automatically by useEffect above
  }, [])

  // CRITICAL FIX: Use useCallback to stabilize the function and prevent infinite loops
  const handleScriptDataReady = useCallback((data: ScriptData) => {
    console.log('📝 handleScriptDataReady called with:', data)
    setScriptData(data)
    console.log('📝 Script data state updated')
    
    // Status will be set automatically by useEffect above
  }, [])

  // Add status logging
  console.log('📊 Current status:', status, 'Videos:', videos.length, 'Script Data:', !!scriptData)

  const startEditing = async () => {
    console.log('🚀 Starting script-based editing process')
    setStatus('processing')
    setProgress(0)
    setError('')

    if (!scriptData) {
      setError('Script data not available')
      setStatus('error')
      return
    }

    try {
      console.log('📡 Making request to backend with script data...')
      const response = await fetch('http://localhost:8001/start-editing', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({
          characters: scriptData.characters,
          script_content: scriptData.scriptContent,
          audio_duration_minutes: scriptData.audioDuration.toString()
        })
      })

      console.log('📡 Response received:', response.status, response.statusText)

      if (!response.ok) {
        const errorData = await response.json()
        console.log('❌ Backend error:', errorData)
        throw new Error(errorData.detail || 'Failed to start editing')
      }

      const data = await response.json()
      console.log('✅ Script-based editing started:', data)

      // REAL-TIME PROGRESS TRACKING
      const progressInterval = setInterval(async () => {
        try {
          const statusResponse = await fetch('http://localhost:8001/status')
          if (statusResponse.ok) {
            const statusData = await statusResponse.json()
            console.log('📊 Real status:', statusData)
            
            setProgress(statusData.progress || 0)
            
            if (statusData.status === 'completed') {
              clearInterval(progressInterval)
              setStatus('completed')
              console.log('✅ Processing completed!')
            } else if (statusData.status === 'error') {
              clearInterval(progressInterval)
              setStatus('error')
              setError('Processing failed')
              console.log('❌ Processing failed!')
            }
          }
        } catch (error) {
          console.error('❌ Status check error:', error)
        }
      }, 1000) // Check every second

    } catch (error) {
      console.error('❌ Start editing error:', error)
      setStatus('error')
      setError(error instanceof Error ? error.message : 'Failed to start editing')
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
          <h1 className="text-4xl font-bold mb-4">🎬 AI Video Editor</h1>
          <p className="text-lg text-muted-foreground mb-2">
            Professional video editing with AI-powered scene selection
          </p>
          <div className="flex items-center justify-center space-x-4 text-sm text-blue-600">
            <span>⚡ FFmpeg-powered processing</span>
            <span>🤖 AI character detection</span>
            <span>🎯 4x faster performance</span>
          </div>
        </div>

        {/* Main Content */}
        <div className="max-w-4xl mx-auto space-y-6">
          {/* Upload Section */}
          <div className="grid grid-cols-1 gap-6">
            <VideoUpload 
              onVideosUploaded={handleVideosUploaded}
              onScriptDataReady={handleScriptDataReady}
            />
          </div>

          {/* Status Section */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileText className="h-5 w-5" />
                Status: {status} | Videos: {videos.length} | Script: {scriptData ? 'Ready' : 'Not Ready'}
              </CardTitle>
            </CardHeader>
            <CardContent>
              {status === 'ready' && (
                <div className="space-y-4">
                  <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <CheckCircle className="h-5 w-5 text-green-600" />
                      <span className="font-medium text-green-800">Ready to start AI editing!</span>
                    </div>
                    <div className="text-sm text-green-700 space-y-1">
                      <p>• Characters: {scriptData?.characters}</p>
                      <p>• Target Duration: {scriptData?.audioDuration} minutes</p>
                      <p>• Script Content: {scriptData?.scriptContent.length || 0} characters</p>
                    </div>
                  </div>
                  <Button onClick={startEditing} className="w-full">
                    <Play className="h-4 w-4 mr-2" />
                    Start AI Video Editing
                  </Button>
                </div>
              )}

              {status === 'idle' && (
                <div className="space-y-4">
                  <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                    <p className="text-yellow-800">Waiting for video upload and script data...</p>
                    <div className="text-sm text-yellow-700 mt-2 space-y-1">
                      <p>• Videos: {videos.length} uploaded</p>
                      <p>• Script Data: {scriptData ? 'Ready' : 'Not provided'}</p>
                    </div>
                  </div>
                </div>
              )}

              {status === 'uploading' && (
                <div className="space-y-4">
                  <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                    <p className="text-blue-800">Uploading videos...</p>
                    <Progress value={progress} className="mt-2" />
                  </div>
                </div>
              )}

              {status === 'processing' && (
                <div className="space-y-4">
                  <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <Play className="h-5 w-5 text-blue-500 animate-pulse" />
                      <span className="font-medium text-blue-800">AI Processing Video</span>
                    </div>
                    
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>Processing...</span>
                        <span>{progress}%</span>
                      </div>
                      <Progress value={progress} />
                    </div>
                  </div>
                </div>
              )}

              {status === 'completed' && (
                <div className="space-y-4">
                  <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <CheckCircle className="h-5 w-5 text-green-600" />
                      <span className="font-medium text-green-800">Video editing completed!</span>
                    </div>
                    <Button onClick={downloadVideo} className="w-full">
                      <Download className="h-4 w-4 mr-2" />
                      Download Final Video
                    </Button>
                  </div>
                </div>
              )}

              {status === 'error' && (
                <div className="space-y-4">
                  <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                    <p className="text-red-800">Error: {error}</p>
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
