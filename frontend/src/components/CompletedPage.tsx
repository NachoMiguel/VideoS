'use client'

import React, { useState } from 'react'
import { useVideoStore } from '@/stores/videoStore'
import { useTestModeStore } from '@/stores/testModeStore'
import { Download, RefreshCw, Share2, CheckCircle2, AlertCircle } from 'lucide-react'
import toast from 'react-hot-toast'

interface VideoStats {
  duration: string
  size: string
  resolution: string
  scenes: number
  characters: number
}

interface CompletedPageProps {
  onRestart?: () => void
}

export default function CompletedPage({ onRestart }: CompletedPageProps) {
  const { sessionId, setCurrentStep } = useVideoStore()
  const { testMode } = useTestModeStore()
  
  const [isDownloading, setIsDownloading] = useState(false)
  const [downloadProgress, setDownloadProgress] = useState(0)
  const [stats, setStats] = useState<VideoStats>({
    duration: '3:45',
    size: '256MB',
    resolution: '1080p',
    scenes: 12,
    characters: 3
  })

  const handleDownload = async () => {
    if (!sessionId) {
      toast.error('Session not found')
      return
    }

    setIsDownloading(true)
    setDownloadProgress(0)

    try {
      const response = await fetch(`/api/v1/download/${sessionId}`)
      
      if (!response.ok) {
        throw new Error('Failed to download video')
      }

      const contentLength = response.headers.get('content-length')
      const total = parseInt(contentLength || '0', 10)
      let loaded = 0

      const reader = response.body?.getReader()
      if (!reader) {
        throw new Error('Failed to initialize download')
      }

      // Create a new ReadableStream and pipe the response to it
      const stream = new ReadableStream({
        async start(controller) {
          while (true) {
            const { done, value } = await reader.read()
            
            if (done) {
              controller.close()
              break
            }

            loaded += value.length
            setDownloadProgress(Math.round((loaded / total) * 100))
            controller.enqueue(value)
          }
        }
      })

      // Create a blob from the stream
      const blob = await new Response(stream).blob()
      const url = window.URL.createObjectURL(blob)
      
      // Create a temporary link and trigger download
      const a = document.createElement('a')
      a.href = url
      a.download = `ai-video-${sessionId}.mp4`
      document.body.appendChild(a)
      a.click()
      
      // Cleanup
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
      
      toast.success('Download completed!')
    } catch (error) {
      console.error('Download error:', error)
      toast.error('Failed to download video')
    } finally {
      setIsDownloading(false)
      setDownloadProgress(0)
    }
  }

  const handleShare = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href)
      toast.success('Link copied to clipboard!')
    } catch (error) {
      console.error('Share error:', error)
      toast.error('Failed to copy link')
    }
  }

  return (
    <div className="max-w-4xl mx-auto">
      {/* Success Header */}
      <div className="text-center mb-12">
        <div className="inline-flex items-center justify-center w-20 h-20 bg-success-100 rounded-full mb-6">
          <CheckCircle2 className="w-10 h-10 text-success-600" />
        </div>
        <h2 className="text-3xl font-bold text-gray-900 mb-2">
          Video Processing Complete!
        </h2>
        <p className="text-lg text-gray-600">
          Your personalized video is ready to download
        </p>
      </div>

      {/* Video Preview */}
      <div className="card mb-8">
        <div className="aspect-video bg-gray-100 rounded-lg mb-6">
          {/* Video player will go here */}
          <div className="w-full h-full flex items-center justify-center">
            <p className="text-gray-500">Video Preview</p>
          </div>
        </div>

        {/* Video Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <p className="text-sm text-gray-500 mb-1">Duration</p>
            <p className="text-lg font-semibold">{stats.duration}</p>
          </div>
          <div>
            <p className="text-sm text-gray-500 mb-1">File Size</p>
            <p className="text-lg font-semibold">{stats.size}</p>
          </div>
          <div>
            <p className="text-sm text-gray-500 mb-1">Resolution</p>
            <p className="text-lg font-semibold">{stats.resolution}</p>
          </div>
          <div>
            <p className="text-sm text-gray-500 mb-1">Scenes</p>
            <p className="text-lg font-semibold">{stats.scenes}</p>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex flex-col md:flex-row items-stretch md:items-center gap-4">
        <button
          onClick={handleDownload}
          disabled={isDownloading}
          className="btn-primary flex-1 flex items-center justify-center"
        >
          {isDownloading ? (
            <>
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
              Downloading... {downloadProgress}%
            </>
          ) : (
            <>
              <Download className="w-4 h-4 mr-2" />
              Download Video
            </>
          )}
        </button>

        <button
          onClick={handleShare}
          className="btn-secondary flex-1 flex items-center justify-center"
        >
          <Share2 className="w-4 h-4 mr-2" />
          Share Link
        </button>

        <button
          onClick={() => setCurrentStep('landing')}
          className="btn-outline flex-1 flex items-center justify-center"
        >
          <RefreshCw className="w-4 h-4 mr-2" />
          Start New Video
        </button>
      </div>

      {/* Test Mode Notice */}
      {testMode && (
        <div className="card bg-warning-50 border-warning-200 mt-8">
          <div className="flex items-start">
            <AlertCircle className="w-5 h-5 text-warning-600 mt-0.5 mr-3" />
            <div>
              <h4 className="text-sm font-medium text-warning-800">
                Test Mode Result
              </h4>
              <p className="text-sm text-warning-700 mt-1">
                This video was processed using test mode settings and saved resources.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
} 