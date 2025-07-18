'use client'

import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { Upload, Video, X, CheckCircle } from 'lucide-react'

interface VideoUploadProps {
  onVideosUploaded: (videos: File[]) => void
}

export function VideoUpload({ onVideosUploaded }: VideoUploadProps) {
  console.log('🎬 VideoUpload rendered with props:', { 
    onVideosUploaded: !!onVideosUploaded, 
    // Remove: onSessionCreated: !!onSessionCreated 
  })
  const [videos, setVideos] = useState<File[]>([])
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState('')

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const videoFiles = acceptedFiles.filter(file => 
      file.type.startsWith('video/') || 
      ['.mp4', '.avi', '.mov', '.mkv'].some(ext => file.name.toLowerCase().endsWith(ext))
    )
    
    if (videoFiles.length === 0) {
      setError('No valid video files selected')
      return
    }

    if (videos.length + videoFiles.length > 3) {
      setError('Maximum 3 videos allowed')
      return
    }

    setVideos(prev => [...prev, ...videoFiles])
    setError('')
  }, [videos])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.avi', '.mov', '.mkv']
    },
    maxFiles: 3
  })

  const removeVideo = (index: number) => {
    setVideos(prev => prev.filter((_, i) => i !== index))
  }

  const uploadVideos = async () => {
    console.log('🚀 uploadVideos function started')
    
    if (videos.length === 0) {
      console.log('❌ No videos to upload')
      return
    }

    setUploading(true)
    console.log('📤 Setting uploading state to true')

    try {
      const formData = new FormData()
      videos.forEach(video => {
        formData.append('files', video)
      })
      
      const response = await fetch('http://localhost:8001/upload-videos', {
        method: 'POST',
        body: formData
      })

      console.log('📥 Response received:', response.status)
      
      if (!response.ok) {
        const errorText = await response.text()
        console.error('❌ Response not OK:', errorText)
        throw new Error(`Upload failed: ${response.status}`)
      }

      const data = await response.json()
      console.log('✅ Upload successful, data:', data)

      // Remove these lines:
      // console.log('📞 Calling onSessionCreated with:', sessionId)
      // onSessionCreated(sessionId)
      
      console.log('📞 Calling onVideosUploaded with:', videos.length, 'videos')
      onVideosUploaded(videos)
      
      setProgress(100)
      setUploading(false)
      console.log('✅ Upload process completed')

    } catch (err) {
      console.error('💥 Upload error:', err)
      setError(err instanceof Error ? err.message : 'Upload failed')
      setUploading(false)
    }
  }

  const handleButtonClick = () => {
    console.log('🔘 Upload button clicked!')
    console.log('📁 Videos selected:', videos.length)
    uploadVideos()
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Video className="h-5 w-5" />
          Upload Videos
        </CardTitle>
        <CardDescription>
          Upload 2-3 video files (MP4, AVI, MOV, MKV)
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Drop Zone */}
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
            isDragActive 
              ? 'border-blue-500 bg-blue-50' 
              : 'border-slate-300 hover:border-slate-400'
          }`}
        >
          <input {...getInputProps()} />
          <Upload className="h-12 w-12 mx-auto mb-4 text-slate-400" />
          <p className="text-slate-600">
            {isDragActive 
              ? 'Drop videos here...' 
              : 'Drag & drop videos here, or click to select'
            }
          </p>
        </div>

        {/* Video List */}
        {videos.length > 0 && (
          <div className="space-y-2">
            <h4 className="font-medium">Selected Videos:</h4>
            {videos.map((video, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-slate-50 rounded-lg">
                <div className="flex items-center gap-2">
                  <Video className="h-4 w-4 text-slate-500" />
                  <span className="text-sm">{video.name}</span>
                  <span className="text-xs text-slate-500">
                    ({(video.size / 1024 / 1024).toFixed(1)} MB)
                  </span>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => removeVideo(index)}
                  className="text-red-500 hover:text-red-700"
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
            ))}
          </div>
        )}

        {/* Upload Progress */}
        {uploading && (
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Uploading...</span>
              <span>{progress}%</span>
            </div>
            <Progress value={progress} />
          </div>
        )}

        {/* Error */}
        {error && (
          <p className="text-red-600 text-sm">{error}</p>
        )}

        {/* Upload Button */}
        {videos.length > 0 && !uploading && (
          <Button onClick={handleButtonClick} className="w-full">
            <Upload className="h-4 w-4 mr-2" />
            Upload {videos.length} Video{videos.length > 1 ? 's' : ''}
          </Button>
        )}
      </CardContent>
    </Card>
  )
}