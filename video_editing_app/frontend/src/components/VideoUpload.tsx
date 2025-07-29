'use client'

import { useState, useCallback, useEffect } from 'react'
import { useDropzone } from 'react-dropzone'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { Upload, Video, X, CheckCircle, FileText, Clock } from 'lucide-react'

interface VideoUploadProps {
  onVideosUploaded: (videos: File[]) => void
  onScriptDataReady: (scriptData: { characters: string, scriptContent: string, audioDuration: number }) => void
}

export function VideoUpload({ onVideosUploaded, onScriptDataReady }: VideoUploadProps) {
  
  const [videos, setVideos] = useState<File[]>([])
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState('')
  
  // Script-based editing inputs
  const [characters, setCharacters] = useState<string>('Jean-Claude Van Damme, Steven Segal')
  const [scriptContent, setScriptContent] = useState<string>('')
  const [audioDuration, setAudioDuration] = useState<number>(22.0)

  // CRITICAL FIX: Trigger script data callback immediately when videos are selected
  useEffect(() => {
    if (videos.length > 0) {
      console.log('📝 Triggering script data callback immediately with videos selected')
      onScriptDataReady({
        characters: characters,
        scriptContent: scriptContent,
        audioDuration: audioDuration
      })
    }
  }, [videos, characters, scriptContent, audioDuration]) // REMOVED onScriptDataReady from dependencies

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
    setProgress(0)
    console.log('📤 Setting uploading state to true')

    try {
      const formData = new FormData()
      videos.forEach(video => {
        formData.append('files', video)
        console.log('📁 Adding video to formData:', video.name, video.size)
      })
      
      console.log('📡 Making request to backend...')
      const response = await fetch('http://localhost:8001/upload-videos', {
        method: 'POST',
        body: formData
      })

      console.log('📥 Response received:', response.status, response.statusText)
      
      if (!response.ok) {
        const errorText = await response.text()
        console.error('❌ Response not OK:', errorText)
        throw new Error(`Upload failed: ${response.status} - ${errorText}`)
      }

      const data = await response.json()
      console.log('✅ Upload successful, data:', data)

      console.log('📞 Calling onVideosUploaded with:', videos.length, 'videos')
      onVideosUploaded(videos)
      
      // REMOVED: No longer need to call onScriptDataReady here since it's handled by useEffect
      
      setProgress(100)
      setUploading(false)
      console.log('✅ Upload process completed')

    } catch (err) {
      console.error('💥 Upload error:', err)
      setError(err instanceof Error ? err.message : 'Upload failed')
      setUploading(false)
      setProgress(0)
    }
  }

  const handleButtonClick = () => {
    console.log('🔘 Upload button clicked!')
    console.log('📁 Videos selected:', videos.length)
    console.log('📝 Script data:', { characters, scriptContent: scriptContent.length, audioDuration })
    uploadVideos()
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Video className="h-5 w-5" />
          Script-Based Video Editor
        </CardTitle>
        <CardDescription>
          Upload videos and provide script content for AI-driven editing
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        
        {/* Character Input */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            <FileText className="h-4 w-4 inline mr-1" />
            Main Characters (comma-separated)
          </label>
          <input
            type="text"
            value={characters}
            onChange={(e) => setCharacters(e.target.value)}
            placeholder="Jean-Claude Van Damme, Steven Segal"
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm p-2"
          />
        </div>

        {/* Script Content Input */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            <FileText className="h-4 w-4 inline mr-1" />
            Script Content (Optional)
          </label>
          <textarea
            value={scriptContent}
            onChange={(e) => setScriptContent(e.target.value)}
            placeholder="Enter your script content here... The AI will analyze this to determine character screen time and scene selection."
            rows={4}
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm p-2"
          />
          <p className="text-xs text-gray-500 mt-1">
            Leave empty to use equal character distribution
          </p>
        </div>

        {/* Audio Duration Input */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            <Clock className="h-4 w-4 inline mr-1" />
            Target Video Duration (minutes)
          </label>
          <input
            type="number"
            value={audioDuration}
            onChange={(e) => setAudioDuration(parseFloat(e.target.value) || 22.0)}
            min="1"
            max="120"
            step="0.5"
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm p-2"
          />
          <p className="text-xs text-gray-500 mt-1">
            Target duration for the final video (default: 22 minutes)
          </p>
        </div>
        
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
            Upload {videos.length} Video{videos.length > 1 ? 's' : ''} & Start Editing
          </Button>
        )}
      </CardContent>
    </Card>
  )
}