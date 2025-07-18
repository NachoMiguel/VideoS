'use client'

import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { Upload, Music, X, CheckCircle } from 'lucide-react'

interface AudioUploadProps {
  onAudioUploaded: (audio: File) => void
}

export function AudioUpload({ onAudioUploaded }: AudioUploadProps) {
  const [audioFile, setAudioFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState('')

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const audioFiles = acceptedFiles.filter(file => 
      file.type.startsWith('audio/') || 
      ['.mp3', '.wav', '.ogg'].some(ext => file.name.toLowerCase().endsWith(ext))
    )
    
    if (audioFiles.length === 0) {
      setError('No valid audio file selected')
      return
    }

    setAudioFile(audioFiles[0])
    setError('')
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'audio/*': ['.mp3', '.wav', '.ogg']
    },
    maxFiles: 1
  })

  const removeAudio = () => {
    setAudioFile(null)
  }

  const uploadAudio = async () => {
    if (!audioFile) return

    setUploading(true)
    setProgress(0)
    setError('')

    try {
      // Upload audio file
      const formData = new FormData()
      formData.append('file', audioFile)
      formData.append('session_id', crypto.randomUUID())

      const response = await fetch('http://localhost:8001/upload-audio', {
        method: 'POST',
        body: formData
      })

      if (!response.ok) throw new Error('Upload failed')

      onAudioUploaded(audioFile)
      setProgress(100)

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed')
    } finally {
      setUploading(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Music className="h-5 w-5" />
          Upload Audio
        </CardTitle>
        <CardDescription>
          Upload your audio file (MP3, WAV, OGG)
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
              ? 'Drop audio here...' 
              : 'Drag & drop audio here, or click to select'
            }
          </p>
        </div>

        {/* Audio File */}
        {audioFile && (
          <div className="space-y-2">
            <h4 className="font-medium">Selected Audio:</h4>
            <div className="flex items-center justify-between p-3 bg-slate-50 rounded-lg">
              <div className="flex items-center gap-2">
                <Music className="h-4 w-4 text-slate-500" />
                <span className="text-sm">{audioFile.name}</span>
                <span className="text-xs text-slate-500">
                  ({(audioFile.size / 1024 / 1024).toFixed(1)} MB)
                </span>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={removeAudio}
                className="text-red-500 hover:text-red-700"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
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
        {audioFile && !uploading && (
          <Button onClick={uploadAudio} className="w-full">
            <Upload className="h-4 w-4 mr-2" />
            Upload Audio
          </Button>
        )}
      </CardContent>
    </Card>
  )
}