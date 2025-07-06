'use client'

import React, { useCallback, useState } from 'react'
import { useVideoStore } from '@/stores/videoStore'
import { useTestModeStore } from '@/stores/testModeStore'
import { Upload, X, Film, AlertCircle, CheckCircle2, ArrowRight } from 'lucide-react'
import { useDropzone } from 'react-dropzone'
import toast from 'react-hot-toast'

interface VideoFile extends File {
  preview: string
  progress: number
}

export default function VideoUpload() {
  const { sessionId, setCurrentStep } = useVideoStore()
  const { testMode } = useTestModeStore()
  
  const [uploadedFiles, setUploadedFiles] = useState<VideoFile[]>([])
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState<{[key: string]: number}>({})

  const MAX_FILES = 3
  const MAX_FILE_SIZE = 400 * 1024 * 1024 // 400MB
  const ALLOWED_TYPES = ['video/mp4', 'video/quicktime', 'video/x-msvideo']

  const validateFile = (file: File) => {
    if (!ALLOWED_TYPES.includes(file.type)) {
      throw new Error('Invalid file type. Please upload MP4, MOV, or AVI files.')
    }
    if (file.size > MAX_FILE_SIZE) {
      throw new Error('File too large. Maximum size is 400MB.')
    }
  }

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (uploadedFiles.length + acceptedFiles.length > MAX_FILES) {
      toast.error(`Maximum ${MAX_FILES} videos allowed`)
      return
    }

    const validFiles = acceptedFiles.map(file => {
      try {
        validateFile(file)
        return {
          ...file,
          preview: URL.createObjectURL(file),
          progress: 0
        } as VideoFile
      } catch (error) {
        toast.error(error instanceof Error ? error.message : 'Invalid file')
        return null
      }
    }).filter((file): file is VideoFile => file !== null)

    setUploadedFiles(prev => [...prev, ...validFiles])
  }, [uploadedFiles])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.mov', '.avi']
    },
    maxFiles: MAX_FILES,
    disabled: isUploading
  })

  const removeFile = (index: number) => {
    setUploadedFiles(prev => {
      const newFiles = [...prev]
      URL.revokeObjectURL(newFiles[index].preview || '')
      newFiles.splice(index, 1)
      return newFiles
    })
  }

  const uploadFiles = async () => {
    if (!sessionId) {
      toast.error('Session not found')
      return
    }

    if (uploadedFiles.length === 0) {
      toast.error('Please upload at least one video')
      return
    }

    setIsUploading(true)

    try {
      // Upload each file with progress tracking
      const uploads = uploadedFiles.map(async (file, index) => {
        const formData = new FormData()
        formData.append('video', file)
        formData.append('session_id', sessionId)
        formData.append('video_index', index.toString())

        const xhr = new XMLHttpRequest()
        
        // Track upload progress
        xhr.upload.onprogress = (event) => {
          if (event.lengthComputable) {
            const progress = (event.loaded / event.total) * 100
            setUploadProgress(prev => ({
              ...prev,
              [file.name]: progress
            }))
          }
        }

        // Return a promise for the upload
        return new Promise((resolve, reject) => {
          xhr.open('POST', '/api/video/upload')
          
          xhr.onload = () => {
            if (xhr.status === 200) {
              resolve(JSON.parse(xhr.response))
            } else {
              reject(new Error(xhr.statusText))
            }
          }
          
          xhr.onerror = () => reject(new Error('Network error'))
          xhr.send(formData)
        })
      })

      await Promise.all(uploads)
      
      toast.success('Videos uploaded successfully!')
      setCurrentStep('processing')
      
    } catch (error) {
      console.error('Upload error:', error)
      toast.error('Failed to upload videos')
    } finally {
      setIsUploading(false)
    }
  }

  return (
    <div className="max-w-4xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h2 className="text-3xl font-bold text-gray-900 mb-2">Upload Videos</h2>
        <p className="text-gray-600">
          Upload up to {MAX_FILES} videos for processing. Maximum 400MB per file.
        </p>
      </div>

      {/* Upload Area */}
      <div className="space-y-6">
        {/* Dropzone */}
        <div 
          {...getRootProps()} 
          className={`
            card cursor-pointer border-2 border-dashed
            ${isDragActive ? 'border-primary-500 bg-primary-50' : 'border-gray-300'}
            ${isUploading ? 'opacity-50 cursor-not-allowed' : ''}
          `}
        >
          <input {...getInputProps()} />
          <div className="text-center py-8">
            <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            {isDragActive ? (
              <p className="text-lg text-primary-600">Drop your videos here...</p>
            ) : (
              <div>
                <p className="text-lg text-gray-600 mb-2">
                  Drag & drop your videos here, or click to select
                </p>
                <p className="text-sm text-gray-500">
                  MP4, MOV, or AVI files up to 400MB
                </p>
              </div>
            )}
          </div>
        </div>

        {/* File List */}
        {uploadedFiles.length > 0 && (
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-gray-900">
              Uploaded Videos ({uploadedFiles.length}/{MAX_FILES})
            </h3>
            
            {uploadedFiles.map((file, index) => (
              <div key={file.name} className="card flex items-center space-x-4">
                <div className="flex-shrink-0">
                  <Film className="w-8 h-8 text-gray-400" />
                </div>
                
                <div className="flex-grow min-w-0">
                  <div className="flex items-center justify-between mb-1">
                    <p className="text-sm font-medium text-gray-900 truncate">
                      {file.name}
                    </p>
                    {!isUploading && (
                      <button 
                        onClick={() => removeFile(index)}
                        className="text-gray-400 hover:text-gray-600"
                      >
                        <X className="w-5 h-5" />
                      </button>
                    )}
                  </div>
                  
                  <p className="text-xs text-gray-500 mb-2">
                    {(file.size / (1024 * 1024)).toFixed(1)}MB
                  </p>
                  
                  {isUploading && (
                    <div className="w-full bg-gray-200 rounded-full h-1.5">
                      <div 
                        className="bg-primary-600 h-1.5 rounded-full transition-all duration-300"
                        style={{ width: `${uploadProgress[file.name] || 0}%` }}
                      />
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Test Mode Warning */}
        {testMode && (
          <div className="card bg-warning-50 border-warning-200">
            <div className="flex items-start">
              <AlertCircle className="w-5 h-5 text-warning-600 mt-0.5 mr-3" />
              <div>
                <h4 className="text-sm font-medium text-warning-800">
                  Test Mode Active
                </h4>
                <p className="text-sm text-warning-700 mt-1">
                  Videos will be processed with test settings and saved resources.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex items-center justify-between pt-4">
          <button
            onClick={() => setCurrentStep('script')}
            className="btn-secondary"
          >
            Back to Script
          </button>

          <button
            onClick={uploadFiles}
            disabled={isUploading || uploadedFiles.length === 0}
            className="btn-primary flex items-center"
          >
            {isUploading ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                Uploading...
              </>
            ) : (
              <>
                <ArrowRight className="w-4 h-4 mr-2" />
                Start Processing
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  )
} 