'use client'

import React, { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { Card } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { useToast } from '@/hooks/use-toast'
import { useWebSocket } from '../hooks/useWebSocket'
import SceneAnalysis from './SceneAnalysis'
import {
  PlayCircle,
  Scissors,
  Upload as UploadIcon,
  Video,
  Users,
  Star
} from 'lucide-react'

interface VideoProcessorProps {
  onComplete?: (result: any) => void
  onError?: (error: string) => void
}

interface ProcessingState {
  status: 'idle' | 'uploading' | 'processing' | 'completed' | 'error'
  progress: number
  sessionId?: string
  error?: string
  result?: any
}

interface Scene {
  id: string
  start_time: number
  end_time: number
  duration: number
  faces: Array<{
    id: string
    character?: string
    confidence?: number
  }>
  quality_score: number
  match_score?: number
  detected_actions: string[]
}

export const VideoProcessor: React.FC<VideoProcessorProps> = ({
  onComplete,
  onError
}) => {
  const [state, setState] = useState<ProcessingState>({
    status: 'idle',
    progress: 0
  })

  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [scenes, setScenes] = useState<Scene[]>([])
  const [selectedSceneId, setSelectedSceneId] = useState<string | undefined>(undefined)
  const [activeTab, setActiveTab] = useState('upload')

  const { toast } = useToast()
  const { connect, disconnect, sendMessage } = useWebSocket({
    onMessage: handleWebSocketMessage
  })

  useEffect(() => {
    return () => {
      disconnect()
    }
  }, [disconnect])

  function handleWebSocketMessage(event: MessageEvent) {
    try {
      const message = JSON.parse(event.data)
      
      switch (message.type) {
        case 'progress':
          setState(prev => ({
            ...prev,
            progress: message.progress,
            status: message.status === 'completed' ? 'completed' : 'processing'
          }))
          break
          
        case 'scenes':
          setScenes(message.scenes)
          setActiveTab('analysis')
          break
          
        case 'error':
          setState(prev => ({
            ...prev,
            status: 'error',
            error: message.error
          }))
          onError?.(message.error)
          break
          
        case 'completion':
          setState(prev => ({
            ...prev,
            status: 'completed',
            result: message.result
          }))
          onComplete?.(message.result)
          break
      }
    } catch (error) {
      console.error('Failed to parse WebSocket message:', error)
    }
  }

  const handleUpload = async (file: File) => {
    try {
      setState(prev => ({
        ...prev,
        status: 'uploading',
        progress: 0
      }))

      const formData = new FormData()
      formData.append('file', file)
      
      const response = await fetch('/api/video/upload', {
        method: 'POST',
        body: formData
      })
      
      if (!response.ok) {
        throw new Error('Upload failed')
      }
      
      const data = await response.json()
      const { session_id } = data
      
      setState(prev => ({
        ...prev,
        sessionId: session_id,
        status: 'idle'
      }))
      
      setSelectedFile(file)
              connect(`ws://localhost:8000/api/video/ws/${session_id}`)
      
    } catch (error) {
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : 'Upload failed',
        variant: "destructive"
      })
    }
  }

  const startProcessing = async () => {
    if (!state.sessionId) return
    
    try {
      setState(prev => ({
        ...prev,
        status: 'processing',
        progress: 0
      }))
      
      const response = await fetch(`/api/video/process/${state.sessionId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      })
      
      if (!response.ok) {
        throw new Error('Processing failed')
      }
      
    } catch (error) {
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : 'Processing failed',
        variant: "destructive"
      })
    }
  }

  const handleSceneSelect = (scene: Scene) => {
    setSelectedSceneId(scene.id)
  }

  return (
    <div className="space-y-6">
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="upload" disabled={state.status === 'processing'}>
            <Video className="h-4 w-4 mr-2" />
            Upload
          </TabsTrigger>
          <TabsTrigger
            value="analysis"
            disabled={!scenes.length || state.status === 'processing'}
          >
            <Star className="h-4 w-4 mr-2" />
            Analysis
          </TabsTrigger>
        </TabsList>

        <TabsContent value="upload">
          <Card className="p-6">
            <div className="space-y-6">
              {/* Upload Section */}
              <div>
                <div className="relative">
                  <input
                    type="file"
                    accept="video/*"
                    className="hidden"
                    onChange={(e) => {
                      const file = e.target.files?.[0]
                      if (file) {
                        handleUpload(file)
                      }
                    }}
                    id="video-upload"
                  />
                  <Button
                    variant="outline"
                    className="w-full"
                    onClick={() => document.getElementById('video-upload')?.click()}
                    disabled={state.status === 'processing'}
                  >
                    <UploadIcon className="mr-2 h-4 w-4" />
                    {selectedFile ? selectedFile.name : 'Select Video'}
                  </Button>
                </div>
              </div>

              {/* Status and Progress */}
              {state.status !== 'idle' && (
                <Progress value={state.progress} />
              )}

              {/* Error Message */}
              {state.error && (
                <div className="rounded-md bg-destructive/15 p-3 text-sm text-destructive">
                  {state.error}
                </div>
              )}

              {/* Controls */}
              <div className="flex space-x-4">
                <Button
                  onClick={startProcessing}
                  disabled={!state.sessionId || state.status === 'processing'}
                  className="flex-1"
                >
                  <PlayCircle className="mr-2 h-4 w-4" />
                  Process Video
                </Button>
              </div>
            </div>
          </Card>
        </TabsContent>

        <TabsContent value="analysis">
          <SceneAnalysis
            scenes={scenes}
            onSceneSelect={handleSceneSelect}
            selectedSceneId={selectedSceneId}
          />
        </TabsContent>
      </Tabs>
    </div>
  )
} 