'use client'

import React, { useEffect, useState } from 'react'
import { useVideoStore } from '@/stores/videoStore'
import { Progress } from '@/components/ui/progress'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Badge } from '@/components/ui/badge'
import {
  CheckCircle2,
  Loader2,
  AlertCircle,
  Clock,
  Activity,
  Users,
  Video,
  FileText,
  Wand2
} from 'lucide-react'
import toast from 'react-hot-toast'

interface ProcessingStatusProps {
  onComplete: () => void
}

interface ProcessingStep {
  id: string
  title: string
  description: string
  status: 'pending' | 'processing' | 'completed' | 'error'
  progress: number
  details?: string[]
  startTime?: number
  endTime?: number
  error?: string
}

export default function ProcessingStatus({ onComplete }: ProcessingStatusProps) {
  const [steps, setSteps] = useState<ProcessingStep[]>([
    {
      id: 'video_analysis',
      title: 'Video Analysis',
      description: 'Analyzing video content and detecting scenes',
      status: 'pending',
      progress: 0
    },
    {
      id: 'face_detection',
      title: 'Face Detection',
      description: 'Detecting and identifying faces in scenes',
      status: 'pending',
      progress: 0
    },
    {
      id: 'scene_selection',
      title: 'Scene Selection',
      description: 'Selecting best scenes based on script',
      status: 'pending',
      progress: 0
    },
    {
      id: 'final_assembly',
      title: 'Final Assembly',
      description: 'Assembling selected scenes into final video',
      status: 'pending',
      progress: 0
    }
  ])

  const { sessionId } = useVideoStore()

  useEffect(() => {
    if (!sessionId) return

    const ws = new WebSocket(`ws://localhost:8000/api/ws/${sessionId}`)

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data)
      
      switch (message.type) {
        case 'progress':
          updateStepProgress(message)
          break
        case 'completed':
          handleCompletion()
          break
        case 'error':
          handleError(message.error)
          break
      }
    }

    return () => ws.close()
  }, [sessionId])

  const updateStepProgress = (message: any) => {
    const { phase, progress, details } = message
    
    setSteps(prevSteps => 
      prevSteps.map(step => {
        if (step.id === phase) {
          return {
            ...step,
            status: 'processing',
            progress,
            details: details || step.details,
            startTime: step.startTime || Date.now()
          }
        } else if (progress === 100) {
          return {
            ...step,
            status: step.status === 'pending' ? 'pending' : 'completed',
            endTime: step.endTime || Date.now()
          }
        }
        return step
      })
    )
  }

  const handleCompletion = () => {
    setSteps(prevSteps =>
      prevSteps.map(step => ({
        ...step,
        status: 'completed',
        progress: 100,
        endTime: step.endTime || Date.now()
      }))
    )
    onComplete()
  }

  const handleError = (error: string) => {
    setSteps(prevSteps =>
      prevSteps.map(step => {
        if (step.status === 'processing') {
          return {
            ...step,
            status: 'error',
            error,
            endTime: Date.now()
          }
        }
        return step
      })
    )
  }

  const getStepIcon = (step: ProcessingStep) => {
    switch (step.id) {
      case 'video_analysis':
        return <Video className="h-5 w-5" />
      case 'face_detection':
        return <Users className="h-5 w-5" />
      case 'scene_selection':
        return <FileText className="h-5 w-5" />
      case 'final_assembly':
        return <Wand2 className="h-5 w-5" />
      default:
        return <Activity className="h-5 w-5" />
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle2 className="h-5 w-5 text-green-500" />
      case 'processing':
        return <Loader2 className="h-5 w-5 text-blue-500 animate-spin" />
      case 'error':
        return <AlertCircle className="h-5 w-5 text-red-500" />
      default:
        return <Clock className="h-5 w-5 text-gray-400" />
    }
  }

  const formatDuration = (start?: number, end?: number) => {
    if (!start || !end) return ''
    const duration = Math.round((end - start) / 1000)
    return `${duration}s`
  }

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <div className="text-center">
        <h2 className="text-3xl font-bold mb-2">Processing Your Video</h2>
        <p className="text-gray-600">
          Please wait while we process your video. This may take several minutes.
        </p>
      </div>

      <div className="grid gap-6">
        {steps.map((step) => (
          <Card key={step.id}>
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {getStepIcon(step)}
                  <span>{step.title}</span>
                </div>
                <div className="flex items-center gap-4">
                  {step.startTime && (
                    <Badge variant="outline" className="text-xs">
                      <Clock className="h-3 w-3 mr-1" />
                      {formatDuration(step.startTime, step.endTime || Date.now())}
                    </Badge>
                  )}
                  {getStatusIcon(step.status)}
                </div>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>{step.description}</span>
                    <span>{step.progress}%</span>
                  </div>
                  <Progress value={step.progress} className="h-2" />
                </div>

                {step.details && step.details.length > 0 && (
                  <ScrollArea className="h-24 rounded-md border p-2">
                    <div className="space-y-1">
                      {step.details.map((detail, index) => (
                        <div
                          key={index}
                          className="text-sm text-gray-600 flex items-center gap-2"
                        >
                          <Activity className="h-3 w-3" />
                          {detail}
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                )}

                {step.error && (
                  <div className="text-sm text-red-500 flex items-center gap-2">
                    <AlertCircle className="h-4 w-4" />
                    {step.error}
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
} 