'use client'

import React from 'react'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle
} from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import {
  Users,
  Star,
  Clock,
  Activity,
  CheckCircle,
  AlertCircle
} from 'lucide-react'

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

interface SceneAnalysisProps {
  scenes: Scene[]
  onSceneSelect?: (scene: Scene) => void
  selectedSceneId?: string
}

export default function SceneAnalysis({
  scenes,
  onSceneSelect,
  selectedSceneId
}: SceneAnalysisProps) {
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const getQualityColor = (score: number) => {
    if (score >= 0.8) return 'text-green-500'
    if (score >= 0.6) return 'text-yellow-500'
    return 'text-red-500'
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Activity className="h-5 w-5" />
          Scene Analysis Results
        </CardTitle>
        <CardDescription>
          Detected {scenes.length} scenes with face recognition and quality scoring
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-[500px] pr-4">
          <div className="space-y-4">
            {scenes.map((scene) => (
              <Card
                key={scene.id}
                className={`cursor-pointer transition-colors hover:bg-accent ${
                  selectedSceneId === scene.id ? 'border-primary' : ''
                }`}
                onClick={() => onSceneSelect?.(scene)}
              >
                <CardContent className="p-4">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <Clock className="h-4 w-4" />
                      <span>
                        {formatTime(scene.start_time)} - {formatTime(scene.end_time)}
                      </span>
                      <Badge variant="outline">
                        {scene.duration.toFixed(1)}s
                      </Badge>
                    </div>
                    <div className="flex items-center gap-2">
                      <Star className={`h-4 w-4 ${getQualityColor(scene.quality_score)}`} />
                      <span className={getQualityColor(scene.quality_score)}>
                        {(scene.quality_score * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>

                  {/* Face Detection Results */}
                  <div className="mb-2">
                    <div className="flex items-center gap-2 mb-1">
                      <Users className="h-4 w-4" />
                      <span className="font-medium">Detected Characters</span>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {scene.faces.map((face) => (
                        <Badge
                          key={face.id}
                          variant={face.character ? 'default' : 'secondary'}
                        >
                          {face.character || 'Unknown'}
                          {face.confidence && ` (${(face.confidence * 100).toFixed(0)}%)`}
                        </Badge>
                      ))}
                    </div>
                  </div>

                  {/* Actions */}
                  {scene.detected_actions.length > 0 && (
                    <div className="mb-2">
                      <div className="flex items-center gap-2 mb-1">
                        <Activity className="h-4 w-4" />
                        <span className="font-medium">Detected Actions</span>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {scene.detected_actions.map((action, index) => (
                          <Badge key={index} variant="outline">
                            {action}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Quality Metrics */}
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span>Face Detection Quality</span>
                      <span>{(scene.quality_score * 100).toFixed(0)}%</span>
                    </div>
                    <Progress
                      value={scene.quality_score * 100}
                      className="h-2"
                    />
                    {scene.match_score && (
                      <>
                        <div className="flex items-center justify-between text-sm">
                          <span>Script Match Score</span>
                          <span>{(scene.match_score * 100).toFixed(0)}%</span>
                        </div>
                        <Progress
                          value={scene.match_score * 100}
                          className="h-2"
                        />
                      </>
                    )}
                  </div>

                  {/* Selection Status */}
                  <div className="flex items-center justify-end mt-2">
                    {selectedSceneId === scene.id ? (
                      <Badge className="bg-green-500">
                        <CheckCircle className="h-3 w-3 mr-1" />
                        Selected
                      </Badge>
                    ) : scene.match_score && scene.match_score > 0.7 ? (
                      <Badge variant="outline" className="text-green-500">
                        <CheckCircle className="h-3 w-3 mr-1" />
                        Recommended
                      </Badge>
                    ) : (
                      <Badge variant="outline" className="text-yellow-500">
                        <AlertCircle className="h-3 w-3 mr-1" />
                        Alternative
                      </Badge>
                    )}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  )
} 