'use client'

import React, { useState, useEffect } from 'react'
import { useVideoStore } from '@/stores/videoStore'
import { useTestModeStore } from '@/stores/testModeStore'
import { Youtube, Settings, TestTube, Zap } from 'lucide-react'
import { useToast } from "@/hooks/use-toast"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Play, Video, Users, Star, CheckCircle, FileText } from 'lucide-react'

interface LandingPageProps {
  onStart?: () => void
}

export default function LandingPage({ onStart }: LandingPageProps) {
  const { 
    youtubeUrl, 
    useDefaultPrompt, 
    customPrompt,
    setYoutubeUrl, 
    setUseDefaultPrompt, 
    setCustomPrompt, 
    setCurrentStep,
    setSessionId,
    setScript
  } = useVideoStore()
  
  const { 
    testMode, 
    useSavedScript,
    setTestMode, 
    setUseSavedScript,
    loadTestResources,
    useKnownCharacters,
    setUseKnownCharacters
  } = useTestModeStore()

  const [isLoading, setIsLoading] = useState(false)
  const { toast } = useToast()

  // Load test resources when test mode is enabled
  useEffect(() => {
    if (testMode) {
      loadTestResources()
    }
  }, [testMode, loadTestResources])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!youtubeUrl.trim()) {
      toast({
        variant: "destructive",
        title: "Error",
        description: "Please enter a YouTube URL"
      })
      return
    }

    // Basic YouTube URL validation
    const youtubeRegex = /^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.be)\/.+/
    if (!youtubeRegex.test(youtubeUrl)) {
      toast({
        variant: "destructive",
        title: "Error",
        description: "Please enter a valid YouTube URL"
      })
      return
    }

    setIsLoading(true)
    
    try {
      // Prepare form data
      const formData = new FormData()
      formData.append('youtube_url', youtubeUrl)
      formData.append('use_default_prompt', useDefaultPrompt.toString())
      formData.append('use_saved_script', useSavedScript.toString())
      
      if (!useDefaultPrompt && customPrompt.trim()) {
        formData.append('custom_prompt', customPrompt)
      }

      // Call backend API
      const response = await fetch('/api/extract-transcript', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.message || 'Failed to process YouTube URL')
      }

      const data = await response.json()
      
      // Update store with response
      setSessionId(data.session_id)
      setScript(data.script)
      
      toast({
        title: "Success",
        description: "Script generated successfully!"
      })
      setCurrentStep('script')
      
    } catch (error) {
      console.error('Error processing YouTube URL:', error)
      toast({
        variant: "destructive",
        title: "Error",
        description: error instanceof Error ? error.message : 'Failed to process YouTube URL'
      })
    } finally {
      setIsLoading(false)
    }
  }

  const features = [
    {
      icon: <Video className="h-6 w-6" />,
      title: "AI Video Analysis",
      description: "Intelligent scene detection and face recognition"
    },
    {
      icon: <FileText className="h-6 w-6" />,
      title: "Smart Script Editing",
      description: "Context-aware AI-powered script modifications"
    },
    {
      icon: <Users className="h-6 w-6" />,
      title: "Face Recognition",
      description: "Advanced character detection and tracking"
    },
    {
      icon: <Zap className="h-6 w-6" />,
      title: "Parallel Processing",
      description: "3-5x faster processing with intelligent optimization"
    },
    {
      icon: <Star className="h-6 w-6" />,
      title: "Quality Scoring",
      description: "Automatic quality assessment and scene selection"
    },
    {
      icon: <CheckCircle className="h-6 w-6" />,
      title: "Professional Output",
      description: "High-quality video compilations with smooth transitions"
    }
  ]

  const phases = [
    {
      phase: "Phase 1",
      title: "Foundation & UI",
      status: "Completed",
      description: "Basic setup and user interface"
    },
    {
      phase: "Phase 2",
      title: "Core Backend Services",
      status: "Completed", 
      description: "Parallel processing architecture and API endpoints"
    },
    {
      phase: "Phase 3",
      title: "Script Processing Pipeline",
      status: "Completed",
      description: "Advanced script editing with context-aware AI"
    },
    {
      phase: "Phase 4",
      title: "Enhanced Video Processing",
      status: "Active",
      description: "Parallel video analysis and intelligent scene detection"
    },
    {
      phase: "Phase 5",
      title: "Advanced UI Components",
      status: "Pending",
      description: "Real-time processing monitoring and results display"
    }
  ]

  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <div className="text-center space-y-4">
        <div className="flex items-center justify-center space-x-2 mb-4">
          <div className="w-12 h-12 bg-primary rounded-lg flex items-center justify-center">
            <Video className="h-6 w-6 text-primary-foreground" />
          </div>
          <h1 className="text-4xl font-bold">AI Video Slicer</h1>
        </div>
        <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
          Create compelling video compilations using AI-powered scene detection, 
          face recognition, and intelligent script processing
        </p>
        <div className="flex items-center justify-center space-x-4">
          <Badge variant="default" className="text-sm">
            <Zap className="h-3 w-3 mr-1" />
            Phase 4 Active
          </Badge>
          <Badge variant="secondary" className="text-sm">
            <Settings className="h-3 w-3 mr-1" />
            Parallel Processing
          </Badge>
        </div>
      </div>

      {/* Features Grid */}
      <div>
        <h2 className="text-2xl font-bold text-center mb-6">Key Features</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => (
            <Card key={index} className="hover:shadow-lg transition-shadow">
              <CardHeader>
                <div className="flex items-center space-x-2">
                  <div className="text-primary">
                    {feature.icon}
                  </div>
                  <CardTitle className="text-lg">{feature.title}</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <CardDescription>{feature.description}</CardDescription>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>

      {/* Phase Progress */}
      <div>
        <h2 className="text-2xl font-bold text-center mb-6">Development Progress</h2>
        <div className="space-y-4">
          {phases.map((phase, index) => (
            <Card key={index}>
              <CardContent className="pt-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <div className="flex items-center space-x-2">
                      <Badge 
                        variant={phase.status === "Completed" ? "default" : 
                                phase.status === "Active" ? "secondary" : "outline"}
                        className="w-16 text-center"
                      >
                        {phase.phase}
                      </Badge>
                      <h3 className="font-semibold">{phase.title}</h3>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Badge 
                      variant={phase.status === "Completed" ? "default" : 
                              phase.status === "Active" ? "secondary" : "outline"}
                    >
                      {phase.status}
                    </Badge>
                  </div>
                </div>
                <p className="text-sm text-muted-foreground mt-2">
                  {phase.description}
                </p>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>

      {/* CTA Section */}
      <div className="text-center space-y-4">
        <Card className="max-w-md mx-auto">
          <CardHeader>
            <CardTitle>Ready to Get Started?</CardTitle>
            <CardDescription>
              Experience the power of AI-driven video processing
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button 
              onClick={onStart} 
              size="lg" 
              className="w-full"
            >
              <Play className="mr-2 h-4 w-4" />
              Start Creating
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  )
} 