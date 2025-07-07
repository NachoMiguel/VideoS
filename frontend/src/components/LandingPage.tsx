'use client'

import React, { useState, useEffect } from 'react'
import { useVideoStore } from '@/stores/videoStore'
import { useTestModeStore } from '@/stores/testModeStore'
import { Youtube, Settings, TestTube, Zap } from 'lucide-react'
import { useToast } from "@/hooks/use-toast"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Play, Video, Users, Star, CheckCircle, FileText } from 'lucide-react'

export default function LandingPage() {
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
      
      // Update store with response and navigate to script step
      setSessionId(data.session_id)
      setScript(data.script)
      setCurrentStep('script')
      
      toast({
        title: "Success",
        description: "Script generated successfully!"
      })
      
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
      </div>

      {/* YouTube URL Input Form */}
      <div className="max-w-2xl mx-auto">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Youtube className="h-5 w-5" />
              Get Started with Your YouTube Video
            </CardTitle>
            <CardDescription>
              Enter a YouTube URL to extract transcript and generate an AI-enhanced script
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              {/* YouTube URL Input */}
              <div className="space-y-2">
                <label htmlFor="youtube-url" className="text-sm font-medium">
                  YouTube URL
                </label>
                <input
                  id="youtube-url"
                  type="url"
                  value={youtubeUrl}
                  onChange={(e) => setYoutubeUrl(e.target.value)}
                  placeholder="https://www.youtube.com/watch?v=..."
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  required
                />
              </div>

              {/* Default Prompt Toggle */}
              <div className="flex items-center space-x-2">
                <input
                  id="default-prompt"
                  type="checkbox"
                  checked={useDefaultPrompt}
                  onChange={(e) => setUseDefaultPrompt(e.target.checked)}
                  className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
                />
                <label htmlFor="default-prompt" className="text-sm font-medium">
                  Use default prompt for script rewriting
                </label>
              </div>

              {/* Custom Prompt Input */}
              {!useDefaultPrompt && (
                <div className="space-y-2">
                  <label htmlFor="custom-prompt" className="text-sm font-medium">
                    Custom Prompt
                  </label>
                  <textarea
                    id="custom-prompt"
                    value={customPrompt}
                    onChange={(e) => setCustomPrompt(e.target.value)}
                    placeholder="Enter your custom prompt for script rewriting..."
                    rows={4}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
              )}

              {/* Test Mode Options */}
              {testMode && (
                <div className="space-y-2 p-3 bg-blue-50 rounded-md">
                  <div className="flex items-center space-x-2">
                    <TestTube className="h-4 w-4 text-blue-600" />
                    <span className="text-sm font-medium text-blue-600">Test Mode Options</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <input
                      id="saved-script"
                      type="checkbox"
                      checked={useSavedScript}
                      onChange={(e) => setUseSavedScript(e.target.checked)}
                      className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
                    />
                    <label htmlFor="saved-script" className="text-sm">
                      Use saved script (skip transcript extraction)
                    </label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <input
                      id="known-characters"
                      type="checkbox"
                      checked={useKnownCharacters}
                      onChange={(e) => setUseKnownCharacters(e.target.checked)}
                      className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
                    />
                    <label htmlFor="known-characters" className="text-sm">
                      Use known characters (skip character detection)
                    </label>
                  </div>
                </div>
              )}

              {/* Submit Button */}
              <button
                type="submit"
                disabled={isLoading}
                className={`w-full py-2 px-4 rounded-md text-white font-medium ${
                  isLoading 
                    ? 'bg-gray-400 cursor-not-allowed' 
                    : 'bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500'
                }`}
              >
                {isLoading ? (
                  <div className="flex items-center justify-center space-x-2">
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    <span>Processing...</span>
                  </div>
                ) : (
                  'Extract & Generate Script'
                )}
              </button>
            </form>
          </CardContent>
        </Card>
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
              onClick={() => {
                // Scroll to top to show the form
                window.scrollTo({ top: 0, behavior: 'smooth' })
              }} 
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