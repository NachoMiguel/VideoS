'use client'

import LandingPage from '@/components/LandingPage'
import ScriptEditor from '@/components/ScriptEditor'
import VideoProcessor from '@/components/VideoProcessor'
import CompletedPage from '@/components/CompletedPage'
import ProcessingStatus from '@/components/ProcessingStatus'
import { Button } from '@/components/ui/button'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { useVideoStore } from '@/stores/videoStore'
import { useTestModeStore } from '@/stores/testModeStore'
import { 
  FileText, 
  Video, 
  Settings, 
  CheckCircle, 
  Zap,
  Star,
  Lock
} from 'lucide-react'

export default function Home() {
  // Fix: Use centralized video store instead of local state
  const { currentStep, script, videos } = useVideoStore()
  const { testMode, setTestMode } = useTestModeStore()

  // Define workflow phases with proper access control
  const phases = [
    {
      id: 'landing',
      title: 'Landing Page',
      description: 'Welcome and project overview',
      icon: <Star className="h-4 w-4" />,
      accessible: true // Always accessible
    },
    {
      id: 'script', 
      title: 'Script Editor',
      description: 'Review and edit your script',
      icon: <FileText className="h-4 w-4" />,
      accessible: !!script // Accessible only if script exists
    },
    {
      id: 'upload',
      title: 'Video Upload', 
      description: 'Upload your videos',
      icon: <Video className="h-4 w-4" />,
      accessible: !!script // Accessible only if script exists
    },
    {
      id: 'processing',
      title: 'Processing',
      description: 'Video processing in progress',
      icon: <Zap className="h-4 w-4" />,
      accessible: !!script && videos.length > 0 // Script + videos required
    },
    {
      id: 'completed',
      title: 'Complete',
      description: 'Download your result',
      icon: <CheckCircle className="h-4 w-4" />,
      accessible: currentStep === 'completed' // Only when actually completed
    }
  ]

  const renderCurrentPhase = () => {
    switch (currentStep) {
      case 'landing':
        return <LandingPage />
      case 'script':
        return <ScriptEditor />
      case 'upload':
        return <VideoProcessor />
      case 'processing':
        return <ProcessingStatus />
      case 'completed':
        return <CompletedPage />
      default:
        return <LandingPage />
    }
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Clean Navigation Header */}
      <div className="border-b bg-card">
        <div className="container mx-auto p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <h1 className="text-xl font-bold">AI Video Slicer</h1>
            </div>
            
            <div className="flex items-center space-x-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setTestMode(!testMode)}
              >
                <Settings className="h-4 w-4 mr-2" />
                {testMode ? "Disable" : "Enable"} Test Mode
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Workflow Progress Indicator (NOT Navigation) */}
      <div className="border-b">
        <div className="container mx-auto p-4">
          <Tabs value={currentStep} className="w-full">
            <TabsList className="grid w-full grid-cols-5">
              {phases.map((phase) => (
                <TabsTrigger 
                  key={phase.id} 
                  value={phase.id} 
                  disabled={!phase.accessible}
                  className={`flex items-center space-x-2 ${
                    !phase.accessible 
                      ? 'opacity-50 cursor-not-allowed' 
                      : currentStep === phase.id 
                        ? 'bg-primary text-primary-foreground' 
                        : ''
                  }`}
                >
                  {!phase.accessible ? (
                    <Lock className="h-3 w-3" />
                  ) : (
                    phase.icon
                  )}
                  <span className="hidden sm:inline">{phase.title}</span>
                </TabsTrigger>
              ))}
            </TabsList>
          </Tabs>
        </div>
      </div>

      {/* Main Content */}
      <main className="container mx-auto py-6">
        {renderCurrentPhase()}
      </main>
    </div>
  )
} 