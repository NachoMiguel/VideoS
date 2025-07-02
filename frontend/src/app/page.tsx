'use client'

import { useState } from 'react'
import LandingPage from '@/components/LandingPage'
import ScriptEditor from '@/components/ScriptEditor'
import VideoProcessor from '@/components/VideoProcessor'
import CompletedPage from '@/components/CompletedPage'
import ProcessingStatus from '@/components/ProcessingStatus'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { 
  FileText, 
  Video, 
  Settings, 
  CheckCircle, 
  Zap,
  Play,
  Users,
  Star
} from 'lucide-react'

export default function Home() {
  const [currentPhase, setCurrentPhase] = useState<'landing' | 'script' | 'video' | 'processing' | 'completed'>('landing')
  const [testMode, setTestMode] = useState(false)

  const phases = [
    {
      id: 'landing',
      title: 'Landing Page',
      description: 'Welcome and project overview',
      icon: <Star className="h-4 w-4" />
    },
    {
      id: 'script',
      title: 'Script Editor',
      description: 'Advanced script editing with AI',
      icon: <FileText className="h-4 w-4" />
    },
    {
      id: 'video',
      title: 'Video Processor',
      description: 'Phase 4: Enhanced parallel video processing',
      icon: <Video className="h-4 w-4" />
    },
    {
      id: 'processing',
      title: 'Processing Status',
      description: 'Real-time processing monitoring',
      icon: <Zap className="h-4 w-4" />
    },
    {
      id: 'completed',
      title: 'Completed Page',
      description: 'Final results and download',
      icon: <CheckCircle className="h-4 w-4" />
    }
  ]

  const renderCurrentPhase = () => {
    switch (currentPhase) {
      case 'landing':
        return <LandingPage onStart={() => setCurrentPhase('script')} />
      case 'script':
        return <ScriptEditor onNext={() => setCurrentPhase('video')} />
      case 'video':
        return <VideoProcessor />
      case 'processing':
        return <ProcessingStatus onComplete={() => setCurrentPhase('completed')} />
      case 'completed':
        return <CompletedPage onRestart={() => setCurrentPhase('landing')} />
      default:
        return <LandingPage onStart={() => setCurrentPhase('script')} />
    }
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Phase Navigation */}
      <div className="border-b bg-card">
        <div className="container mx-auto p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <h1 className="text-xl font-bold">AI Video Slicer</h1>
              <div className="flex items-center space-x-2">
                <Badge variant="outline" className="text-xs">
                  Phase 4
                </Badge>
                <Badge variant={testMode ? "default" : "secondary"} className="text-xs">
                  {testMode ? "Test Mode" : "Production"}
                </Badge>
              </div>
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

      {/* Phase Tabs */}
      <div className="border-b">
        <div className="container mx-auto p-4">
          <Tabs value={currentPhase} onValueChange={(value) => setCurrentPhase(value as any)}>
            <TabsList className="grid w-full grid-cols-5">
              {phases.map((phase) => (
                <TabsTrigger key={phase.id} value={phase.id} className="flex items-center space-x-2">
                  {phase.icon}
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

      {/* Phase Overview */}
      <div className="border-t bg-muted/50">
        <div className="container mx-auto p-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Zap className="h-5 w-5" />
                Phase 4 Features
              </CardTitle>
              <CardDescription>
                Enhanced parallel video processing with intelligent scene detection
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="flex items-center space-x-2">
                  <Video className="h-4 w-4 text-blue-500" />
                  <span className="text-sm">Parallel Video Analysis</span>
                </div>
                <div className="flex items-center space-x-2">
                  <Users className="h-4 w-4 text-green-500" />
                  <span className="text-sm">Face Recognition</span>
                </div>
                <div className="flex items-center space-x-2">
                  <Star className="h-4 w-4 text-yellow-500" />
                  <span className="text-sm">Quality Scoring</span>
                </div>
                <div className="flex items-center space-x-2">
                  <Play className="h-4 w-4 text-purple-500" />
                  <span className="text-sm">Scene Detection</span>
                </div>
                <div className="flex items-center space-x-2">
                  <Zap className="h-4 w-4 text-orange-500" />
                  <span className="text-sm">3-5x Performance</span>
                </div>
                <div className="flex items-center space-x-2">
                  <CheckCircle className="h-4 w-4 text-green-500" />
                  <span className="text-sm">Intelligent Selection</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}

// Badge component for the overview
function Badge({ variant, className, children }: { 
  variant: 'default' | 'secondary' | 'outline'
  className?: string
  children: React.ReactNode 
}) {
  const baseClasses = "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors"
  const variantClasses = {
    default: "border-transparent bg-primary text-primary-foreground hover:bg-primary/80",
    secondary: "border-transparent bg-secondary text-secondary-foreground hover:bg-secondary/80",
    outline: "text-foreground"
  }
  
  return (
    <div className={`${baseClasses} ${variantClasses[variant]} ${className || ''}`}>
      {children}
    </div>
  )
} 