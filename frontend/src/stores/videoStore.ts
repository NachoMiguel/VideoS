import { create } from 'zustand'

export type WorkflowStep = 'landing' | 'script' | 'upload' | 'processing' | 'completed'

export interface VideoFile {
  file: File
  name: string
  size: number
  type: string
  preview?: string
}

export interface ScriptData {
  content: string
  source: 'saved' | 'generated' | 'manual'
  youtube_url?: string
  original_transcript?: string
  prompt_used?: string
}

export interface ProcessingPhase {
  phase: string
  progress: number
  status: string
  message: string
  timestamp?: string
}

export interface VideoStoreState {
  // Core workflow state
  currentStep: WorkflowStep
  sessionId: string | null
  
  // YouTube and script data
  youtubeUrl: string
  useDefaultPrompt: boolean
  customPrompt: string
  script: ScriptData | null
  
  // Video files
  videos: VideoFile[]
  
  // Processing state
  isProcessing: boolean
  processingPhases: ProcessingPhase[]
  currentPhase: ProcessingPhase | null
  overallProgress: number
  
  // Output
  outputVideoUrl: string | null
  downloadUrl: string | null
  
  // Error handling
  error: string | null
  
  // Actions
  setCurrentStep: (step: WorkflowStep) => void
  setSessionId: (id: string | null) => void
  setYoutubeUrl: (url: string) => void
  setUseDefaultPrompt: (use: boolean) => void
  setCustomPrompt: (prompt: string) => void
  setScript: (script: ScriptData | null) => void
  setVideos: (videos: VideoFile[]) => void
  addVideo: (video: VideoFile) => void
  removeVideo: (index: number) => void
  
  // Processing actions
  setIsProcessing: (processing: boolean) => void
  addProcessingPhase: (phase: ProcessingPhase) => void
  updateCurrentPhase: (phase: ProcessingPhase) => void
  setOverallProgress: (progress: number) => void
  
  // Output actions
  setOutputVideoUrl: (url: string | null) => void
  setDownloadUrl: (url: string | null) => void
  
  // Error actions
  setError: (error: string | null) => void
  
  // Utility actions
  reset: () => void
  resetToScript: () => void
}

const initialState = {
  currentStep: 'landing' as WorkflowStep,
  sessionId: null,
  youtubeUrl: '',
  useDefaultPrompt: true,
  customPrompt: '',
  script: null,
  videos: [],
  isProcessing: false,
  processingPhases: [],
  currentPhase: null,
  overallProgress: 0,
  outputVideoUrl: null,
  downloadUrl: null,
  error: null,
}

export const useVideoStore = create<VideoStoreState>((set, get) => ({
  ...initialState,

  // Basic setters
  setCurrentStep: (step) => set({ currentStep: step }),
  setSessionId: (id) => set({ sessionId: id }),
  setYoutubeUrl: (url) => set({ youtubeUrl: url }),
  setUseDefaultPrompt: (use) => set({ useDefaultPrompt: use }),
  setCustomPrompt: (prompt) => set({ customPrompt: prompt }),
  setScript: (script) => set({ script }),
  setVideos: (videos) => set({ videos }),
  
  // Video management
  addVideo: (video) => {
    const { videos } = get()
    if (videos.length < 3) { // Max 3 videos
      set({ videos: [...videos, video] })
    }
  },
  
  removeVideo: (index) => {
    const { videos } = get()
    const newVideos = videos.filter((_, i) => i !== index)
    set({ videos: newVideos })
  },
  
  // Processing state management
  setIsProcessing: (processing) => set({ isProcessing: processing }),
  
  addProcessingPhase: (phase) => {
    const { processingPhases } = get()
    set({ 
      processingPhases: [...processingPhases, phase],
      currentPhase: phase 
    })
  },
  
  updateCurrentPhase: (phase) => {
    set({ 
      currentPhase: phase,
      overallProgress: phase.progress 
    })
  },
  
  setOverallProgress: (progress) => set({ overallProgress: progress }),
  
  // Output management
  setOutputVideoUrl: (url) => set({ outputVideoUrl: url }),
  setDownloadUrl: (url) => set({ downloadUrl: url }),
  
  // Error management
  setError: (error) => set({ error }),
  
  // Utility functions
  reset: () => set(initialState),
  
  resetToScript: () => set({
    currentStep: 'script',
    videos: [],
    isProcessing: false,
    processingPhases: [],
    currentPhase: null,
    overallProgress: 0,
    outputVideoUrl: null,
    downloadUrl: null,
    error: null,
  }),
}))

// Selectors for common derived state
export const useVideoStoreSelectors = {
  hasVideos: () => useVideoStore(state => state.videos.length > 0),
  hasScript: () => useVideoStore(state => state.script !== null),
  canProceedToProcessing: () => useVideoStore(state => 
    state.script !== null && state.videos.length > 0 && state.videos.length <= 3
  ),
  isCompleted: () => useVideoStore(state => 
    state.currentStep === 'completed' && state.outputVideoUrl !== null
  ),
} 