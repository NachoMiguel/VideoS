import { create } from 'zustand'

export interface TestModeSettings {
  // Main mode toggle
  testMode: boolean
  
  // Test mode specific toggles
  useSavedScript: boolean
  useKnownCharacters: boolean
  useSavedAudio: boolean
  
  // Available saved resources
  savedScripts: string[]
  savedAudioFiles: string[]
  selectedAudioFile: string | null
  
  // Known characters for testing
  knownCharacters: string[]
}

export interface TestModeStoreState extends TestModeSettings {
  // Actions
  setTestMode: (enabled: boolean) => void
  setUseSavedScript: (use: boolean) => void
  setUseKnownCharacters: (use: boolean) => void
  setUseSavedAudio: (use: boolean) => void
  setSelectedAudioFile: (file: string | null) => void
  setSavedScripts: (scripts: string[]) => void
  setSavedAudioFiles: (files: string[]) => void
  setKnownCharacters: (characters: string[]) => void
  
  // Utility actions
  resetTestSettings: () => void
  loadTestResources: () => Promise<void>
}

const initialState: TestModeSettings = {
  testMode: false,
  useSavedScript: false,
  useKnownCharacters: false,
  useSavedAudio: false,
  savedScripts: [],
  savedAudioFiles: [],
  selectedAudioFile: null,
  knownCharacters: ['Jean Claude Vandamme', 'Steven Seagal'],
}

export const useTestModeStore = create<TestModeStoreState>((set, get) => ({
  ...initialState,

  // Main toggles
  setTestMode: (enabled) => {
    set({ testMode: enabled })
    
    // If disabling test mode, reset all test settings
    if (!enabled) {
      set({
        useSavedScript: false,
        useKnownCharacters: false,
        useSavedAudio: false,
        selectedAudioFile: null,
      })
    }
  },

  setUseSavedScript: (use) => set({ useSavedScript: use }),
  setUseKnownCharacters: (use) => set({ useKnownCharacters: use }),
  setUseSavedAudio: (use) => {
    set({ useSavedAudio: use })
    // If not using saved audio, clear selection
    if (!use) {
      set({ selectedAudioFile: null })
    }
  },

  // Resource management
  setSelectedAudioFile: (file) => set({ selectedAudioFile: file }),
  setSavedScripts: (scripts) => set({ savedScripts: scripts }),
  setSavedAudioFiles: (files) => set({ savedAudioFiles: files }),
  setKnownCharacters: (characters) => set({ knownCharacters: characters }),

  // Utility functions
  resetTestSettings: () => set({
    useSavedScript: false,
    useKnownCharacters: false,
    useSavedAudio: false,
    selectedAudioFile: null,
  }),

  loadTestResources: async () => {
    const { testMode } = get()
    
    if (!testMode) return

    try {
      // Load saved scripts
      const scriptsResponse = await fetch('/api/test-data/scripts')
      if (scriptsResponse.ok) {
        const scriptsData = await scriptsResponse.json()
        set({ savedScripts: scriptsData.scripts.map((s: any) => s.filename) })
      }

      // Load saved audio files
      const audioResponse = await fetch('/api/test-data/audio')
      if (audioResponse.ok) {
        const audioData = await audioResponse.json()
        set({ savedAudioFiles: audioData.audio_files.map((a: any) => a.filename) })
      }
    } catch (error) {
      console.error('Failed to load test resources:', error)
    }
  },
}))

// Selectors for derived state
export const useTestModeSelectors = {
  // Check if any test mode features are enabled
  hasTestFeatures: () => useTestModeStore(state => 
    state.useSavedScript || state.useKnownCharacters || state.useSavedAudio
  ),
  
  // Get test mode configuration for API calls
  getTestConfig: () => useTestModeStore(state => ({
    testMode: state.testMode,
    useSavedScript: state.useSavedScript,
    useKnownCharacters: state.useKnownCharacters,
    useSavedAudio: state.useSavedAudio,
    selectedAudioFile: state.selectedAudioFile,
    knownCharacters: state.knownCharacters,
  })),
  
  // Check if should skip API calls
  shouldSkipTranscript: () => useTestModeStore(state => 
    state.testMode && state.useSavedScript
  ),
  
  shouldSkipCharacterExtraction: () => useTestModeStore(state => 
    state.testMode && state.useKnownCharacters
  ),
  
  shouldSkipAudioGeneration: () => useTestModeStore(state => 
    state.testMode && state.useSavedAudio && state.selectedAudioFile !== null
  ),
} 