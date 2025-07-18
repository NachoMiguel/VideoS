import { create } from 'zustand'

interface AppStore {
  videos: File[]
  audioFile: File | null
  status: 'idle' | 'uploading' | 'ready' | 'processing' | 'completed' | 'error'
  progress: number
  setVideos: (videos: File[]) => void
  setAudioFile: (audio: File) => void
  setStatus: (status: 'idle' | 'uploading' | 'ready' | 'processing' | 'completed' | 'error') => void
  setProgress: (progress: number) => void
}

export const useAppStore = create<AppStore>((set, get) => ({
  videos: [],
  audioFile: null,
  status: 'idle',
  progress: 0,
  
  setVideos: (videos) => {
    set({ videos })
    const { audioFile } = get()
    if (audioFile) set({ status: 'ready' })
  },
  
  setAudioFile: (audioFile) => {
    set({ audioFile })
    const { videos } = get()
    if (videos.length > 0) set({ status: 'ready' })
  },
  
  setStatus: (status) => set({ status }),
  setProgress: (progress) => set({ progress })
})) 