'use client'

import React, { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { Card } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { useToast } from '@/hooks/use-toast'
import { useWebSocket } from '../hooks/useWebSocket'
import { PlayCircle, Scissors, Upload as UploadIcon } from 'lucide-react'
import type { UploadFile } from '@/types/upload'

interface VideoProcessorProps {
  onComplete?: (result: any) => void;
  onError?: (error: string) => void;
}

interface ProcessingState {
  status: 'idle' | 'uploading' | 'processing' | 'completed' | 'error';
  progress: number;
  sessionId?: string;
  error?: string;
  result?: any;
}

export const VideoProcessor: React.FC<VideoProcessorProps> = ({ onComplete, onError }) => {
  const [state, setState] = useState<ProcessingState>({
    status: 'idle',
    progress: 0
  });

  const [selectedFile, setSelectedFile] = useState<UploadFile | null>(null);
  const [extractionRange, setExtractionRange] = useState<{ start: number; end: number }>({
    start: 0,
    end: 0
  });

  const { toast } = useToast();
  const { connect, disconnect, sendMessage } = useWebSocket({
    onMessage: handleWebSocketMessage
  });

  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  function handleWebSocketMessage(event: MessageEvent) {
    try {
      const message = JSON.parse(event.data);
      
      switch (message.type) {
        case 'progress':
          setState(prev => ({
            ...prev,
            progress: message.progress,
            status: message.status === 'completed' ? 'completed' : 'processing'
          }));
          break;
          
        case 'error':
          setState(prev => ({
            ...prev,
            status: 'error',
            error: message.error
          }));
          onError?.(message.error);
          break;
          
        case 'completion':
          setState(prev => ({
            ...prev,
            status: 'completed',
            progress: 100,
            result: message.data
          }));
          onComplete?.(message.data);
          break;
      }
    } catch (error) {
      console.error('Failed to parse WebSocket message:', error);
    }
  }

  const handleUpload = async (file: File) => {
    try {
      setState({ status: 'uploading', progress: 0 });
      
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await fetch('/api/video/upload', {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        throw new Error('Upload failed');
      }
      
      const data = await response.json();
      const { session_id } = data;
      
      setState(prev => ({
        ...prev,
        sessionId: session_id,
        status: 'idle'
      }));
      
      // Connect WebSocket for real-time updates
      connect(`ws://localhost:8000/api/video/ws/${session_id}`);
      
      return false; // Prevent default upload behavior
    } catch (error) {
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : 'Upload failed',
        variant: "destructive"
      });
      setState(prev => ({
        ...prev,
        status: 'error',
        error: error instanceof Error ? error.message : 'Upload failed'
      }));
      return false;
    }
  };

  const startProcessing = async () => {
    if (!state.sessionId) {
      toast({
        title: "Error",
        description: "Please upload a video first",
        variant: "destructive"
      });
      return;
    }

    try {
      setState(prev => ({ ...prev, status: 'processing', progress: 0 }));
      
      const response = await fetch(`/api/video/process/${state.sessionId}`, {
        method: 'POST'
      });
      
      if (!response.ok) {
        throw new Error('Processing failed');
      }
      
    } catch (error) {
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : 'Processing failed',
        variant: "destructive"
      });
      setState(prev => ({
        ...prev,
        status: 'error',
        error: error instanceof Error ? error.message : 'Processing failed'
      }));
    }
  };

  const extractScene = async () => {
    if (!state.sessionId) {
      toast({
        title: "Error",
        description: "Please upload a video first",
        variant: "destructive"
      });
      return;
    }

    try {
      const response = await fetch(`/api/video/extract-scene/${state.sessionId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          start_time: extractionRange.start,
          end_time: extractionRange.end
        })
      });
      
      if (!response.ok) {
        throw new Error('Scene extraction failed');
      }
      
      const result = await response.json();
      toast({
        title: "Success",
        description: "Scene extracted successfully"
      });
      
    } catch (error) {
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : 'Scene extraction failed',
        variant: "destructive"
      });
    }
  };

  return (
    <Card className="w-full max-w-3xl p-6">
      <div className="space-y-6">
        {/* Upload Section */}
        <div>
          <div className="relative">
            <input
              type="file"
              accept="video/*"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) {
                  handleUpload(file);
                }
              }}
              id="video-upload"
            />
            <Button
              variant="outline"
              className="w-full"
              onClick={() => document.getElementById('video-upload')?.click()}
            >
              <UploadIcon className="mr-2 h-4 w-4" />
              Select Video
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

          <Button
            onClick={extractScene}
            disabled={!state.sessionId || state.status === 'processing'}
            variant="outline"
            className="flex-1"
          >
            <Scissors className="mr-2 h-4 w-4" />
            Extract Scene
          </Button>
        </div>

        {/* Scene Extraction Controls */}
        {state.sessionId && (
          <div className="space-y-4 rounded-lg border p-4">
            <h3 className="text-lg font-medium">Scene Extraction</h3>
            <div className="space-y-2">
              <div className="grid w-full items-center gap-1.5">
                <Label htmlFor="start-time">Start Time (seconds)</Label>
                <Input
                  id="start-time"
                  type="number"
                  value={extractionRange.start}
                  onChange={e => setExtractionRange(prev => ({
                    ...prev,
                    start: Number(e.target.value)
                  }))}
                  min={0}
                />
              </div>
              <div className="grid w-full items-center gap-1.5">
                <Label htmlFor="end-time">End Time (seconds)</Label>
                <Input
                  id="end-time"
                  type="number"
                  value={extractionRange.end}
                  onChange={e => setExtractionRange(prev => ({
                    ...prev,
                    end: Number(e.target.value)
                  }))}
                  min={0}
                />
              </div>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
}; 