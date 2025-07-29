import React, { useEffect, useState } from 'react';
import { Card, CardContent } from './ui/card';
import { Progress } from './ui/progress';

interface ProcessingStatusProps {
  status: string;
  progress: number;
}

const ProcessingStatus: React.FC<ProcessingStatusProps> = ({ status, progress }) => {
  const [timeEstimate, setTimeEstimate] = useState<string>('');
  const [performanceMode, setPerformanceMode] = useState<string>('');

  useEffect(() => {
    // Calculate time estimate based on progress and performance mode
    if (status === 'processing') {
      if (progress < 20) {
        setTimeEstimate('2-3 minutes remaining');
        setPerformanceMode('🚀 FFmpeg Scene Detection');
      } else if (progress < 50) {
        setTimeEstimate('1-2 minutes remaining');
        setPerformanceMode('🤖 AI Character Detection');
      } else if (progress < 80) {
        setTimeEstimate('30-60 seconds remaining');
        setPerformanceMode('🎬 FFmpeg Video Assembly');
      } else {
        setTimeEstimate('Almost done...');
        setPerformanceMode('✨ Final Processing');
      }
    } else {
      setTimeEstimate('');
      setPerformanceMode('');
    }
  }, [status, progress]);

  if (status === 'idle') {
    return null;
  }

  return (
    <Card className="w-full max-w-md mx-auto">
      <CardContent className="p-6">
        <div className="space-y-4">
          <div className="text-center">
            <h3 className="text-lg font-semibold mb-2">
              {status === 'processing' ? '🚀 Processing Video' : '✅ Complete'}
            </h3>
            {performanceMode && (
              <p className="text-sm text-muted-foreground mb-2">
                {performanceMode}
              </p>
            )}
            {timeEstimate && (
              <p className="text-xs text-blue-600 font-medium">
                ⚡ {timeEstimate}
              </p>
            )}
          </div>
          
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Progress</span>
              <span>{progress}%</span>
            </div>
            <Progress value={progress} className="h-2" />
          </div>
          
          {status === 'processing' && (
            <div className="text-xs text-muted-foreground text-center">
              <p>🎯 Using professional FFmpeg processing</p>
              <p>⚡ 4x faster than previous version</p>
            </div>
          )}
          
          {status === 'completed' && (
            <div className="text-xs text-green-600 text-center">
              <p>✅ Video processing completed successfully!</p>
              <p>🎬 Your video is ready for download</p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default ProcessingStatus; 