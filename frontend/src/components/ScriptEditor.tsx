'use client'

import React, { useState, useCallback, useRef, useEffect } from 'react'
import { useVideoStore } from '@/stores/videoStore'
import { Edit3, Scissors, Plus, Zap, Trash2, ArrowRight, Undo, Redo, Save, Eye, EyeOff } from 'lucide-react'
import { useToast } from '@/hooks/use-toast'

// Script modification actions with keyboard shortcuts
const ModificationActions = [
  { id: 'shorten', label: 'Shorten', icon: Scissors, description: 'Make it more concise', shortcut: 'Ctrl+1' },
  { id: 'expand', label: 'Expand', icon: Plus, description: 'Add more detail', shortcut: 'Ctrl+2' },
  { id: 'rewrite', label: 'Rewrite', icon: Edit3, description: 'Improve the text', shortcut: 'Ctrl+3' },
  { id: 'make_engaging', label: 'Engaging', icon: Zap, description: 'Make it more compelling', shortcut: 'Ctrl+4' },
  { id: 'delete', label: 'Delete', icon: Trash2, description: 'Remove this text', shortcut: 'Ctrl+Del' },
];

interface TextSelection {
  selectedText: string;
  contextBefore: string;
  contextAfter: string;
  startIndex: number;
  endIndex: number;
}

interface ModificationPreview {
  action: string;
  originalText: string;
  modifiedText: string;
  contextBefore: string;
  contextAfter: string;
}

interface ScriptHistory {
  content: string;
  timestamp: number;
  action?: string;
}

interface ScriptEditorProps {
  onNext?: () => void
}

export default function ScriptEditor({ onNext }: ScriptEditorProps) {
  const { 
    script, 
    sessionId,
    setScript, 
    setCurrentStep 
  } = useVideoStore()
  
  const { toast } = useToast()
  
  const [textSelection, setTextSelection] = useState<TextSelection | null>(null)
  const [isModifying, setIsModifying] = useState(false)
  const [modificationPreview, setModificationPreview] = useState<ModificationPreview | null>(null)
  const [showPreview, setShowPreview] = useState(false)
  const [history, setHistory] = useState<ScriptHistory[]>([])
  const [historyIndex, setHistoryIndex] = useState(-1)
  const [bulkModifications, setBulkModifications] = useState<any[]>([])
  
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const modificationPanelRef = useRef<HTMLDivElement>(null)

  // Initialize history with current script
  useEffect(() => {
    if (script && history.length === 0) {
      setHistory([{ content: script.content, timestamp: Date.now() }])
      setHistoryIndex(0)
    }
  }, [script, history.length])

  // Handle text selection
  const handleTextSelection = useCallback(() => {
    const textarea = textareaRef.current
    if (!textarea || !script) return

    const start = textarea.selectionStart
    const end = textarea.selectionEnd
    
    if (start === end) {
      setTextSelection(null)
      return
    }

    const content = script.content
    const selectedText = content.substring(start, end)
    
    // Get context (50 characters before and after, or edge of text)
    const contextStart = Math.max(0, start - 50)
    const contextEnd = Math.min(content.length, end + 50)
    
    const contextBefore = content.substring(contextStart, start)
    const contextAfter = content.substring(end, contextEnd)

    setTextSelection({
      selectedText,
      contextBefore,
      contextAfter,
      startIndex: start,
      endIndex: end
    })
  }, [script])

  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!textSelection || isModifying) return

      if (e.ctrlKey || e.metaKey) {
        const action = ModificationActions.find(a => {
          if (a.shortcut === 'Ctrl+1' && e.key === '1') return true
          if (a.shortcut === 'Ctrl+2' && e.key === '2') return true
          if (a.shortcut === 'Ctrl+3' && e.key === '3') return true
          if (a.shortcut === 'Ctrl+4' && e.key === '4') return true
          if (a.shortcut === 'Ctrl+Del' && e.key === 'Delete') return true
          return false
        })

        if (action) {
          e.preventDefault()
          handleModification(action.id)
        }
      }

      // Undo/Redo
      if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
        e.preventDefault()
        handleUndo()
      }
      if ((e.ctrlKey || e.metaKey) && (e.key === 'y' || (e.key === 'z' && e.shiftKey))) {
        e.preventDefault()
        handleRedo()
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [textSelection, isModifying, historyIndex, history])

  // Handle script modification
  const handleModification = async (action: string) => {
    if (!textSelection || !script || !sessionId) return

    setIsModifying(true)
    
    try {
      const response = await fetch('/api/v1/modify-script', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          action,
          selected_text: textSelection.selectedText,
          context_before: textSelection.contextBefore,
          context_after: textSelection.contextAfter
        })
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      
      if (data.success) {
        setModificationPreview({
          action,
          originalText: textSelection.selectedText,
          modifiedText: data.modified_text,
          contextBefore: textSelection.contextBefore,
          contextAfter: textSelection.contextAfter
        })
        setShowPreview(true)
      } else {
        toast({
          variant: "destructive",
          title: "Error",
          description: data.error || 'Modification failed'
        })
      }
    } catch (error) {
      console.error('Modification error:', error)
      toast({
        variant: "destructive",
        title: "Error",
        description: 'Failed to modify text. Please try again.'
      })
    } finally {
      setIsModifying(false)
    }
  }

  // Apply modification
  const applyModification = () => {
    if (!modificationPreview || !textSelection || !script) return

    const newContent = 
      script.content.substring(0, textSelection.startIndex) +
      modificationPreview.modifiedText +
      script.content.substring(textSelection.endIndex)

    // Add to history
    const newHistory = history.slice(0, historyIndex + 1)
    newHistory.push({
      content: newContent,
      timestamp: Date.now(),
      action: modificationPreview.action
    })
    
    setHistory(newHistory)
    setHistoryIndex(newHistory.length - 1)
    
    setScript({ ...script, content: newContent })
    setModificationPreview(null)
    setShowPreview(false)
    setTextSelection(null)
    
    toast({ 
      title: `Text ${modificationPreview.action} applied successfully`,
      variant: "success" 
    })
  }

  // Reject modification
  const rejectModification = () => {
    setModificationPreview(null)
    setShowPreview(false)
  }

  // Undo/Redo functionality
  const handleUndo = () => {
    if (historyIndex > 0 && script) {
      const previousState = history[historyIndex - 1]
      setScript({ ...script, content: previousState.content })
      setHistoryIndex(historyIndex - 1)
      toast({ 
        title: 'Undone',
        variant: "success" 
      })
    }
  }

  const handleRedo = () => {
    if (historyIndex < history.length - 1 && script) {
      const nextState = history[historyIndex + 1]
      setScript({ ...script, content: nextState.content })
      setHistoryIndex(historyIndex + 1)
      toast({ 
        title: 'Redone',
        variant: "success" 
      })
    }
  }

  // Handle bulk modifications (future feature)
  const handleBulkModification = async () => {
    // Implementation for bulk modifications
    toast({ 
      title: 'Bulk modifications coming soon!',
      description: 'ℹ️'
    })
  }

  const handleProceed = () => {
    if (!script?.content.trim()) {
      toast({ 
        title: 'Script cannot be empty',
        variant: "destructive" 
      })
      return
    }

    toast({ 
      title: 'Script ready! Proceeding to video upload...',
      variant: "success" 
    })
    setCurrentStep('upload')
  }

  if (!script) {
    return (
      <div className="max-w-4xl mx-auto">
        <div className="card text-center">
          <p className="text-gray-500">No script available. Please go back and generate a script first.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-7xl mx-auto">
      <div className="mb-8">
        <h2 className="text-3xl font-bold text-gray-900 mb-2">Advanced Script Editor</h2>
        <p className="text-gray-600">
          Select text and use AI-powered modifications. Use keyboard shortcuts for quick editing.
        </p>
      </div>

      <div className="grid lg:grid-cols-4 gap-6">
        {/* Main Editor */}
        <div className="lg:col-span-3">
          <div className="card">
            {/* Editor Header */}
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-lg font-semibold text-gray-900">Script Content</h3>
                <p className="text-sm text-gray-600 mt-1">
                  Source: {script.source === 'saved' ? 'Saved Script' : 'Generated from YouTube'}
                </p>
              </div>
              
              {/* Editor Controls */}
              <div className="flex items-center space-x-2">
                <button
                  onClick={handleUndo}
                  disabled={historyIndex <= 0}
                  className="p-2 text-gray-400 hover:text-gray-600 disabled:opacity-50 disabled:cursor-not-allowed"
                  title="Undo (Ctrl+Z)"
                >
                  <Undo className="w-4 h-4" />
                </button>
                
                <button
                  onClick={handleRedo}
                  disabled={historyIndex >= history.length - 1}
                  className="p-2 text-gray-400 hover:text-gray-600 disabled:opacity-50 disabled:cursor-not-allowed"
                  title="Redo (Ctrl+Y)"
                >
                  <Redo className="w-4 h-4" />
                </button>
                
                <div className="h-4 border-l border-gray-300 mx-2" />
                
                <button
                  onClick={() => setShowPreview(!showPreview)}
                  className="p-2 text-gray-400 hover:text-gray-600"
                  title="Toggle Preview"
                >
                  {showPreview ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            {/* Text Editor */}
            <textarea
              ref={textareaRef}
              value={script.content}
              onChange={(e) => setScript({ ...script, content: e.target.value })}
              onSelect={handleTextSelection}
              onMouseUp={handleTextSelection}
              onKeyUp={handleTextSelection}
              className="script-editor min-h-[500px]"
              placeholder="Your script content will appear here..."
            />

            {/* Selection Info */}
            {textSelection && (
              <div className="mt-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
                <p className="text-sm text-blue-800">
                  <strong>Selected:</strong> "{textSelection.selectedText.length > 50 
                    ? textSelection.selectedText.substring(0, 50) + '...' 
                    : textSelection.selectedText}"
                </p>
                <p className="text-xs text-blue-600 mt-1">
                  {textSelection.selectedText.length} characters selected. Choose a modification action.
                </p>
              </div>
            )}
          </div>

          {/* Modification Preview */}
          {modificationPreview && showPreview && (
            <div className="card mt-6">
              <div className="mb-4">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">Modification Preview</h3>
                <p className="text-sm text-gray-600">
                  Action: <span className="font-medium capitalize">{modificationPreview.action.replace('_', ' ')}</span>
                </p>
              </div>

              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <h4 className="text-sm font-medium text-gray-700 mb-2">Original</h4>
                  <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
                    <p className="text-sm text-gray-800">{modificationPreview.originalText}</p>
                  </div>
                </div>
                
                <div>
                  <h4 className="text-sm font-medium text-gray-700 mb-2">Modified</h4>
                  <div className="p-3 bg-green-50 border border-green-200 rounded-lg">
                    <p className="text-sm text-gray-800">{modificationPreview.modifiedText}</p>
                  </div>
                </div>
              </div>

              <div className="flex items-center justify-end space-x-3 mt-4">
                <button
                  onClick={rejectModification}
                  className="btn-secondary"
                >
                  Reject
                </button>
                <button
                  onClick={applyModification}
                  className="btn-primary"
                >
                  Apply Changes
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Modification Actions */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Modification Actions</h3>
            
            {textSelection ? (
              <div className="space-y-2">
                {ModificationActions.map((action) => {
                  const Icon = action.icon
                  return (
                    <button
                      key={action.id}
                      onClick={() => handleModification(action.id)}
                      disabled={isModifying}
                      className="w-full flex items-center space-x-3 p-3 text-left hover:bg-gray-50 rounded-lg border border-gray-200 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      <Icon className="w-4 h-4 text-gray-500" />
                      <div className="flex-1">
                        <p className="text-sm font-medium text-gray-900">{action.label}</p>
                        <p className="text-xs text-gray-500">{action.description}</p>
                      </div>
                      <span className="text-xs text-gray-400">{action.shortcut}</span>
                    </button>
                  )
                })}
              </div>
            ) : (
              <p className="text-sm text-gray-500 text-center py-4">
                Select text to see modification options
              </p>
            )}

            {isModifying && (
              <div className="mt-4 p-3 bg-blue-50 rounded-lg">
                <p className="text-sm text-blue-800 text-center">
                  Processing modification...
                </p>
              </div>
            )}
          </div>

          {/* Script Information */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Script Information</h3>
            
            <div className="space-y-3 text-sm">
              <div>
                <span className="font-medium text-gray-700">Length:</span>
                <span className="ml-2 text-gray-600">
                  {script.content.length} characters
                </span>
              </div>
              
              <div>
                <span className="font-medium text-gray-700">Words:</span>
                <span className="ml-2 text-gray-600">
                  ~{Math.round(script.content.split(' ').length)} words
                </span>
              </div>

              <div>
                <span className="font-medium text-gray-700">Est. Duration:</span>
                <span className="ml-2 text-gray-600">
                  ~{Math.round(script.content.split(' ').length / 150)} min
                </span>
              </div>

              <div>
                <span className="font-medium text-gray-700">Modifications:</span>
                <span className="ml-2 text-gray-600">
                  {history.length - 1} changes
                </span>
              </div>
            </div>
          </div>

          {/* Keyboard Shortcuts */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Shortcuts</h3>
            
            <div className="space-y-2 text-xs">
              <div className="flex justify-between">
                <span className="text-gray-600">Shorten</span>
                <span className="text-gray-400">Ctrl+1</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Expand</span>
                <span className="text-gray-400">Ctrl+2</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Rewrite</span>
                <span className="text-gray-400">Ctrl+3</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Engaging</span>
                <span className="text-gray-400">Ctrl+4</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Delete</span>
                <span className="text-gray-400">Ctrl+Del</span>
              </div>
              <div className="border-t border-gray-200 pt-2 mt-2">
                <div className="flex justify-between">
                  <span className="text-gray-600">Undo</span>
                  <span className="text-gray-400">Ctrl+Z</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Redo</span>
                  <span className="text-gray-400">Ctrl+Y</span>
                </div>
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="space-y-3">
            <button
              onClick={handleProceed}
              className="btn-primary w-full flex items-center justify-center"
            >
              <ArrowRight className="w-4 h-4 mr-2" />
              Proceed to Video Upload
            </button>
            
            <button
              onClick={() => setCurrentStep('landing')}
              className="btn-secondary w-full"
            >
              Back to Start
            </button>
          </div>
        </div>
      </div>
    </div>
  )
} 