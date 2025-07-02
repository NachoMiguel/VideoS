import { useCallback, useEffect, useRef } from 'react';

interface WebSocketHookOptions {
  onMessage?: (event: MessageEvent) => void;
  onOpen?: (event: Event) => void;
  onClose?: (event: CloseEvent) => void;
  onError?: (event: Event) => void;
}

export function useWebSocket(options: WebSocketHookOptions = {}) {
  const wsRef = useRef<WebSocket | null>(null);

  const connect = useCallback((url: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    const ws = new WebSocket(url);

    ws.onopen = (event) => {
      options.onOpen?.(event);
    };

    ws.onmessage = (event) => {
      options.onMessage?.(event);
    };

    ws.onclose = (event) => {
      options.onClose?.(event);
    };

    ws.onerror = (event) => {
      options.onError?.(event);
    };

    wsRef.current = ws;
  }, [options]);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  const sendMessage = useCallback((data: string | ArrayBufferLike | Blob | ArrayBufferView) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(data);
    }
  }, []);

  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  return {
    connect,
    disconnect,
    sendMessage,
    getWebSocket: () => wsRef.current
  };
} 