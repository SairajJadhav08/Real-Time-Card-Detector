import { useState, useEffect, useRef, useCallback } from 'react';

interface Detection {
  rank: string;
  suit: string;
  confidence: number;
  bbox: [number, number, number, number];
  card_name?: string;
}

interface WebSocketMessage {
  type: 'frame' | 'detection' | 'error' | 'status' | 'pong';
  data?: any;
  detections?: Detection[];
  message?: string;
  status?: string;
  processing_time?: number;
}

interface UseWebSocketReturn {
  isConnected: boolean;
  detections: Detection[];
  lastProcessingTime: number;
  error: string | null;
  sendFrame: (imageData: string) => void;
  connect: () => void;
  disconnect: () => void;
  clearDetections: () => void;
}

export const useWebSocket = (url: string): UseWebSocketReturn => {
  const [isConnected, setIsConnected] = useState(false);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [lastProcessingTime, setLastProcessingTime] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;
  const reconnectDelay = 3000;

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('WebSocket connected to:', url);
        setIsConnected(true);
        setError(null);
        reconnectAttempts.current = 0;
        
        // Send a ping to test the connection
        try {
          ws.send(JSON.stringify({ type: 'ping', timestamp: new Date().toISOString() }));
          console.log('Ping sent to server');
        } catch (err) {
          console.error('Failed to send ping:', err);
        }
      };

      ws.onmessage = (event) => {
        try {
          // Validate that event.data exists and is not empty
          if (!event.data || event.data.trim() === '') {
            console.warn('Received empty WebSocket message');
            return;
          }

          // Validate that event.data is a string
          if (typeof event.data !== 'string') {
            console.warn('Received non-string WebSocket message:', typeof event.data);
            return;
          }

          const message: WebSocketMessage = JSON.parse(event.data);
          
          // Validate message structure
          if (!message || typeof message !== 'object' || !message.type) {
            console.warn('Received invalid message structure:', message);
            return;
          }
          
          switch (message.type) {
            case 'detection':
              if (message.detections && Array.isArray(message.detections)) {
                setDetections(message.detections);
                if (typeof message.processing_time === 'number') {
                  setLastProcessingTime(message.processing_time);
                }
              }
              break;
              
            case 'error':
              console.error('WebSocket error message:', message.message);
              setError(message.message || 'Unknown error occurred');
              break;
              
            case 'status':
              console.log('WebSocket status:', message.status);
              break;
              
            case 'pong':
              // Handle pong response
              console.log('Received pong from server');
              break;
              
            default:
              console.log('Unknown message type:', message.type, message);
          }
        } catch (err) {
          console.error('Failed to parse WebSocket message:', {
            error: err,
            rawData: event.data,
            dataType: typeof event.data,
            dataLength: event.data?.length
          });
          setError('Failed to parse server response');
        }
      };

      ws.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason);
        setIsConnected(false);
        wsRef.current = null;
        
        // Attempt to reconnect if not a manual disconnect
        if (event.code !== 1000 && reconnectAttempts.current < maxReconnectAttempts) {
          reconnectAttempts.current++;
          console.log(`Attempting to reconnect (${reconnectAttempts.current}/${maxReconnectAttempts})...`);
          
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, reconnectDelay);
        } else if (reconnectAttempts.current >= maxReconnectAttempts) {
          setError('Failed to reconnect to server. Please refresh the page.');
        }
      };

      ws.onerror = (event) => {
        console.error('WebSocket error occurred:', {
          event,
          url,
          readyState: ws.readyState,
          readyStateText: {
            0: 'CONNECTING',
            1: 'OPEN', 
            2: 'CLOSING',
            3: 'CLOSED'
          }[ws.readyState]
        });
        setError(`Connection error: Unable to connect to ${url}`);
      };
    } catch (err) {
      console.error('Failed to create WebSocket connection:', err);
      setError('Failed to connect to server');
    }
  }, [url]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    if (wsRef.current) {
      wsRef.current.close(1000, 'Manual disconnect');
      wsRef.current = null;
    }
    
    setIsConnected(false);
    setDetections([]);
    setError(null);
    reconnectAttempts.current = 0;
  }, []);

  const sendFrame = useCallback((imageData: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      try {
        const message = {
          type: 'frame',
          image: imageData
        };
        wsRef.current.send(JSON.stringify(message));
      } catch (err) {
        console.error('Failed to send frame:', err);
        setError('Failed to send frame to server');
      }
    } else {
      console.warn('WebSocket is not connected');
      setError('Not connected to server');
    }
  }, []);

  const clearDetections = useCallback(() => {
    setDetections([]);
    setError(null);
  }, []);

  // Auto-connect on mount
  useEffect(() => {
    connect();
    
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, []);

  return {
    isConnected,
    detections,
    lastProcessingTime,
    error,
    sendFrame,
    connect,
    disconnect,
    clearDetections
  };
};

export default useWebSocket;