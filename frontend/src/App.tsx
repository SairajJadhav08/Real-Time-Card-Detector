import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Webcam from 'react-webcam';
import { Camera, Settings, History, BarChart3, Zap, AlertCircle } from 'lucide-react';
import './App.css';

// Components
import CameraFeed from './components/CameraFeed';
import DetectionOverlay from './components/DetectionOverlay';
import StatsPanel from './components/StatsPanel';
import SettingsPanel from './components/SettingsPanel';

// Hooks
import { useWebSocket } from './hooks/useWebSocket';
import { useDetectionHistory } from './hooks/useDetectionHistory';
import { useDetectionStats } from './hooks/useDetectionStats';

// Create a simple loading spinner component inline
const LoadingSpinner = ({ size = 'md' }: { size?: 'sm' | 'md' | 'lg' }) => {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-8 h-8', 
    lg: 'w-12 h-12'
  };

  return (
    <div className={`${sizeClasses[size]} animate-spin rounded-full border-4 border-white/20 border-t-white`} />
  );
};

// DetectionResult interface removed as it's not used

function App() {
  // State management

  const [isDetecting, setIsDetecting] = useState(false);
  const [activePanel, setActivePanel] = useState<'camera' | 'history' | 'stats' | 'settings'>('camera');
  const [theme, setTheme] = useState<'light' | 'dark'>('dark');
  const [showNotification, setShowNotification] = useState(false);
  const [notificationMessage, setNotificationMessage] = useState('');
  const [processingTime, setProcessingTime] = useState(0);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [cameraReady, setCameraReady] = useState(false);

  // Refs
  const webcamRef = useRef<Webcam>(null);

  // Custom hooks
  const { sendFrame, isConnected, detections, lastProcessingTime } = useWebSocket('ws://localhost:8000/ws/client1');
  const { history, addDetection, clearHistory } = useDetectionHistory();
  const { refreshAll } = useDetectionStats();

  // Detection handling
  useEffect(() => {
    if (detections && detections.length > 0) {
      setProcessingTime(lastProcessingTime);
      
      // Add to history if significant detections
      detections.forEach(detection => {
        if (detection.confidence > 0.7) {
          addDetection({
             rank: detection.rank,
             suit: detection.suit,
             confidence: detection.confidence
           });
        }
      });
      
      // Show notification for new detection
      const cardNames = detections
        .filter(d => d.confidence > 0.7)
        .map(d => `${d.rank} of ${d.suit}`)
        .join(', ');
      
      if (cardNames) {
        showNotificationMessage(`Detected: ${cardNames}`);
      }
      
      // Update stats
      refreshAll();
    }
  }, [detections, lastProcessingTime, addDetection, refreshAll]);



  // Camera error handling
  const handleCameraError = useCallback((error: string | DOMException) => {
    console.error('Camera error:', error);
    setCameraError(typeof error === 'string' ? error : error.message);
    setCameraReady(false);
    setIsDetecting(false);
  }, []);

  // Camera ready handler
  const handleCameraReady = useCallback(() => {
    console.log('Camera is ready');
    setCameraError(null);
    setCameraReady(true);
  }, []);

  // Camera frame capture and sending
  const captureAndSend = useCallback(() => {
    if (webcamRef.current && isConnected && isDetecting && cameraReady) {
      try {
        const imageSrc = webcamRef.current.getScreenshot();
        if (imageSrc) {
          sendFrame(imageSrc);
        }
      } catch (error) {
        console.error('Error capturing frame:', error);
        handleCameraError('Failed to capture camera frame');
      }
    }
  }, [isConnected, isDetecting, sendFrame, cameraReady, handleCameraError]);

  // Auto-capture frames when detecting
  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (isDetecting && isConnected) {
      interval = setInterval(captureAndSend, 200); // 5 FPS
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isDetecting, isConnected, captureAndSend]);

  // Notification helper
  const showNotificationMessage = (message: string) => {
    setNotificationMessage(message);
    setShowNotification(true);
    setTimeout(() => setShowNotification(false), 3000);
  };

  // Theme toggle function removed as it's not used

  // Panel navigation
  const renderActivePanel = () => {
    switch (activePanel) {
      case 'camera':
        return (
          <div className="flex-1 flex flex-col">
            <CameraFeed
              webcamRef={webcamRef}
              isDetecting={isDetecting}
              onToggleDetection={() => setIsDetecting(!isDetecting)}
              isConnected={isConnected}
              processingTime={processingTime}
              cameraError={cameraError}
              cameraReady={cameraReady}
              onCameraError={handleCameraError}
              onCameraReady={handleCameraReady}
            />
            <DetectionOverlay
              detections={detections || []}
              isDetecting={isDetecting}
            />
          </div>
        );
      case 'history':
        return (
          <div className="p-4">
            <h2 className="text-2xl font-bold mb-4">Detection History</h2>
            {/* Temporary placeholder until DetectionHistory component is created */}
            <div className="bg-white/5 backdrop-blur-md rounded-lg p-4">
              <p className="text-gray-400 mb-4">Recent detections will appear here</p>
              {history.length > 0 ? (
                <div className="space-y-2">
                  {history.slice(0, 10).map((detection, index) => (
                    <div key={index} className="bg-white/10 rounded-lg p-3 flex justify-between items-center">
                      <span className="font-medium">{detection.rank} of {detection.suit}</span>
                      <span className="text-sm text-gray-400">{(detection.confidence * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-gray-500 text-center py-8">No detections yet</p>
              )}
              {history.length > 0 && (
                <button 
                  onClick={clearHistory}
                  className="mt-4 px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg transition-colors"
                >
                  Clear History
                </button>
              )}
            </div>
          </div>
        );
      case 'stats':
        return (
          <StatsPanel
            isOpen={true}
            onClose={() => setActivePanel('camera')}
          />
        );
      case 'settings':
        return (
          <SettingsPanel
            isOpen={true}
            onClose={() => setActivePanel('camera')}
            settings={{
              theme,
              autoDetect: isDetecting,
              detectionInterval: 200,
              confidenceThreshold: 0.7,
              showConfidence: true,
              soundEnabled: false,
              animationsEnabled: true,
              maxHistory: 1000,
              cameraResolution: '1280x720',
              frameRate: 5
            }}
            onSettingsChange={(newSettings) => {
              setTheme(newSettings.theme);
              setIsDetecting(newSettings.autoDetect);
            }}
          />
        );
      default:
        return null;
    }
  };

  return (
    <div className={`min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 text-white transition-all duration-300 ${theme}`}>
      {/* Header */}
      <header className="bg-black/20 backdrop-blur-md border-b border-white/10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <motion.div 
              className="flex items-center space-x-3"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5 }}
            >
              <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                <Zap className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                  Playing Card Detector
                </h1>
                <p className="text-xs text-gray-400">Real-time AI Recognition</p>
              </div>
            </motion.div>

            {/* Connection Status */}
            <motion.div 
              className="flex items-center space-x-2"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
              <span className="text-sm text-gray-300">
                {isConnected ? 'Connected' : 'Disconnected'}
              </span>
            </motion.div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex h-[calc(100vh-4rem)]">
        {/* Sidebar Navigation */}
        <motion.nav 
          className="w-20 bg-black/20 backdrop-blur-md border-r border-white/10 flex flex-col items-center py-6 space-y-4"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
        >
          {[
            { id: 'camera', icon: Camera, label: 'Camera' },
            { id: 'history', icon: History, label: 'History' },
            { id: 'stats', icon: BarChart3, label: 'Stats' },
            { id: 'settings', icon: Settings, label: 'Settings' },
          ].map((item, index) => (
            <motion.button
              key={item.id}
              onClick={() => setActivePanel(item.id as any)}
              className={`w-12 h-12 rounded-xl flex items-center justify-center transition-all duration-200 ${
                activePanel === item.id
                  ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/25'
                  : 'bg-white/5 text-gray-400 hover:bg-white/10 hover:text-white'
              }`}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: 0.4 + index * 0.1 }}
              title={item.label}
            >
              <item.icon className="w-6 h-6" />
            </motion.button>
          ))}
        </motion.nav>

        {/* Main Panel */}
        <motion.main 
          className="flex-1 overflow-hidden"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.5 }}
        >
          <AnimatePresence mode="wait">
            <motion.div
              key={activePanel}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
              className="h-full"
            >
              {renderActivePanel()}
            </motion.div>
          </AnimatePresence>
        </motion.main>
      </div>

      {/* Notification Toast */}
      <AnimatePresence>
        {showNotification && (
          <motion.div
            initial={{ opacity: 0, y: 50, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 50, scale: 0.9 }}
            className="fixed bottom-6 right-6 bg-green-600 text-white px-6 py-3 rounded-lg shadow-lg flex items-center space-x-2 z-50"
          >
            <AlertCircle className="w-5 h-5" />
            <span>{notificationMessage}</span>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Loading Overlay */}
      {!isConnected && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-white/10 backdrop-blur-md rounded-2xl p-8 text-center">
            <LoadingSpinner size="lg" />
            <p className="mt-4 text-lg font-medium">Connecting to server...</p>
            <p className="text-sm text-gray-400 mt-2">Please ensure the backend is running</p>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;