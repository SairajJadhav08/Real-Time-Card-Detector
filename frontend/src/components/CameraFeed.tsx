import React, { RefObject } from 'react';
import { motion } from 'framer-motion';
import Webcam from 'react-webcam';
import { Camera, Play, Pause, Wifi, WifiOff, Clock } from 'lucide-react';

interface CameraFeedProps {
  webcamRef: RefObject<Webcam>;
  isDetecting: boolean;
  onToggleDetection: () => void;
  isConnected: boolean;
  processingTime: number;
  cameraError: string | null;
  cameraReady: boolean;
  onCameraError: (error: string | DOMException) => void;
  onCameraReady: () => void;
}

const CameraFeed: React.FC<CameraFeedProps> = ({
  webcamRef,
  isDetecting,
  onToggleDetection,
  isConnected,
  processingTime,
  cameraError,
  cameraReady,
  onCameraError,
  onCameraReady
}) => {
  const videoConstraints = {
    width: 1280,
    height: 720,
    facingMode: "user"
  };

  return (
    <div className="relative h-full flex flex-col">
      {/* Camera Controls Header */}
      <div className="bg-black/20 backdrop-blur-md border-b border-white/10 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Camera className="w-6 h-6 text-blue-400" />
            <div>
              <h2 className="text-lg font-semibold">Live Camera Feed</h2>
              <p className="text-sm text-gray-400">
                {isDetecting ? 'Detecting cards...' : 'Detection paused'}
              </p>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            {/* Processing Time */}
            <div className="flex items-center space-x-2 text-sm text-gray-400">
              <Clock className="w-4 h-4" />
              <span>{(processingTime * 1000).toFixed(0)}ms</span>
            </div>
            
            {/* Connection Status */}
            <div className="flex items-center space-x-2">
              {isConnected ? (
                <>
                  <Wifi className="w-5 h-5 text-green-400" />
                  <span className="text-sm text-green-400">Connected</span>
                </>
              ) : (
                <>
                  <WifiOff className="w-5 h-5 text-red-400" />
                  <span className="text-sm text-red-400">Disconnected</span>
                </>
              )}
            </div>
            
            {/* Detection Toggle Button */}
            <motion.button
              onClick={onToggleDetection}
              disabled={!isConnected || !cameraReady || !!cameraError}
              className={`px-6 py-2 rounded-lg font-medium flex items-center space-x-2 transition-all duration-200 ${
                isDetecting
                  ? 'bg-red-600 hover:bg-red-700 text-white'
                  : 'bg-green-600 hover:bg-green-700 text-white'
              } disabled:opacity-50 disabled:cursor-not-allowed`}
              whileHover={{ scale: (isConnected && cameraReady && !cameraError) ? 1.05 : 1 }}
              whileTap={{ scale: (isConnected && cameraReady && !cameraError) ? 0.95 : 1 }}
            >
              {isDetecting ? (
                <>
                  <Pause className="w-4 h-4" />
                  <span>Stop Detection</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  <span>Start Detection</span>
                </>
              )}
            </motion.button>
          </div>
        </div>
      </div>

      {/* Camera Feed Container */}
      <div className="flex-1 relative bg-black/30 flex items-center justify-center overflow-hidden">
        <motion.div
          className="relative w-full h-full max-w-4xl max-h-full"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
        >
          {/* Webcam Component */}
          <Webcam
            ref={webcamRef}
            audio={false}
            screenshotFormat="image/jpeg"
            videoConstraints={videoConstraints}
            className="w-full h-full object-cover rounded-lg shadow-2xl"
            style={{
              transform: 'scaleX(-1)', // Mirror the video
            }}
            onUserMedia={onCameraReady}
            onUserMediaError={onCameraError}
          />
          
          {/* Detection Status Overlay */}
          {isDetecting && (
            <motion.div
              className="absolute top-4 left-4 bg-green-600/90 backdrop-blur-sm text-white px-3 py-2 rounded-lg flex items-center space-x-2"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
            >
              <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
              <span className="text-sm font-medium">Detecting...</span>
            </motion.div>
          )}
          
          {/* Frame Rate Indicator */}
          <div className="absolute top-4 right-4 bg-black/50 backdrop-blur-sm text-white px-3 py-2 rounded-lg">
            <span className="text-sm font-mono">5 FPS</span>
          </div>
          
          {/* Center Crosshair */}
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="relative">
              <div className="w-8 h-8 border-2 border-white/50 rounded-full" />
              <div className="absolute inset-0 w-8 h-8 border-t-2 border-blue-400 rounded-full animate-spin" />
            </div>
          </div>
          
          {/* Corner Guides */}
          <div className="absolute inset-4 pointer-events-none">
            {/* Top Left */}
            <div className="absolute top-0 left-0 w-8 h-8 border-l-2 border-t-2 border-white/30" />
            {/* Top Right */}
            <div className="absolute top-0 right-0 w-8 h-8 border-r-2 border-t-2 border-white/30" />
            {/* Bottom Left */}
            <div className="absolute bottom-0 left-0 w-8 h-8 border-l-2 border-b-2 border-white/30" />
            {/* Bottom Right */}
            <div className="absolute bottom-0 right-0 w-8 h-8 border-r-2 border-b-2 border-white/30" />
          </div>
          
          {/* Detection Zone Indicator */}
          {isDetecting && (
            <motion.div
              className="absolute inset-8 border-2 border-blue-400/50 rounded-lg"
              animate={{
                borderColor: ['rgba(59, 130, 246, 0.5)', 'rgba(59, 130, 246, 1)', 'rgba(59, 130, 246, 0.5)'],
              }}
              transition={{
                duration: 2,
                repeat: Infinity,
                ease: 'easeInOut'
              }}
            />
          )}
        </motion.div>
        
        {/* Error Overlays */}
        {cameraError && (
          <motion.div
            className="absolute inset-0 flex items-center justify-center bg-black/70 backdrop-blur-sm"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
          >
            <div className="text-center">
              <Camera className="w-16 h-16 text-red-400 mx-auto mb-4" />
              <h3 className="text-xl font-semibold text-white mb-2">Camera Error</h3>
              <p className="text-gray-400 max-w-md mb-4">
                {cameraError}
              </p>
              <button
                onClick={() => window.location.reload()}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
              >
                Reload Page
              </button>
            </div>
          </motion.div>
        )}
        
        {!isConnected && !cameraError && (
          <motion.div
            className="absolute inset-0 flex items-center justify-center bg-black/70 backdrop-blur-sm"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
          >
            <div className="text-center">
              <WifiOff className="w-16 h-16 text-red-400 mx-auto mb-4" />
              <h3 className="text-xl font-semibold text-white mb-2">Server Connection Lost</h3>
              <p className="text-gray-400 max-w-md">
                Unable to connect to the detection server. Please check if the backend is running.
              </p>
            </div>
          </motion.div>
        )}
        
        {!cameraReady && !cameraError && isConnected && (
          <motion.div
            className="absolute inset-0 flex items-center justify-center bg-black/50 backdrop-blur-sm"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
          >
            <div className="text-center">
              <Camera className="w-16 h-16 text-gray-400 mx-auto mb-4 animate-pulse" />
              <h3 className="text-xl font-semibold text-white mb-2">Initializing Camera</h3>
              <p className="text-gray-400 max-w-md">
                Please allow camera access to start detecting playing cards.
              </p>
            </div>
          </motion.div>
        )}
      </div>
      
      {/* Instructions Footer */}
      <div className="bg-black/20 backdrop-blur-md border-t border-white/10 p-4">
        <div className="flex items-center justify-center space-x-8 text-sm text-gray-400">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-blue-400 rounded-full" />
            <span>Hold cards clearly in view</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-green-400 rounded-full" />
            <span>Good lighting recommended</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-yellow-400 rounded-full" />
            <span>Keep cards steady for best results</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CameraFeed;