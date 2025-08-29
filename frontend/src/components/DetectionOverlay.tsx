import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Zap, Target } from 'lucide-react';

interface Detection {
  rank: string;
  suit: string;
  confidence: number;
  bbox: [number, number, number, number];
  card_name?: string;
}

interface DetectionOverlayProps {
  detections: Detection[];
  isDetecting: boolean;
}

const DetectionOverlay: React.FC<DetectionOverlayProps> = ({ detections, isDetecting }) => {
  const getSuitColor = (suit: string) => {
    switch (suit.toLowerCase()) {
      case 'hearts':
      case 'diamonds':
        return 'text-red-400';
      case 'clubs':
      case 'spades':
        return 'text-gray-800';
      default:
        return 'text-blue-400';
    }
  };

  const getSuitSymbol = (suit: string) => {
    switch (suit.toLowerCase()) {
      case 'hearts':
        return '♥';
      case 'diamonds':
        return '♦';
      case 'clubs':
        return '♣';
      case 'spades':
        return '♠';
      default:
        return '?';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-400 border-green-400';
    if (confidence >= 0.6) return 'text-yellow-400 border-yellow-400';
    return 'text-red-400 border-red-400';
  };

  const formatRank = (rank: string) => {
    switch (rank.toLowerCase()) {
      case 'a':
      case 'ace':
        return 'A';
      case 'j':
      case 'jack':
        return 'J';
      case 'q':
      case 'queen':
        return 'Q';
      case 'k':
      case 'king':
        return 'K';
      default:
        return rank;
    }
  };

  return (
    <div className="absolute inset-0 pointer-events-none">
      {/* Detection Results Panel */}
      <AnimatePresence>
        {detections.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className="absolute bottom-6 left-6 right-6 bg-black/80 backdrop-blur-md rounded-xl p-4 border border-white/20"
          >
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-2">
                <Target className="w-5 h-5 text-green-400" />
                <h3 className="text-lg font-semibold text-white">
                  Detected Cards ({detections.length})
                </h3>
              </div>
              <motion.div
                className="flex items-center space-x-1 text-green-400"
                animate={{ scale: [1, 1.1, 1] }}
                transition={{ duration: 1, repeat: Infinity }}
              >
                <Zap className="w-4 h-4" />
                <span className="text-sm font-medium">Live</span>
              </motion.div>
            </div>
            
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
              <AnimatePresence>
                {detections.map((detection, index) => (
                  <motion.div
                    key={`${detection.rank}-${detection.suit}-${index}`}
                    initial={{ opacity: 0, scale: 0.8, rotateY: -90 }}
                    animate={{ opacity: 1, scale: 1, rotateY: 0 }}
                    exit={{ opacity: 0, scale: 0.8, rotateY: 90 }}
                    transition={{ 
                      duration: 0.5, 
                      delay: index * 0.1,
                      type: "spring",
                      stiffness: 300,
                      damping: 20
                    }}
                    className={`bg-white/10 backdrop-blur-sm rounded-lg p-3 border-2 ${getConfidenceColor(detection.confidence)} relative overflow-hidden`}
                  >
                    {/* Card Animation Background */}
                    <motion.div
                      className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent"
                      animate={{ x: [-100, 200] }}
                      transition={{ duration: 2, repeat: Infinity, delay: index * 0.5 }}
                    />
                    
                    {/* Card Content */}
                    <div className="relative z-10">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center space-x-2">
                          <span className="text-2xl font-bold text-white">
                            {formatRank(detection.rank)}
                          </span>
                          <span className={`text-2xl ${getSuitColor(detection.suit)}`}>
                            {getSuitSymbol(detection.suit)}
                          </span>
                        </div>
                        <motion.div
                          className={`px-2 py-1 rounded text-xs font-medium ${getConfidenceColor(detection.confidence)} bg-black/30`}
                          animate={{ scale: [1, 1.05, 1] }}
                          transition={{ duration: 1, repeat: Infinity, delay: index * 0.2 }}
                        >
                          {(detection.confidence * 100).toFixed(0)}%
                        </motion.div>
                      </div>
                      
                      <div className="text-sm text-gray-300 capitalize">
                        {formatRank(detection.rank)} of {detection.suit}
                      </div>
                      
                      {/* Confidence Bar */}
                      <div className="mt-2 bg-gray-700 rounded-full h-1.5 overflow-hidden">
                        <motion.div
                          className={`h-full rounded-full ${
                            detection.confidence >= 0.8 ? 'bg-green-400' :
                            detection.confidence >= 0.6 ? 'bg-yellow-400' : 'bg-red-400'
                          }`}
                          initial={{ width: 0 }}
                          animate={{ width: `${detection.confidence * 100}%` }}
                          transition={{ duration: 0.8, delay: index * 0.1 }}
                        />
                      </div>
                    </div>
                    
                    {/* High Confidence Glow Effect */}
                    {detection.confidence >= 0.8 && (
                      <motion.div
                        className="absolute inset-0 border-2 border-green-400 rounded-lg"
                        animate={{
                          boxShadow: [
                            '0 0 0 0 rgba(34, 197, 94, 0.7)',
                            '0 0 0 10px rgba(34, 197, 94, 0)',
                            '0 0 0 0 rgba(34, 197, 94, 0.7)'
                          ]
                        }}
                        transition={{ duration: 2, repeat: Infinity }}
                      />
                    )}
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      
      {/* No Detections Message */}
      <AnimatePresence>
        {isDetecting && detections.length === 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute bottom-6 left-1/2 transform -translate-x-1/2 bg-black/60 backdrop-blur-md rounded-lg px-6 py-3 border border-white/20"
          >
            <div className="flex items-center space-x-3 text-gray-300">
              <motion.div
                className="w-2 h-2 bg-blue-400 rounded-full"
                animate={{ scale: [1, 1.2, 1], opacity: [1, 0.5, 1] }}
                transition={{ duration: 1.5, repeat: Infinity }}
              />
              <span className="text-sm">Scanning for playing cards...</span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      
      {/* Detection Success Animation */}
      <AnimatePresence>
        {detections.length > 0 && (
          <motion.div
            initial={{ opacity: 0, scale: 0 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0 }}
            className="absolute top-6 left-1/2 transform -translate-x-1/2"
          >
            <motion.div
              className="bg-green-600 text-white px-4 py-2 rounded-full flex items-center space-x-2 shadow-lg"
              animate={{ y: [0, -10, 0] }}
              transition={{ duration: 0.6, type: "spring" }}
            >
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 0.5 }}
              >
                <Target className="w-4 h-4" />
              </motion.div>
              <span className="text-sm font-medium">
                {detections.length} card{detections.length > 1 ? 's' : ''} detected!
              </span>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
      
      {/* Scanning Animation Overlay */}
      {isDetecting && (
        <motion.div
          className="absolute inset-0 pointer-events-none"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        >
          {/* Scanning Line */}
          <motion.div
            className="absolute left-0 right-0 h-0.5 bg-gradient-to-r from-transparent via-blue-400 to-transparent"
            animate={{ y: [0, window.innerHeight, 0] }}
            transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
          />
          
          {/* Corner Scanning Indicators */}
          <div className="absolute inset-4">
            {[0, 1, 2, 3].map((corner) => (
              <motion.div
                key={corner}
                className={`absolute w-6 h-6 border-2 border-blue-400 ${
                  corner === 0 ? 'top-0 left-0 border-r-0 border-b-0' :
                  corner === 1 ? 'top-0 right-0 border-l-0 border-b-0' :
                  corner === 2 ? 'bottom-0 left-0 border-r-0 border-t-0' :
                  'bottom-0 right-0 border-l-0 border-t-0'
                }`}
                animate={{
                  opacity: [0.3, 1, 0.3],
                  scale: [1, 1.1, 1]
                }}
                transition={{
                  duration: 2,
                  repeat: Infinity,
                  delay: corner * 0.5
                }}
              />
            ))}
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default DetectionOverlay;