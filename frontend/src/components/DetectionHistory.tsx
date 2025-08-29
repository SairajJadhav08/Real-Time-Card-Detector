import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { History, Filter, Search, Calendar, Clock, Trash2 } from 'lucide-react';

interface Detection {
  id: number;
  rank: string;
  suit: string;
  confidence: number;
  timestamp: string;
  card_name?: string;
}

interface DetectionHistoryProps {
  history: Detection[];
  onClearHistory: () => void;
}

const DetectionHistory: React.FC<DetectionHistoryProps> = ({ history, onClearHistory }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [showFilters, setShowFilters] = useState(false);
  const [sortBy, setSortBy] = useState<'timestamp' | 'confidence' | 'card'>('timestamp');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  // Filter and sort history
  const filteredHistory = history
    .filter(detection => {
      if (!searchTerm) return true;
      const searchLower = searchTerm.toLowerCase();
      return (
        detection.rank.toLowerCase().includes(searchLower) ||
        detection.suit.toLowerCase().includes(searchLower) ||
        (detection.card_name && detection.card_name.toLowerCase().includes(searchLower))
      );
    })
    .sort((a, b) => {
      let comparison = 0;
      
      switch (sortBy) {
        case 'timestamp':
          comparison = new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime();
          break;
        case 'confidence':
          comparison = a.confidence - b.confidence;
          break;
        case 'card':
          comparison = `${a.rank} of ${a.suit}`.localeCompare(`${b.rank} of ${b.suit}`);
          break;
      }
      
      return sortOrder === 'asc' ? comparison : -comparison;
    });

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return {
      date: date.toLocaleDateString(),
      time: date.toLocaleTimeString()
    };
  };

  const formatRank = (rank: string) => {
    const rankMap: { [key: string]: string } = {
      'A': 'Ace',
      'K': 'King',
      'Q': 'Queen',
      'J': 'Jack'
    };
    return rankMap[rank] || rank;
  };

  const getSuitSymbol = (suit: string) => {
    const suitMap: { [key: string]: string } = {
      'hearts': '♥️',
      'diamonds': '♦️',
      'clubs': '♣️',
      'spades': '♠️'
    };
    return suitMap[suit.toLowerCase()] || suit;
  };

  const getSuitColor = (suit: string) => {
    const redSuits = ['hearts', 'diamonds'];
    return redSuits.includes(suit.toLowerCase()) ? 'text-red-400' : 'text-gray-300';
  };

  return (
    <div className="h-full flex flex-col bg-gray-900">
      {/* Header */}
      <div className="bg-black/20 backdrop-blur-md border-b border-white/10 p-4">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <History className="w-6 h-6 text-purple-400" />
            <div>
              <h2 className="text-lg font-semibold text-white">Detection History</h2>
              <p className="text-sm text-gray-400">
                {filteredHistory.length} of {history.length} detections
              </p>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setShowFilters(!showFilters)}
              className={`p-2 rounded-lg transition-colors ${
                showFilters ? 'bg-purple-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              <Filter className="w-4 h-4" />
            </button>
            
            <button
              onClick={onClearHistory}
              className="p-2 rounded-lg bg-red-600 text-white hover:bg-red-700 transition-colors"
              title="Clear all history"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </div>
        </div>
        
        {/* Search Bar */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search by rank, suit, or card name..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:border-purple-500 focus:outline-none"
          />
        </div>
        
        {/* Filters */}
        <AnimatePresence>
          {showFilters && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="mt-4 p-4 bg-gray-800 rounded-lg border border-gray-600"
            >
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Sort By
                  </label>
                  <select
                    value={sortBy}
                    onChange={(e) => setSortBy(e.target.value as 'timestamp' | 'confidence' | 'card')}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white focus:border-purple-500 focus:outline-none"
                  >
                    <option value="timestamp">Timestamp</option>
                    <option value="confidence">Confidence</option>
                    <option value="card">Card Name</option>
                  </select>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Order
                  </label>
                  <select
                    value={sortOrder}
                    onChange={(e) => setSortOrder(e.target.value as 'asc' | 'desc')}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white focus:border-purple-500 focus:outline-none"
                  >
                    <option value="desc">Newest First</option>
                    <option value="asc">Oldest First</option>
                  </select>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
      
      {/* History List */}
      <div className="flex-1 overflow-y-auto">
        {filteredHistory.length === 0 ? (
          <div className="flex items-center justify-center h-full text-center">
            <div>
              <History className="w-16 h-16 text-gray-400 mx-auto mb-4" />
              <h3 className="text-xl font-semibold text-white mb-2">
                {history.length === 0 ? 'No Detections Yet' : 'No Results Found'}
              </h3>
              <p className="text-gray-400 max-w-md">
                {history.length === 0
                  ? 'Start detecting cards to see your detection history here.'
                  : 'Try adjusting your search terms or filters.'}
              </p>
            </div>
          </div>
        ) : (
          <div className="p-4 space-y-3">
            <AnimatePresence>
              {filteredHistory.map((detection, index) => {
                const { date, time } = formatTimestamp(detection.timestamp);
                
                return (
                  <motion.div
                    key={detection.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.3, delay: index * 0.05 }}
                    className="bg-gray-800 rounded-lg p-4 border border-gray-700 hover:border-gray-600 transition-colors"
                  >
                    <div className="flex items-center justify-between">
                      {/* Card Info */}
                      <div className="flex items-center space-x-4">
                        <div className="w-12 h-16 bg-white rounded-lg flex items-center justify-center text-black font-bold text-lg shadow-lg">
                          <div className="text-center">
                            <div className={getSuitColor(detection.suit)}>
                              {getSuitSymbol(detection.suit)}
                            </div>
                            <div className="text-xs">{detection.rank}</div>
                          </div>
                        </div>
                        
                        <div>
                          <h3 className="text-lg font-semibold text-white">
                            {formatRank(detection.rank)} of {detection.suit}
                          </h3>
                          <div className="flex items-center space-x-4 text-sm text-gray-400">
                            <div className="flex items-center space-x-1">
                              <Calendar className="w-3 h-3" />
                              <span>{date}</span>
                            </div>
                            <div className="flex items-center space-x-1">
                              <Clock className="w-3 h-3" />
                              <span>{time}</span>
                            </div>
                          </div>
                        </div>
                      </div>
                      
                      {/* Confidence */}
                      <div className="text-right">
                        <div className={`text-lg font-bold ${
                          detection.confidence >= 0.8 ? 'text-green-400' :
                          detection.confidence >= 0.6 ? 'text-yellow-400' : 'text-red-400'
                        }`}>
                          {(detection.confidence * 100).toFixed(1)}%
                        </div>
                        <div className="w-16 bg-gray-700 rounded-full h-2 mt-1">
                          <div
                            className={`h-full rounded-full ${
                              detection.confidence >= 0.8 ? 'bg-green-400' :
                              detection.confidence >= 0.6 ? 'bg-yellow-400' : 'bg-red-400'
                            }`}
                            style={{ width: `${detection.confidence * 100}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  </motion.div>
                );
              })}
            </AnimatePresence>
          </div>
        )}
      </div>
    </div>
  );
};

export default DetectionHistory;