import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { History, Download, Filter, Search, Calendar, Clock, TrendingUp, X } from 'lucide-react';
import axios from 'axios';

interface Detection {
  id: number;
  rank: string;
  suit: string;
  confidence: number;
  timestamp: string;
  card_name?: string;
}

interface HistoryFilter {
  start_date?: string;
  end_date?: string;
  rank?: string;
  suit?: string;
  min_confidence?: number;
  limit?: number;
}

interface HistoryPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

const HistoryPanel: React.FC<HistoryPanelProps> = ({ isOpen, onClose }) => {
  const [detections, setDetections] = useState<Detection[]>([]);
  const [loading, setLoading] = useState(false);
  const [filter, setFilter] = useState<HistoryFilter>({ limit: 50 });
  const [searchTerm, setSearchTerm] = useState('');
  const [showFilters, setShowFilters] = useState(false);
  const [exporting, setExporting] = useState(false);

  const API_BASE_URL = 'http://localhost:8000';

  useEffect(() => {
    if (isOpen) {
      fetchHistory();
    }
  }, [isOpen, filter]);

  const fetchHistory = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API_BASE_URL}/api/history`, {
        params: filter
      });
      setDetections(response.data.detections || []);
    } catch (error) {
      console.error('Failed to fetch history:', error);
    } finally {
      setLoading(false);
    }
  };

  const exportHistory = async (format: 'json' | 'csv') => {
    setExporting(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/api/export`, {
        format,
        filter
      }, {
        responseType: 'blob'
      });
      
      const blob = new Blob([response.data]);
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `card_detections.${format}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Failed to export history:', error);
    } finally {
      setExporting(false);
    }
  };

  const clearHistory = async () => {
    if (window.confirm('Are you sure you want to clear all detection history?')) {
      try {
        await axios.delete(`${API_BASE_URL}/api/history`);
        setDetections([]);
      } catch (error) {
        console.error('Failed to clear history:', error);
      }
    }
  };

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

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return {
      date: date.toLocaleDateString(),
      time: date.toLocaleTimeString()
    };
  };

  const filteredDetections = detections.filter(detection => {
    if (!searchTerm) return true;
    const searchLower = searchTerm.toLowerCase();
    return (
      detection.rank.toLowerCase().includes(searchLower) ||
      detection.suit.toLowerCase().includes(searchLower) ||
      `${detection.rank} of ${detection.suit}`.toLowerCase().includes(searchLower)
    );
  });

  const ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'];
  const suits = ['hearts', 'diamonds', 'clubs', 'spades'];

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
          onClick={onClose}
        >
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.9, opacity: 0 }}
            className="bg-gray-900 rounded-xl shadow-2xl w-full max-w-4xl max-h-[90vh] overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="bg-gradient-to-r from-blue-600 to-purple-600 p-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <History className="w-6 h-6 text-white" />
                  <h2 className="text-2xl font-bold text-white">Detection History</h2>
                  <span className="bg-white/20 text-white px-3 py-1 rounded-full text-sm">
                    {filteredDetections.length} records
                  </span>
                </div>
                <button
                  onClick={onClose}
                  className="text-white hover:bg-white/20 p-2 rounded-lg transition-colors"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
            </div>

            {/* Controls */}
            <div className="p-6 border-b border-gray-700">
              <div className="flex flex-wrap items-center gap-4">
                {/* Search */}
                <div className="flex-1 min-w-64">
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                    <input
                      type="text"
                      placeholder="Search cards..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="w-full pl-10 pr-4 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:border-blue-500 focus:outline-none"
                    />
                  </div>
                </div>

                {/* Filter Toggle */}
                <button
                  onClick={() => setShowFilters(!showFilters)}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-colors ${
                    showFilters ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  <Filter className="w-4 h-4" />
                  <span>Filters</span>
                </button>

                {/* Export */}
                <div className="flex space-x-2">
                  <button
                    onClick={() => exportHistory('json')}
                    disabled={exporting}
                    className="flex items-center space-x-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors disabled:opacity-50"
                  >
                    <Download className="w-4 h-4" />
                    <span>JSON</span>
                  </button>
                  <button
                    onClick={() => exportHistory('csv')}
                    disabled={exporting}
                    className="flex items-center space-x-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors disabled:opacity-50"
                  >
                    <Download className="w-4 h-4" />
                    <span>CSV</span>
                  </button>
                </div>

                {/* Clear History */}
                <button
                  onClick={clearHistory}
                  className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors"
                >
                  Clear All
                </button>
              </div>

              {/* Advanced Filters */}
              <AnimatePresence>
                {showFilters && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    className="mt-4 p-4 bg-gray-800 rounded-lg overflow-hidden"
                  >
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                      {/* Date Range */}
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                          Start Date
                        </label>
                        <input
                          type="date"
                          value={filter.start_date || ''}
                          onChange={(e) => setFilter({ ...filter, start_date: e.target.value })}
                          className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white focus:border-blue-500 focus:outline-none"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                          End Date
                        </label>
                        <input
                          type="date"
                          value={filter.end_date || ''}
                          onChange={(e) => setFilter({ ...filter, end_date: e.target.value })}
                          className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white focus:border-blue-500 focus:outline-none"
                        />
                      </div>

                      {/* Rank Filter */}
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                          Rank
                        </label>
                        <select
                          value={filter.rank || ''}
                          onChange={(e) => setFilter({ ...filter, rank: e.target.value || undefined })}
                          className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white focus:border-blue-500 focus:outline-none"
                        >
                          <option value="">All Ranks</option>
                          {ranks.map(rank => (
                            <option key={rank} value={rank}>{rank}</option>
                          ))}
                        </select>
                      </div>

                      {/* Suit Filter */}
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                          Suit
                        </label>
                        <select
                          value={filter.suit || ''}
                          onChange={(e) => setFilter({ ...filter, suit: e.target.value || undefined })}
                          className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white focus:border-blue-500 focus:outline-none"
                        >
                          <option value="">All Suits</option>
                          {suits.map(suit => (
                            <option key={suit} value={suit}>
                              {suit.charAt(0).toUpperCase() + suit.slice(1)}
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>

                    <div className="mt-4 flex items-center space-x-4">
                      {/* Confidence Filter */}
                      <div className="flex-1">
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                          Min Confidence: {filter.min_confidence || 0}%
                        </label>
                        <input
                          type="range"
                          min="0"
                          max="100"
                          value={(filter.min_confidence || 0) * 100}
                          onChange={(e) => setFilter({ ...filter, min_confidence: parseInt(e.target.value) / 100 })}
                          className="w-full"
                        />
                      </div>

                      {/* Limit */}
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                          Limit
                        </label>
                        <select
                          value={filter.limit || 50}
                          onChange={(e) => setFilter({ ...filter, limit: parseInt(e.target.value) })}
                          className="px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white focus:border-blue-500 focus:outline-none"
                        >
                          <option value={25}>25</option>
                          <option value={50}>50</option>
                          <option value={100}>100</option>
                          <option value={200}>200</option>
                        </select>
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* History List */}
            <div className="flex-1 overflow-y-auto max-h-96">
              {loading ? (
                <div className="flex items-center justify-center p-12">
                  <motion.div
                    className="w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full"
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  />
                </div>
              ) : filteredDetections.length === 0 ? (
                <div className="flex flex-col items-center justify-center p-12 text-gray-400">
                  <History className="w-16 h-16 mb-4 opacity-50" />
                  <p className="text-lg font-medium">No detections found</p>
                  <p className="text-sm">Try adjusting your filters or start detecting cards</p>
                </div>
              ) : (
                <div className="p-6">
                  <div className="grid gap-3">
                    <AnimatePresence>
                      {filteredDetections.map((detection, index) => {
                        const { date, time } = formatTimestamp(detection.timestamp);
                        return (
                          <motion.div
                            key={detection.id}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -20 }}
                            transition={{ delay: index * 0.05 }}
                            className="bg-gray-800 rounded-lg p-4 border border-gray-700 hover:border-gray-600 transition-colors"
                          >
                            <div className="flex items-center justify-between">
                              <div className="flex items-center space-x-4">
                                {/* Card Display */}
                                <div className="flex items-center space-x-2">
                                  <span className="text-2xl font-bold text-white">
                                    {formatRank(detection.rank)}
                                  </span>
                                  <span className={`text-2xl ${getSuitColor(detection.suit)}`}>
                                    {getSuitSymbol(detection.suit)}
                                  </span>
                                </div>

                                {/* Card Name */}
                                <div>
                                  <p className="text-white font-medium">
                                    {formatRank(detection.rank)} of {detection.suit}
                                  </p>
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
                </div>
              )}
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default HistoryPanel;