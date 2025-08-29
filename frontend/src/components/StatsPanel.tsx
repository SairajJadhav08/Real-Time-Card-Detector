import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { BarChart3, TrendingUp, Target, Clock, Calendar, X, RefreshCw } from 'lucide-react';
import axios from 'axios';

interface DetectionStats {
  total_detections: number;
  unique_cards: number;
  avg_confidence: number;
  detection_rate: number;
  most_detected_card: string;
  least_detected_card: string;
  daily_stats: Array<{
    date: string;
    count: number;
  }>;
}

interface CardFrequency {
  card_name: string;
  count: number;
  percentage: number;
  avg_confidence: number;
}

interface StatsPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

const StatsPanel: React.FC<StatsPanelProps> = ({ isOpen, onClose }) => {
  const [stats, setStats] = useState<DetectionStats | null>(null);
  const [cardFrequency, setCardFrequency] = useState<CardFrequency[]>([]);
  const [loading, setLoading] = useState(false);
  const [timeRange, setTimeRange] = useState<'7d' | '30d' | '90d' | 'all'>('30d');

  const API_BASE_URL = 'http://localhost:8000';

  const fetchStats = useCallback(async () => {
    setLoading(true);
    try {
      const [statsResponse, frequencyResponse] = await Promise.all([
        axios.get(`${API_BASE_URL}/api/stats`, {
          params: { days: timeRange === 'all' ? undefined : parseInt(timeRange) }
        }),
        axios.get(`${API_BASE_URL}/api/card-frequency`, {
          params: { days: timeRange === 'all' ? undefined : parseInt(timeRange) }
        })
      ]);
      
      setStats(statsResponse.data);
      setCardFrequency(frequencyResponse.data.cards || []);
    } catch (error) {
      console.error('Failed to fetch stats:', error);
    } finally {
      setLoading(false);
    }
  }, [timeRange]);

  useEffect(() => {
    if (isOpen) {
      fetchStats();
    }
  }, [isOpen, timeRange, fetchStats]);

  const getSuitColor = (cardName: string) => {
    if (cardName.includes('hearts') || cardName.includes('diamonds')) {
      return 'text-red-400';
    }
    if (cardName.includes('clubs') || cardName.includes('spades')) {
      return 'text-gray-800';
    }
    return 'text-blue-400';
  };

  const getSuitSymbol = (cardName: string) => {
    if (cardName.includes('hearts')) return '♥';
    if (cardName.includes('diamonds')) return '♦';
    if (cardName.includes('clubs')) return '♣';
    if (cardName.includes('spades')) return '♠';
    return '?';
  };

  const formatCardName = (cardName: string) => {
    const parts = cardName.split(' of ');
    if (parts.length === 2) {
      const rank = parts[0].toUpperCase();
      const suit = parts[1].toLowerCase();
      return { rank, suit };
    }
    return { rank: cardName, suit: '' };
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-400';
    if (confidence >= 0.6) return 'text-yellow-400';
    return 'text-red-400';
  };

  const timeRangeOptions = [
    { value: '7d', label: 'Last 7 days' },
    { value: '30d', label: 'Last 30 days' },
    { value: '90d', label: 'Last 90 days' },
    { value: 'all', label: 'All time' }
  ];

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
            className="bg-gray-900 rounded-xl shadow-2xl w-full max-w-6xl max-h-[90vh] overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="bg-gradient-to-r from-purple-600 to-pink-600 p-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <BarChart3 className="w-6 h-6 text-white" />
                  <h2 className="text-2xl font-bold text-white">Detection Analytics</h2>
                </div>
                <div className="flex items-center space-x-4">
                  <select
                    value={timeRange}
                    onChange={(e) => setTimeRange(e.target.value as any)}
                    className="bg-white/20 text-white border border-white/30 rounded-lg px-3 py-2 focus:outline-none focus:border-white/50"
                  >
                    {timeRangeOptions.map(option => (
                      <option key={option.value} value={option.value} className="bg-gray-800">
                        {option.label}
                      </option>
                    ))}
                  </select>
                  <button
                    onClick={fetchStats}
                    disabled={loading}
                    className="text-white hover:bg-white/20 p-2 rounded-lg transition-colors disabled:opacity-50"
                  >
                    <RefreshCw className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
                  </button>
                  <button
                    onClick={onClose}
                    className="text-white hover:bg-white/20 p-2 rounded-lg transition-colors"
                  >
                    <X className="w-5 h-5" />
                  </button>
                </div>
              </div>
            </div>

            <div className="overflow-y-auto max-h-[calc(90vh-120px)]">
              {loading ? (
                <div className="flex items-center justify-center p-12">
                  <motion.div
                    className="w-8 h-8 border-4 border-purple-600 border-t-transparent rounded-full"
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  />
                </div>
              ) : stats ? (
                <div className="p-6 space-y-6">
                  {/* Overview Stats */}
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.1 }}
                      className="bg-gradient-to-br from-blue-600 to-blue-700 rounded-xl p-6 text-white"
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-blue-100 text-sm font-medium">Total Detections</p>
                          <p className="text-3xl font-bold">{stats.total_detections.toLocaleString()}</p>
                        </div>
                        <Target className="w-8 h-8 text-blue-200" />
                      </div>
                    </motion.div>

                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.2 }}
                      className="bg-gradient-to-br from-green-600 to-green-700 rounded-xl p-6 text-white"
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-green-100 text-sm font-medium">Unique Cards</p>
                          <p className="text-3xl font-bold">{stats.unique_cards}</p>
                        </div>
                        <BarChart3 className="w-8 h-8 text-green-200" />
                      </div>
                    </motion.div>

                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.3 }}
                      className="bg-gradient-to-br from-yellow-600 to-yellow-700 rounded-xl p-6 text-white"
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-yellow-100 text-sm font-medium">Avg Confidence</p>
                          <p className="text-3xl font-bold">{(stats.avg_confidence * 100).toFixed(1)}%</p>
                        </div>
                        <TrendingUp className="w-8 h-8 text-yellow-200" />
                      </div>
                    </motion.div>

                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.4 }}
                      className="bg-gradient-to-br from-purple-600 to-purple-700 rounded-xl p-6 text-white"
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-purple-100 text-sm font-medium">Detection Rate</p>
                          <p className="text-3xl font-bold">{stats.detection_rate.toFixed(1)}/min</p>
                        </div>
                        <Clock className="w-8 h-8 text-purple-200" />
                      </div>
                    </motion.div>
                  </div>

                  {/* Daily Activity Chart */}
                  {stats.daily_stats && stats.daily_stats.length > 0 && (
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.5 }}
                      className="bg-gray-800 rounded-xl p-6"
                    >
                      <div className="flex items-center space-x-2 mb-6">
                        <Calendar className="w-5 h-5 text-gray-400" />
                        <h3 className="text-lg font-semibold text-white">Daily Activity</h3>
                      </div>
                      
                      <div className="space-y-3">
                        {stats.daily_stats.map((day, index) => {
                          const maxCount = Math.max(...stats.daily_stats.map(d => d.count));
                          const percentage = maxCount > 0 ? (day.count / maxCount) * 100 : 0;
                          
                          return (
                            <motion.div
                              key={day.date}
                              initial={{ opacity: 0, x: -20 }}
                              animate={{ opacity: 1, x: 0 }}
                              transition={{ delay: 0.6 + index * 0.1 }}
                              className="flex items-center space-x-4"
                            >
                              <div className="w-20 text-sm text-gray-400">
                                {new Date(day.date).toLocaleDateString('en-US', { 
                                  month: 'short', 
                                  day: 'numeric' 
                                })}
                              </div>
                              <div className="flex-1 bg-gray-700 rounded-full h-6 overflow-hidden">
                                <motion.div
                                  className="h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-end pr-2"
                                  initial={{ width: 0 }}
                                  animate={{ width: `${percentage}%` }}
                                  transition={{ duration: 0.8, delay: 0.8 + index * 0.1 }}
                                >
                                  {day.count > 0 && (
                                    <span className="text-white text-xs font-medium">
                                      {day.count}
                                    </span>
                                  )}
                                </motion.div>
                              </div>
                            </motion.div>
                          );
                        })}
                      </div>
                    </motion.div>
                  )}

                  {/* Card Frequency */}
                  {cardFrequency.length > 0 && (
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.6 }}
                      className="bg-gray-800 rounded-xl p-6"
                    >
                      <div className="flex items-center space-x-2 mb-6">
                        <BarChart3 className="w-5 h-5 text-gray-400" />
                        <h3 className="text-lg font-semibold text-white">Most Detected Cards</h3>
                      </div>
                      
                      <div className="grid gap-3">
                        {cardFrequency.slice(0, 10).map((card, index) => {
                          const { rank } = formatCardName(card.card_name);
                          const maxCount = cardFrequency[0]?.count || 1;
                          const percentage = (card.count / maxCount) * 100;
                          
                          return (
                            <motion.div
                              key={card.card_name}
                              initial={{ opacity: 0, x: -20 }}
                              animate={{ opacity: 1, x: 0 }}
                              transition={{ delay: 0.7 + index * 0.05 }}
                              className="bg-gray-700 rounded-lg p-4"
                            >
                              <div className="flex items-center justify-between mb-2">
                                <div className="flex items-center space-x-3">
                                  <div className="flex items-center space-x-1">
                                    <span className="text-xl font-bold text-white">{rank}</span>
                                    <span className={`text-xl ${getSuitColor(card.card_name)}`}>
                                      {getSuitSymbol(card.card_name)}
                                    </span>
                                  </div>
                                  <span className="text-gray-300 capitalize">{card.card_name}</span>
                                </div>
                                <div className="text-right">
                                  <div className="text-white font-semibold">{card.count} times</div>
                                  <div className={`text-sm ${getConfidenceColor(card.avg_confidence)}`}>
                                    {(card.avg_confidence * 100).toFixed(1)}% avg
                                  </div>
                                </div>
                              </div>
                              
                              <div className="bg-gray-600 rounded-full h-2 overflow-hidden">
                                <motion.div
                                  className="h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full"
                                  initial={{ width: 0 }}
                                  animate={{ width: `${percentage}%` }}
                                  transition={{ duration: 0.8, delay: 0.8 + index * 0.05 }}
                                />
                              </div>
                              
                              <div className="mt-1 text-xs text-gray-400">
                                {card.percentage.toFixed(1)}% of all detections
                              </div>
                            </motion.div>
                          );
                        })}
                      </div>
                    </motion.div>
                  )}

                  {/* Performance Insights */}
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.7 }}
                    className="bg-gray-800 rounded-xl p-6"
                  >
                    <div className="flex items-center space-x-2 mb-6">
                      <TrendingUp className="w-5 h-5 text-gray-400" />
                      <h3 className="text-lg font-semibold text-white">Performance Insights</h3>
                    </div>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="space-y-4">
                        <div className="bg-gray-700 rounded-lg p-4">
                          <h4 className="text-white font-medium mb-2">Most Detected Card</h4>
                          <p className="text-green-400 text-lg font-semibold capitalize">
                            {stats.most_detected_card || 'N/A'}
                          </p>
                        </div>
                        
                        <div className="bg-gray-700 rounded-lg p-4">
                          <h4 className="text-white font-medium mb-2">Least Detected Card</h4>
                          <p className="text-yellow-400 text-lg font-semibold capitalize">
                            {stats.least_detected_card || 'N/A'}
                          </p>
                        </div>
                      </div>
                      
                      <div className="space-y-4">
                        <div className="bg-gray-700 rounded-lg p-4">
                          <h4 className="text-white font-medium mb-2">Detection Quality</h4>
                          <div className="flex items-center space-x-2">
                            <div className={`w-3 h-3 rounded-full ${
                              stats.avg_confidence >= 0.8 ? 'bg-green-400' :
                              stats.avg_confidence >= 0.6 ? 'bg-yellow-400' : 'bg-red-400'
                            }`} />
                            <span className="text-white">
                              {stats.avg_confidence >= 0.8 ? 'Excellent' :
                               stats.avg_confidence >= 0.6 ? 'Good' : 'Needs Improvement'}
                            </span>
                          </div>
                        </div>
                        
                        <div className="bg-gray-700 rounded-lg p-4">
                          <h4 className="text-white font-medium mb-2">Activity Level</h4>
                          <div className="flex items-center space-x-2">
                            <div className={`w-3 h-3 rounded-full ${
                              stats.detection_rate >= 5 ? 'bg-green-400' :
                              stats.detection_rate >= 1 ? 'bg-yellow-400' : 'bg-red-400'
                            }`} />
                            <span className="text-white">
                              {stats.detection_rate >= 5 ? 'Very Active' :
                               stats.detection_rate >= 1 ? 'Moderate' : 'Low Activity'}
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center p-12 text-gray-400">
                  <BarChart3 className="w-16 h-16 mb-4 opacity-50" />
                  <p className="text-lg font-medium">No statistics available</p>
                  <p className="text-sm">Start detecting cards to see analytics</p>
                </div>
              )}
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default StatsPanel;