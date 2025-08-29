import { useState, useEffect, useCallback } from 'react';
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

interface CardFrequencyResponse {
  cards: CardFrequency[];
  total_unique_cards: number;
  most_frequent: string;
  least_frequent: string;
}

interface TimelineData {
  date: string;
  hour: number;
  count: number;
  avg_confidence: number;
}

interface UseDetectionStatsReturn {
  stats: DetectionStats | null;
  cardFrequency: CardFrequency[];
  timeline: TimelineData[];
  loading: boolean;
  error: string | null;
  fetchStats: (days?: number) => Promise<void>;
  fetchCardFrequency: (days?: number) => Promise<void>;
  fetchTimeline: (days?: number) => Promise<void>;
  refreshAll: (days?: number) => Promise<void>;
  getTopCards: (limit?: number) => CardFrequency[];
  getRecentActivity: (hours?: number) => TimelineData[];
  calculateTrends: () => {
    dailyTrend: number;
    weeklyTrend: number;
    confidenceTrend: number;
  };
}

const API_BASE_URL = 'http://localhost:8000';

export const useDetectionStats = (): UseDetectionStatsReturn => {
  const [stats, setStats] = useState<DetectionStats | null>(null);
  const [cardFrequency, setCardFrequency] = useState<CardFrequency[]>([]);
  const [timeline, setTimeline] = useState<TimelineData[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastDays, setLastDays] = useState<number | undefined>();

  const fetchStats = useCallback(async (days?: number) => {
    setError(null);
    
    try {
      const params = days ? { days } : {};
      const response = await axios.get<DetectionStats>(`${API_BASE_URL}/api/stats`, {
        params,
        timeout: 10000
      });
      
      setStats(response.data);
      setLastDays(days);
    } catch (err) {
      console.error('Failed to fetch detection stats:', err);
      if (axios.isAxiosError(err)) {
        if (err.code === 'ECONNABORTED') {
          setError('Request timeout. Please check your connection.');
        } else if (err.response?.status === 404) {
          setError('Stats endpoint not found. Please check the server.');
        } else if (err.response && err.response.status >= 500) {
          setError('Server error. Please try again later.');
        } else {
          setError(err.response?.data?.message || 'Failed to fetch statistics');
        }
      } else {
        setError('Network error. Please check your connection.');
      }
    }
  }, []);

  const fetchCardFrequency = useCallback(async (days?: number) => {
    setError(null);
    
    try {
      const params = days ? { days } : {};
      const response = await axios.get<CardFrequencyResponse>(`${API_BASE_URL}/api/card-frequency`, {
        params,
        timeout: 10000
      });
      
      setCardFrequency(response.data.cards || []);
    } catch (err) {
      console.error('Failed to fetch card frequency:', err);
      if (axios.isAxiosError(err)) {
        if (err.code === 'ECONNABORTED') {
          setError('Request timeout. Please check your connection.');
        } else {
          setError(err.response?.data?.message || 'Failed to fetch card frequency');
        }
      } else {
        setError('Network error. Please check your connection.');
      }
    }
  }, []);

  const fetchTimeline = useCallback(async (days?: number) => {
    setError(null);
    
    try {
      const params = days ? { days } : {};
      const response = await axios.get<{ timeline: TimelineData[] }>(`${API_BASE_URL}/api/timeline`, {
        params,
        timeout: 10000
      });
      
      setTimeline(response.data.timeline || []);
    } catch (err) {
      console.error('Failed to fetch timeline:', err);
      if (axios.isAxiosError(err)) {
        if (err.code === 'ECONNABORTED') {
          setError('Request timeout. Please check your connection.');
        } else if (err.response?.status === 404) {
          // Timeline endpoint might not exist, ignore error
          console.warn('Timeline endpoint not available');
        } else {
          setError(err.response?.data?.message || 'Failed to fetch timeline');
        }
      } else {
        setError('Network error. Please check your connection.');
      }
    }
  }, []);

  const refreshAll = useCallback(async (days?: number) => {
    setLoading(true);
    setError(null);
    
    try {
      await Promise.all([
        fetchStats(days),
        fetchCardFrequency(days),
        fetchTimeline(days)
      ]);
    } catch (err) {
      console.error('Failed to refresh all stats:', err);
    } finally {
      setLoading(false);
    }
  }, [fetchStats, fetchCardFrequency, fetchTimeline]);

  const getTopCards = useCallback((limit: number = 10): CardFrequency[] => {
    return cardFrequency
      .sort((a, b) => b.count - a.count)
      .slice(0, limit);
  }, [cardFrequency]);

  const getRecentActivity = useCallback((hours: number = 24): TimelineData[] => {
    const cutoffTime = new Date();
    cutoffTime.setHours(cutoffTime.getHours() - hours);
    
    return timeline.filter(item => {
      const itemDate = new Date(item.date);
      itemDate.setHours(item.hour);
      return itemDate >= cutoffTime;
    }).sort((a, b) => {
      const dateA = new Date(a.date);
      dateA.setHours(a.hour);
      const dateB = new Date(b.date);
      dateB.setHours(b.hour);
      return dateB.getTime() - dateA.getTime();
    });
  }, [timeline]);

  const calculateTrends = useCallback(() => {
    if (!stats || !stats.daily_stats || stats.daily_stats.length < 2) {
      return {
        dailyTrend: 0,
        weeklyTrend: 0,
        confidenceTrend: 0
      };
    }

    const dailyStats = stats.daily_stats.sort((a, b) => 
      new Date(a.date).getTime() - new Date(b.date).getTime()
    );

    // Calculate daily trend (last 2 days)
    const lastTwo = dailyStats.slice(-2);
    const dailyTrend = lastTwo.length === 2 
      ? ((lastTwo[1].count - lastTwo[0].count) / Math.max(lastTwo[0].count, 1)) * 100
      : 0;

    // Calculate weekly trend (last 7 days vs previous 7 days)
    let weeklyTrend = 0;
    if (dailyStats.length >= 14) {
      const lastWeek = dailyStats.slice(-7).reduce((sum, day) => sum + day.count, 0);
      const prevWeek = dailyStats.slice(-14, -7).reduce((sum, day) => sum + day.count, 0);
      weeklyTrend = ((lastWeek - prevWeek) / Math.max(prevWeek, 1)) * 100;
    }

    // Confidence trend (simplified - based on current avg vs target)
    const targetConfidence = 0.8;
    const confidenceTrend = ((stats.avg_confidence - targetConfidence) / targetConfidence) * 100;

    return {
      dailyTrend: Math.round(dailyTrend * 10) / 10,
      weeklyTrend: Math.round(weeklyTrend * 10) / 10,
      confidenceTrend: Math.round(confidenceTrend * 10) / 10
    };
  }, [stats]);

  // Auto-fetch initial stats
  useEffect(() => {
    refreshAll(30); // Default to last 30 days
  }, [refreshAll]);

  // Auto-refresh every 60 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      refreshAll(lastDays);
    }, 60000);
    
    return () => clearInterval(interval);
  }, [refreshAll, lastDays]);

  return {
    stats,
    cardFrequency,
    timeline,
    loading,
    error,
    fetchStats,
    fetchCardFrequency,
    fetchTimeline,
    refreshAll,
    getTopCards,
    getRecentActivity,
    calculateTrends
  };
};

export default useDetectionStats;