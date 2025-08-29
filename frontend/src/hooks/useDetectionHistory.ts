import { useState, useEffect, useCallback } from 'react';
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

interface DetectionHistory {
  detections: Detection[];
  total_count: number;
  filtered_count: number;
}

interface UseDetectionHistoryReturn {
  history: Detection[];
  totalCount: number;
  filteredCount: number;
  loading: boolean;
  error: string | null;
  fetchHistory: (filter?: HistoryFilter) => Promise<void>;
  addDetection: (detection: Omit<Detection, 'id' | 'timestamp'>) => void;
  clearHistory: () => Promise<void>;
  exportHistory: (format: 'json' | 'csv', filter?: HistoryFilter) => Promise<void>;
  refreshHistory: () => Promise<void>;
}

const API_BASE_URL = 'http://localhost:8000';

export const useDetectionHistory = (): UseDetectionHistoryReturn => {
  const [history, setHistory] = useState<Detection[]>([]);
  const [totalCount, setTotalCount] = useState(0);
  const [filteredCount, setFilteredCount] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastFilter, setLastFilter] = useState<HistoryFilter | undefined>();

  const fetchHistory = useCallback(async (filter?: HistoryFilter) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.get<DetectionHistory>(`${API_BASE_URL}/api/history`, {
        params: filter,
        timeout: 10000
      });
      
      const data = response.data;
      setHistory(data.detections || []);
      setTotalCount(data.total_count || 0);
      setFilteredCount(data.filtered_count || data.detections?.length || 0);
      setLastFilter(filter);
    } catch (err) {
      console.error('Failed to fetch detection history:', err);
      if (axios.isAxiosError(err)) {
        if (err.code === 'ECONNABORTED') {
          setError('Request timeout. Please check your connection.');
        } else if (err.response?.status === 404) {
          setError('History endpoint not found. Please check the server.');
        } else if (err.response && err.response.status >= 500) {
          setError('Server error. Please try again later.');
        } else {
          setError(err.response?.data?.message || 'Failed to fetch history');
        }
      } else {
        setError('Network error. Please check your connection.');
      }
    } finally {
      setLoading(false);
    }
  }, []);

  const addDetection = useCallback((detection: Omit<Detection, 'id' | 'timestamp'>) => {
    const newDetection: Detection = {
      ...detection,
      id: Date.now(), // Temporary ID
      timestamp: new Date().toISOString()
    };
    
    setHistory(prev => [newDetection, ...prev]);
    setTotalCount(prev => prev + 1);
    setFilteredCount(prev => prev + 1);
  }, []);

  const clearHistory = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      await axios.delete(`${API_BASE_URL}/api/history`, {
        timeout: 10000
      });
      
      setHistory([]);
      setTotalCount(0);
      setFilteredCount(0);
    } catch (err) {
      console.error('Failed to clear history:', err);
      if (axios.isAxiosError(err)) {
        if (err.code === 'ECONNABORTED') {
          setError('Request timeout. Please try again.');
        } else {
          setError(err.response?.data?.message || 'Failed to clear history');
        }
      } else {
        setError('Network error. Please check your connection.');
      }
    } finally {
      setLoading(false);
    }
  }, []);

  const exportHistory = useCallback(async (format: 'json' | 'csv', filter?: HistoryFilter) => {
    setError(null);
    
    try {
      const response = await axios.post(`${API_BASE_URL}/api/export`, {
        format,
        filter: filter || lastFilter
      }, {
        responseType: 'blob',
        timeout: 30000
      });
      
      // Create download link
      const blob = new Blob([response.data]);
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      
      // Generate filename with timestamp
      const timestamp = new Date().toISOString().split('T')[0];
      link.download = `card_detections_${timestamp}.${format}`;
      
      // Trigger download
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      // Cleanup
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Failed to export history:', err);
      if (axios.isAxiosError(err)) {
        if (err.code === 'ECONNABORTED') {
          setError('Export timeout. Please try again with fewer records.');
        } else {
          setError(err.response?.data?.message || 'Failed to export history');
        }
      } else {
        setError('Network error. Please check your connection.');
      }
    }
  }, [lastFilter]);

  const refreshHistory = useCallback(async () => {
    await fetchHistory(lastFilter);
  }, [fetchHistory, lastFilter]);

  // Auto-fetch initial history
  useEffect(() => {
    fetchHistory({ limit: 50 });
  }, [fetchHistory]);

  // Auto-refresh every 30 seconds if no filter is applied
  useEffect(() => {
    if (!lastFilter || Object.keys(lastFilter).length <= 1) {
      const interval = setInterval(() => {
        refreshHistory();
      }, 30000);
      
      return () => clearInterval(interval);
    }
  }, [refreshHistory, lastFilter]);

  return {
    history,
    totalCount,
    filteredCount,
    loading,
    error,
    fetchHistory,
    addDetection,
    clearHistory,
    exportHistory,
    refreshHistory
  };
};

export default useDetectionHistory;