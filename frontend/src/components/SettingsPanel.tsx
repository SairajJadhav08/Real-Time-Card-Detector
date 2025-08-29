import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Settings, Save, RotateCcw, X, Camera, Zap, Target, Palette, Volume2 } from 'lucide-react';
import axios from 'axios';

interface ModelInfo {
  model_type: string;
  model_path: string;
  confidence_threshold: number;
  nms_threshold: number;
  input_size: [number, number];
  classes: string[];
  performance: {
    avg_inference_time: number;
    total_detections: number;
    success_rate: number;
  };
}

interface AppSettings {
  theme: 'dark' | 'light';
  autoDetect: boolean;
  detectionInterval: number;
  confidenceThreshold: number;
  showConfidence: boolean;
  soundEnabled: boolean;
  animationsEnabled: boolean;
  maxHistory: number;
  cameraResolution: string;
  frameRate: number;
}

interface SettingsPanelProps {
  isOpen: boolean;
  onClose: () => void;
  settings: AppSettings;
  onSettingsChange: (settings: AppSettings) => void;
}

const SettingsPanel: React.FC<SettingsPanelProps> = ({ 
  isOpen, 
  onClose, 
  settings, 
  onSettingsChange 
}) => {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [localSettings, setLocalSettings] = useState<AppSettings>(settings);
  const [activeTab, setActiveTab] = useState<'general' | 'detection' | 'camera' | 'model'>('general');

  const API_BASE_URL = 'http://localhost:8000';

  useEffect(() => {
    if (isOpen) {
      fetchModelInfo();
      setLocalSettings(settings);
    }
  }, [isOpen, settings]);

  const fetchModelInfo = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API_BASE_URL}/api/model-info`);
      setModelInfo(response.data);
    } catch (error) {
      console.error('Failed to fetch model info:', error);
    } finally {
      setLoading(false);
    }
  };

  const saveSettings = async () => {
    setSaving(true);
    try {
      // Update backend configuration if needed
      await axios.post(`${API_BASE_URL}/api/config`, {
        confidence_threshold: localSettings.confidenceThreshold,
        detection_interval: localSettings.detectionInterval
      });
      
      // Update local settings
      onSettingsChange(localSettings);
      
      // Save to localStorage
      localStorage.setItem('cardDetectorSettings', JSON.stringify(localSettings));
      
      onClose();
    } catch (error) {
      console.error('Failed to save settings:', error);
    } finally {
      setSaving(false);
    }
  };

  const resetSettings = () => {
    const defaultSettings: AppSettings = {
      theme: 'dark',
      autoDetect: true,
      detectionInterval: 100,
      confidenceThreshold: 0.5,
      showConfidence: true,
      soundEnabled: true,
      animationsEnabled: true,
      maxHistory: 1000,
      cameraResolution: '1280x720',
      frameRate: 30
    };
    setLocalSettings(defaultSettings);
  };

  const updateSetting = <K extends keyof AppSettings>(key: K, value: AppSettings[K]) => {
    setLocalSettings(prev => ({ ...prev, [key]: value }));
  };

  const tabs = [
    { id: 'general', label: 'General', icon: Settings },
    { id: 'detection', label: 'Detection', icon: Target },
    { id: 'camera', label: 'Camera', icon: Camera },
    { id: 'model', label: 'Model', icon: Zap }
  ];

  const resolutionOptions = [
    { value: '640x480', label: '640×480 (VGA)' },
    { value: '1280x720', label: '1280×720 (HD)' },
    { value: '1920x1080', label: '1920×1080 (Full HD)' },
    { value: '2560x1440', label: '2560×1440 (2K)' }
  ];

  const frameRateOptions = [15, 24, 30, 60];

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
            <div className="bg-gradient-to-r from-indigo-600 to-blue-600 p-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <Settings className="w-6 h-6 text-white" />
                  <h2 className="text-2xl font-bold text-white">Settings</h2>
                </div>
                <div className="flex items-center space-x-2">
                  <button
                    onClick={resetSettings}
                    className="flex items-center space-x-2 px-4 py-2 bg-white/20 hover:bg-white/30 text-white rounded-lg transition-colors"
                  >
                    <RotateCcw className="w-4 h-4" />
                    <span>Reset</span>
                  </button>
                  <button
                    onClick={saveSettings}
                    disabled={saving}
                    className="flex items-center space-x-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors disabled:opacity-50"
                  >
                    <Save className="w-4 h-4" />
                    <span>{saving ? 'Saving...' : 'Save'}</span>
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

            <div className="flex h-[calc(90vh-120px)]">
              {/* Sidebar */}
              <div className="w-64 bg-gray-800 border-r border-gray-700">
                <div className="p-4">
                  <nav className="space-y-2">
                    {tabs.map((tab) => {
                      const Icon = tab.icon;
                      return (
                        <button
                          key={tab.id}
                          onClick={() => setActiveTab(tab.id as any)}
                          className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors ${
                            activeTab === tab.id
                              ? 'bg-blue-600 text-white'
                              : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                          }`}
                        >
                          <Icon className="w-5 h-5" />
                          <span className="font-medium">{tab.label}</span>
                        </button>
                      );
                    })}
                  </nav>
                </div>
              </div>

              {/* Content */}
              <div className="flex-1 overflow-y-auto">
                <div className="p-6">
                  {activeTab === 'general' && (
                    <motion.div
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      className="space-y-6"
                    >
                      <h3 className="text-xl font-semibold text-white mb-4">General Settings</h3>
                      
                      {/* Theme */}
                      <div className="bg-gray-800 rounded-lg p-4">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-3">
                            <Palette className="w-5 h-5 text-gray-400" />
                            <div>
                              <h4 className="text-white font-medium">Theme</h4>
                              <p className="text-gray-400 text-sm">Choose your preferred theme</p>
                            </div>
                          </div>
                          <select
                            value={localSettings.theme}
                            onChange={(e) => updateSetting('theme', e.target.value as 'dark' | 'light')}
                            className="bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white focus:border-blue-500 focus:outline-none"
                          >
                            <option value="dark">Dark</option>
                            <option value="light">Light</option>
                          </select>
                        </div>
                      </div>

                      {/* Sound */}
                      <div className="bg-gray-800 rounded-lg p-4">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-3">
                            <Volume2 className="w-5 h-5 text-gray-400" />
                            <div>
                              <h4 className="text-white font-medium">Sound Effects</h4>
                              <p className="text-gray-400 text-sm">Play sounds on detection</p>
                            </div>
                          </div>
                          <label className="relative inline-flex items-center cursor-pointer">
                            <input
                              type="checkbox"
                              checked={localSettings.soundEnabled}
                              onChange={(e) => updateSetting('soundEnabled', e.target.checked)}
                              className="sr-only peer"
                            />
                            <div className="w-11 h-6 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                          </label>
                        </div>
                      </div>

                      {/* Animations */}
                      <div className="bg-gray-800 rounded-lg p-4">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-3">
                            <motion.div
                              animate={{ rotate: 360 }}
                              transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                            >
                              <Zap className="w-5 h-5 text-gray-400" />
                            </motion.div>
                            <div>
                              <h4 className="text-white font-medium">Animations</h4>
                              <p className="text-gray-400 text-sm">Enable UI animations</p>
                            </div>
                          </div>
                          <label className="relative inline-flex items-center cursor-pointer">
                            <input
                              type="checkbox"
                              checked={localSettings.animationsEnabled}
                              onChange={(e) => updateSetting('animationsEnabled', e.target.checked)}
                              className="sr-only peer"
                            />
                            <div className="w-11 h-6 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                          </label>
                        </div>
                      </div>

                      {/* Max History */}
                      <div className="bg-gray-800 rounded-lg p-4">
                        <div>
                          <h4 className="text-white font-medium mb-2">Maximum History Records</h4>
                          <p className="text-gray-400 text-sm mb-4">Limit the number of stored detections</p>
                          <input
                            type="range"
                            min="100"
                            max="5000"
                            step="100"
                            value={localSettings.maxHistory}
                            onChange={(e) => updateSetting('maxHistory', parseInt(e.target.value))}
                            className="w-full"
                          />
                          <div className="flex justify-between text-sm text-gray-400 mt-2">
                            <span>100</span>
                            <span className="text-white font-medium">{localSettings.maxHistory}</span>
                            <span>5000</span>
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  )}

                  {activeTab === 'detection' && (
                    <motion.div
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      className="space-y-6"
                    >
                      <h3 className="text-xl font-semibold text-white mb-4">Detection Settings</h3>
                      
                      {/* Auto Detect */}
                      <div className="bg-gray-800 rounded-lg p-4">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-3">
                            <Target className="w-5 h-5 text-gray-400" />
                            <div>
                              <h4 className="text-white font-medium">Auto Detection</h4>
                              <p className="text-gray-400 text-sm">Automatically detect cards in real-time</p>
                            </div>
                          </div>
                          <label className="relative inline-flex items-center cursor-pointer">
                            <input
                              type="checkbox"
                              checked={localSettings.autoDetect}
                              onChange={(e) => updateSetting('autoDetect', e.target.checked)}
                              className="sr-only peer"
                            />
                            <div className="w-11 h-6 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                          </label>
                        </div>
                      </div>

                      {/* Detection Interval */}
                      <div className="bg-gray-800 rounded-lg p-4">
                        <div>
                          <h4 className="text-white font-medium mb-2">Detection Interval</h4>
                          <p className="text-gray-400 text-sm mb-4">Time between detections (milliseconds)</p>
                          <input
                            type="range"
                            min="50"
                            max="1000"
                            step="50"
                            value={localSettings.detectionInterval}
                            onChange={(e) => updateSetting('detectionInterval', parseInt(e.target.value))}
                            className="w-full"
                          />
                          <div className="flex justify-between text-sm text-gray-400 mt-2">
                            <span>50ms (Fast)</span>
                            <span className="text-white font-medium">{localSettings.detectionInterval}ms</span>
                            <span>1000ms (Slow)</span>
                          </div>
                        </div>
                      </div>

                      {/* Confidence Threshold */}
                      <div className="bg-gray-800 rounded-lg p-4">
                        <div>
                          <h4 className="text-white font-medium mb-2">Confidence Threshold</h4>
                          <p className="text-gray-400 text-sm mb-4">Minimum confidence for valid detections</p>
                          <input
                            type="range"
                            min="0.1"
                            max="0.9"
                            step="0.05"
                            value={localSettings.confidenceThreshold}
                            onChange={(e) => updateSetting('confidenceThreshold', parseFloat(e.target.value))}
                            className="w-full"
                          />
                          <div className="flex justify-between text-sm text-gray-400 mt-2">
                            <span>10% (Permissive)</span>
                            <span className="text-white font-medium">{(localSettings.confidenceThreshold * 100).toFixed(0)}%</span>
                            <span>90% (Strict)</span>
                          </div>
                        </div>
                      </div>

                      {/* Show Confidence */}
                      <div className="bg-gray-800 rounded-lg p-4">
                        <div className="flex items-center justify-between">
                          <div>
                            <h4 className="text-white font-medium">Show Confidence Scores</h4>
                            <p className="text-gray-400 text-sm">Display confidence percentages on detections</p>
                          </div>
                          <label className="relative inline-flex items-center cursor-pointer">
                            <input
                              type="checkbox"
                              checked={localSettings.showConfidence}
                              onChange={(e) => updateSetting('showConfidence', e.target.checked)}
                              className="sr-only peer"
                            />
                            <div className="w-11 h-6 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                          </label>
                        </div>
                      </div>
                    </motion.div>
                  )}

                  {activeTab === 'camera' && (
                    <motion.div
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      className="space-y-6"
                    >
                      <h3 className="text-xl font-semibold text-white mb-4">Camera Settings</h3>
                      
                      {/* Resolution */}
                      <div className="bg-gray-800 rounded-lg p-4">
                        <div>
                          <h4 className="text-white font-medium mb-2">Camera Resolution</h4>
                          <p className="text-gray-400 text-sm mb-4">Higher resolution improves accuracy but reduces performance</p>
                          <select
                            value={localSettings.cameraResolution}
                            onChange={(e) => updateSetting('cameraResolution', e.target.value)}
                            className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white focus:border-blue-500 focus:outline-none"
                          >
                            {resolutionOptions.map(option => (
                              <option key={option.value} value={option.value}>
                                {option.label}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>

                      {/* Frame Rate */}
                      <div className="bg-gray-800 rounded-lg p-4">
                        <div>
                          <h4 className="text-white font-medium mb-2">Frame Rate</h4>
                          <p className="text-gray-400 text-sm mb-4">Frames per second for camera capture</p>
                          <div className="grid grid-cols-4 gap-2">
                            {frameRateOptions.map(fps => (
                              <button
                                key={fps}
                                onClick={() => updateSetting('frameRate', fps)}
                                className={`px-4 py-2 rounded transition-colors ${
                                  localSettings.frameRate === fps
                                    ? 'bg-blue-600 text-white'
                                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                                }`}
                              >
                                {fps} FPS
                              </button>
                            ))}
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  )}

                  {activeTab === 'model' && (
                    <motion.div
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      className="space-y-6"
                    >
                      <h3 className="text-xl font-semibold text-white mb-4">Model Information</h3>
                      
                      {loading ? (
                        <div className="flex items-center justify-center p-12">
                          <motion.div
                            className="w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full"
                            animate={{ rotate: 360 }}
                            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                          />
                        </div>
                      ) : modelInfo ? (
                        <div className="space-y-4">
                          <div className="bg-gray-800 rounded-lg p-4">
                            <h4 className="text-white font-medium mb-3">Model Details</h4>
                            <div className="grid grid-cols-2 gap-4 text-sm">
                              <div>
                                <span className="text-gray-400">Type:</span>
                                <span className="text-white ml-2">{modelInfo.model_type}</span>
                              </div>
                              <div>
                                <span className="text-gray-400">Path:</span>
                                <span className="text-white ml-2 font-mono text-xs">{modelInfo.model_path}</span>
                              </div>
                              <div>
                                <span className="text-gray-400">Input Size:</span>
                                <span className="text-white ml-2">{modelInfo.input_size.join('×')}</span>
                              </div>
                              <div>
                                <span className="text-gray-400">Classes:</span>
                                <span className="text-white ml-2">{modelInfo.classes.length}</span>
                              </div>
                            </div>
                          </div>

                          <div className="bg-gray-800 rounded-lg p-4">
                            <h4 className="text-white font-medium mb-3">Performance Metrics</h4>
                            <div className="grid grid-cols-3 gap-4">
                              <div className="text-center">
                                <div className="text-2xl font-bold text-blue-400">
                                  {modelInfo.performance.avg_inference_time.toFixed(1)}ms
                                </div>
                                <div className="text-sm text-gray-400">Avg Inference</div>
                              </div>
                              <div className="text-center">
                                <div className="text-2xl font-bold text-green-400">
                                  {modelInfo.performance.total_detections.toLocaleString()}
                                </div>
                                <div className="text-sm text-gray-400">Total Detections</div>
                              </div>
                              <div className="text-center">
                                <div className="text-2xl font-bold text-yellow-400">
                                  {(modelInfo.performance.success_rate * 100).toFixed(1)}%
                                </div>
                                <div className="text-sm text-gray-400">Success Rate</div>
                              </div>
                            </div>
                          </div>

                          <div className="bg-gray-800 rounded-lg p-4">
                            <h4 className="text-white font-medium mb-3">Supported Cards</h4>
                            <div className="max-h-40 overflow-y-auto">
                              <div className="grid grid-cols-4 gap-2 text-sm">
                                {modelInfo.classes.map((cardClass, index) => (
                                  <div key={index} className="bg-gray-700 rounded px-2 py-1 text-gray-300">
                                    {cardClass}
                                  </div>
                                ))}
                              </div>
                            </div>
                          </div>
                        </div>
                      ) : (
                        <div className="bg-gray-800 rounded-lg p-8 text-center">
                          <Zap className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                          <p className="text-gray-400">Model information not available</p>
                        </div>
                      )}
                    </motion.div>
                  )}
                </div>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default SettingsPanel;