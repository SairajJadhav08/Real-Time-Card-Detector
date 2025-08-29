import React from 'react';
import { motion } from 'framer-motion';

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg' | 'xl';
  color?: 'blue' | 'purple' | 'green' | 'red' | 'yellow' | 'white';
  text?: string;
  className?: string;
}

const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({ 
  size = 'md', 
  color = 'blue', 
  text,
  className = '' 
}) => {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-8 h-8',
    lg: 'w-12 h-12',
    xl: 'w-16 h-16'
  };

  const colorClasses = {
    blue: 'border-blue-600 border-t-blue-200',
    purple: 'border-purple-600 border-t-purple-200',
    green: 'border-green-600 border-t-green-200',
    red: 'border-red-600 border-t-red-200',
    yellow: 'border-yellow-600 border-t-yellow-200',
    white: 'border-white border-t-gray-300'
  };

  const textColorClasses = {
    blue: 'text-blue-600',
    purple: 'text-purple-600',
    green: 'text-green-600',
    red: 'text-red-600',
    yellow: 'text-yellow-600',
    white: 'text-white'
  };

  return (
    <div className={`flex flex-col items-center justify-center ${className}`}>
      <motion.div
        className={`${sizeClasses[size]} border-4 ${colorClasses[color]} border-solid rounded-full`}
        animate={{ rotate: 360 }}
        transition={{
          duration: 1,
          repeat: Infinity,
          ease: 'linear'
        }}
      />
      {text && (
        <motion.p
          className={`mt-3 text-sm font-medium ${textColorClasses[color]}`}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
        >
          {text}
        </motion.p>
      )}
    </div>
  );
};

// Preset spinner components for common use cases
export const LoadingSpinnerSmall: React.FC<{ className?: string }> = ({ className }) => (
  <LoadingSpinner size="sm" className={className} />
);

export const LoadingSpinnerLarge: React.FC<{ text?: string; className?: string }> = ({ text, className }) => (
  <LoadingSpinner size="lg" text={text} className={className} />
);

export const LoadingSpinnerFullscreen: React.FC<{ text?: string }> = ({ text = 'Loading...' }) => (
  <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
    <div className="bg-gray-800 rounded-lg p-8 shadow-2xl">
      <LoadingSpinner size="xl" color="white" text={text} />
    </div>
  </div>
);

// Inline spinner for buttons
export const ButtonSpinner: React.FC<{ className?: string }> = ({ className = '' }) => (
  <motion.div
    className={`w-4 h-4 border-2 border-white border-t-transparent rounded-full ${className}`}
    animate={{ rotate: 360 }}
    transition={{
      duration: 1,
      repeat: Infinity,
      ease: 'linear'
    }}
  />
);

// Dots loading animation
export const LoadingDots: React.FC<{ color?: string; className?: string }> = ({ 
  color = 'text-blue-600', 
  className = '' 
}) => {
  return (
    <div className={`flex space-x-1 ${className}`}>
      {[0, 1, 2].map((index) => (
        <motion.div
          key={index}
          className={`w-2 h-2 rounded-full ${color.startsWith('text-') ? `bg-${color.slice(5)}` : 'bg-blue-600'}`}
          animate={{
            scale: [1, 1.2, 1],
            opacity: [0.7, 1, 0.7]
          }}
          transition={{
            duration: 0.6,
            repeat: Infinity,
            delay: index * 0.2
          }}
        />
      ))}
    </div>
  );
};

// Pulse loading animation
export const LoadingPulse: React.FC<{ className?: string }> = ({ className = '' }) => (
  <motion.div
    className={`w-8 h-8 bg-blue-600 rounded-full ${className}`}
    animate={{
      scale: [1, 1.2, 1],
      opacity: [0.7, 1, 0.7]
    }}
    transition={{
      duration: 1.5,
      repeat: Infinity,
      ease: 'easeInOut'
    }}
  />
);

export default LoadingSpinner;