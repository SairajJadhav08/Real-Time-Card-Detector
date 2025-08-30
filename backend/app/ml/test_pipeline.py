#!/usr/bin/env python3
"""
Test script to verify the ML training pipeline components work correctly.
This script performs basic functionality tests without requiring actual training data.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test core ML imports
        import cv2
        import numpy as np
        from PIL import Image
        print("✓ OpenCV and PIL imported successfully")
        
        # Test ML training imports
        try:
            import torch
            import torchvision
            print(f"✓ PyTorch {torch.__version__} imported successfully")
        except ImportError:
            print("⚠ PyTorch not available - will use CPU training")
        
        try:
            from ultralytics import YOLO
            print("✓ YOLOv8 (ultralytics) imported successfully")
        except ImportError:
            print("✗ YOLOv8 not available - install with: pip install ultralytics")
            return False
        
        try:
            import albumentations as A
            print("✓ Albumentations imported successfully")
        except ImportError:
            print("✗ Albumentations not available - install with: pip install albumentations")
            return False
        
        # Test our custom modules
        from data_collector import CardDataCollector
        from data_augmentation import CardDataAugmentor
        from transfer_learning import CardDetectorYOLO
        from training_pipeline import CardDetectionTrainingPipeline
        print("✓ All custom modules imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_data_collector():
    """Test data collector initialization."""
    print("\nTesting data collector...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            collector = CardDataCollector(temp_dir)
            collector.setup_directories()
            
            # Check if directories were created
            expected_dirs = ['raw_images', 'processed_images', 'annotations', 'train', 'val', 'test', 'augmented']
            for dir_name in expected_dirs:
                dir_path = Path(temp_dir) / 'data' / dir_name
                if not dir_path.exists():
                    print(f"✗ Directory {dir_name} not created")
                    return False
            
            print("✓ Data collector setup successful")
            return True
            
    except Exception as e:
        print(f"✗ Data collector error: {e}")
        return False

def test_data_augmentor():
    """Test data augmentation setup."""
    print("\nTesting data augmentor...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            augmentor = CardDataAugmentor(temp_dir)
            
            # Test augmentation pipeline creation
            transform = augmentor._create_augmentation_pipeline('train')
            if transform is None:
                print("✗ Failed to create augmentation pipeline")
                return False
            
            print("✓ Data augmentor setup successful")
            return True
            
    except Exception as e:
        print(f"✗ Data augmentor error: {e}")
        return False

def test_yolo_detector():
    """Test YOLO detector initialization."""
    print("\nTesting YOLO detector...")
    
    try:
        detector = CardDetectorYOLO(model_size='n')  # Use nano for testing
        
        # Test model loading
        model = detector._load_pretrained_model()
        if model is None:
            print("✗ Failed to load pretrained model")
            return False
        
        print("✓ YOLO detector setup successful")
        return True
        
    except Exception as e:
        print(f"✗ YOLO detector error: {e}")
        return False

def test_training_pipeline():
    """Test training pipeline initialization."""
    print("\nTesting training pipeline...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = CardDetectionTrainingPipeline(temp_dir)
            
            # Test configuration loading
            if not hasattr(pipeline, 'config'):
                print("✗ Pipeline configuration not loaded")
                return False
            
            # Test directory setup
            pipeline._setup_directories()
            
            print("✓ Training pipeline setup successful")
            return True
            
    except Exception as e:
        print(f"✗ Training pipeline error: {e}")
        return False

def test_gpu_availability():
    """Test GPU availability for training."""
    print("\nTesting GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU available: {gpu_name} (Count: {gpu_count})")
            print(f"  CUDA Version: {torch.version.cuda}")
            return True
        else:
            print("⚠ No GPU available - training will use CPU (slower)")
            return True  # Not an error, just slower
            
    except ImportError:
        print("⚠ PyTorch not available - cannot check GPU")
        return True

def run_all_tests():
    """Run all tests and provide summary."""
    print("=" * 60)
    print("CARD DETECTION ML PIPELINE TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Data Collector", test_data_collector),
        ("Data Augmentor", test_data_augmentor),
        ("YOLO Detector", test_yolo_detector),
        ("Training Pipeline", test_training_pipeline),
        ("GPU Availability", test_gpu_availability)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your ML pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Run 'python training_pipeline.py' to start training")
        print("2. Follow the interactive prompts to collect and train")
        print("3. Check the README.md for detailed instructions")
    else:
        print("\n⚠ Some tests failed. Please check the error messages above.")
        print("\nCommon solutions:")
        print("1. Install missing dependencies: pip install -r ../../requirements.txt")
        print("2. Make sure you're in the correct directory")
        print("3. Check that Python version is 3.8+")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)