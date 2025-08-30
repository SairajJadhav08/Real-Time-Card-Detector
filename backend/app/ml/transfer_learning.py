import torch
import torch.nn as nn
from ultralytics import YOLO
import cv2
import numpy as np
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import yaml
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

class CardDetectorYOLO:
    """Transfer learning implementation using YOLOv8 for card detection"""
    
    def __init__(self, model_size: str = 'n', data_dir: str = "../../data/augmented"):
        """
        Initialize YOLO-based card detector
        
        Args:
            model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
                       'n' = nano (fastest, least accurate)
                       's' = small
                       'm' = medium  
                       'l' = large
                       'x' = extra large (slowest, most accurate)
            data_dir: Directory containing augmented dataset
        """
        self.model_size = model_size
        self.data_dir = Path(data_dir)
        self.model = None
        self.class_names = []
        self.training_results = {}
        
        # Training hyperparameters
        self.hyperparams = {
            'epochs': 100,
            'batch_size': 16,
            'imgsz': 640,
            'lr0': 0.01,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'patience': 50,  # Early stopping patience
            'save_period': 10,  # Save checkpoint every N epochs
        }
        
    def setup_dataset_config(self) -> str:
        """Create YOLO dataset configuration file"""
        
        # Load class names
        classes_file = self.data_dir / "classes.txt"
        if not classes_file.exists():
            raise FileNotFoundError(f"Classes file not found: {classes_file}")
            
        with open(classes_file, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
            
        # Create dataset config
        config = {
            'path': str(self.data_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images', 
            'test': 'test/images',
            'nc': len(self.class_names),
            'names': {i: name for i, name in enumerate(self.class_names)}
        }
        
        # Save config file
        config_file = self.data_dir / "dataset.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        print(f"Dataset config created: {config_file}")
        print(f"Classes: {len(self.class_names)}")
        
        return str(config_file)
        
    def load_pretrained_model(self):
        """Load pre-trained YOLOv8 model"""
        
        model_name = f"yolov8{self.model_size}.pt"
        
        print(f"Loading pre-trained YOLOv8{self.model_size.upper()} model...")
        
        try:
            self.model = YOLO(model_name)
            print(f"✅ Pre-trained model loaded: {model_name}")
            
            # Print model info
            print(f"Model parameters: {sum(p.numel() for p in self.model.model.parameters()):,}")
            print(f"Model size: {self.model_size.upper()}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
            
    def train_model(self, 
                   resume: bool = False,
                   pretrained: bool = True,
                   device: str = 'auto'):
        """Train the model using transfer learning"""
        
        if self.model is None:
            self.load_pretrained_model()
            
        # Setup dataset config
        dataset_config = self.setup_dataset_config()
        
        # Create results directory
        results_dir = self.data_dir / "training_results" / f"yolov8{self.model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🚀 Starting training...")
        print(f"Model: YOLOv8{self.model_size.upper()}")
        print(f"Dataset: {dataset_config}")
        print(f"Results: {results_dir}")
        print(f"Classes: {len(self.class_names)}")
        
        try:
            # Train the model
            results = self.model.train(
                data=dataset_config,
                epochs=self.hyperparams['epochs'],
                batch=self.hyperparams['batch_size'],
                imgsz=self.hyperparams['imgsz'],
                lr0=self.hyperparams['lr0'],
                weight_decay=self.hyperparams['weight_decay'],
                warmup_epochs=self.hyperparams['warmup_epochs'],
                patience=self.hyperparams['patience'],
                save_period=self.hyperparams['save_period'],
                project=str(results_dir.parent),
                name=results_dir.name,
                pretrained=pretrained,
                resume=resume,
                device=device,
                verbose=True,
                plots=True,
                save=True,
                save_txt=True,
                save_conf=True
            )
            
            self.training_results = results
            
            print(f"\n✅ Training completed!")
            print(f"Best model saved to: {results.save_dir}")
            
            # Save training summary
            self._save_training_summary(results_dir)
            
            return results
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            raise
            
    def validate_model(self, model_path: Optional[str] = None):
        """Validate the trained model"""
        
        if model_path:
            self.model = YOLO(model_path)
        elif self.model is None:
            raise ValueError("No model loaded. Please train or load a model first.")
            
        dataset_config = self.data_dir / "dataset.yaml"
        
        print("🔍 Validating model...")
        
        try:
            # Run validation
            results = self.model.val(
                data=str(dataset_config),
                split='val',
                save_json=True,
                save_txt=True,
                plots=True,
                verbose=True
            )
            
            # Print validation metrics
            print(f"\n📊 Validation Results:")
            print(f"mAP50: {results.box.map50:.4f}")
            print(f"mAP50-95: {results.box.map:.4f}")
            print(f"Precision: {results.box.mp:.4f}")
            print(f"Recall: {results.box.mr:.4f}")
            
            return results
            
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            raise
            
    def test_model(self, model_path: Optional[str] = None):
        """Test the model on test set"""
        
        if model_path:
            self.model = YOLO(model_path)
        elif self.model is None:
            raise ValueError("No model loaded. Please train or load a model first.")
            
        dataset_config = self.data_dir / "dataset.yaml"
        
        print("🧪 Testing model...")
        
        try:
            # Run testing
            results = self.model.val(
                data=str(dataset_config),
                split='test',
                save_json=True,
                save_txt=True,
                plots=True,
                verbose=True
            )
            
            # Print test metrics
            print(f"\n📊 Test Results:")
            print(f"mAP50: {results.box.map50:.4f}")
            print(f"mAP50-95: {results.box.map:.4f}")
            print(f"Precision: {results.box.mp:.4f}")
            print(f"Recall: {results.box.mr:.4f}")
            
            return results
            
        except Exception as e:
            print(f"❌ Testing failed: {e}")
            raise
            
    def predict_image(self, image_path: str, conf_threshold: float = 0.5) -> List[Dict]:
        """Predict cards in a single image"""
        
        if self.model is None:
            raise ValueError("No model loaded. Please train or load a model first.")
            
        try:
            # Run prediction
            results = self.model.predict(
                source=image_path,
                conf=conf_threshold,
                save=False,
                verbose=False
            )
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Get class name and parse rank/suit
                        class_name = self.class_names[class_id]
                        rank, suit = class_name.split('_')
                        
                        detections.append({
                            'rank': rank,
                            'suit': suit,
                            'confidence': float(confidence),
                            'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                            'class_name': class_name
                        })
                        
            return detections
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return []
            
    def export_model(self, 
                    model_path: str,
                    export_format: str = 'onnx',
                    optimize: bool = True):
        """Export trained model to different formats"""
        
        if self.model is None:
            self.model = YOLO(model_path)
            
        print(f"📦 Exporting model to {export_format.upper()} format...")
        
        try:
            # Export model
            exported_path = self.model.export(
                format=export_format,
                optimize=optimize,
                verbose=True
            )
            
            print(f"✅ Model exported to: {exported_path}")
            return exported_path
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
            raise
            
    def _save_training_summary(self, results_dir: Path):
        """Save training summary and metrics"""
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'model_size': self.model_size,
            'hyperparameters': self.hyperparams,
            'dataset_info': {
                'num_classes': len(self.class_names),
                'class_names': self.class_names,
                'data_dir': str(self.data_dir)
            },
            'training_results': str(self.training_results) if self.training_results else None
        }
        
        summary_file = results_dir / "training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"Training summary saved: {summary_file}")
        
    def create_training_script(self, output_file: str = "train_card_detector.py"):
        """Create a standalone training script"""
        
        script_content = f'''#!/usr/bin/env python3
"""
Card Detection Training Script
Generated automatically for transfer learning with YOLOv8
"""

from pathlib import Path
import sys

# Add the ml module to path
sys.path.append(str(Path(__file__).parent))

from transfer_learning import CardDetectorYOLO

def main():
    """Main training function"""
    
    print("🃏 Card Detection Training with YOLOv8")
    print("=====================================\n")
    
    # Configuration
    MODEL_SIZE = '{self.model_size}'  # Options: 'n', 's', 'm', 'l', 'x'
    DATA_DIR = "{self.data_dir}"
    
    # Initialize detector
    detector = CardDetectorYOLO(
        model_size=MODEL_SIZE,
        data_dir=DATA_DIR
    )
    
    # Customize hyperparameters if needed
    detector.hyperparams.update({{
        'epochs': 100,
        'batch_size': 16,
        'imgsz': 640,
        'patience': 50
    }})
    
    try:
        # Train the model
        print("Starting training...")
        results = detector.train_model(
            pretrained=True,  # Use transfer learning
            device='auto'     # Auto-detect GPU/CPU
        )
        
        # Validate the model
        print("\nValidating model...")
        val_results = detector.validate_model()
        
        # Test the model
        print("\nTesting model...")
        test_results = detector.test_model()
        
        print("\n✅ Training completed successfully!")
        print(f"Best model saved in: {{results.save_dir}}")
        
        # Export model for deployment
        print("\nExporting model...")
        best_model_path = results.save_dir / "weights" / "best.pt"
        detector.export_model(str(best_model_path), 'onnx')
        
    except Exception as e:
        print(f"❌ Training failed: {{e}}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())
'''
        
        script_path = Path(output_file)
        with open(script_path, 'w') as f:
            f.write(script_content)
            
        print(f"Training script created: {script_path}")
        
def main():
    """Interactive main function"""
    
    print("🃏 Card Detection Transfer Learning with YOLOv8")
    print("================================================\n")
    
    # Get user preferences
    print("Model size options:")
    print("  n = nano (fastest, least accurate)")
    print("  s = small")
    print("  m = medium (recommended)")
    print("  l = large")
    print("  x = extra large (slowest, most accurate)")
    
    model_size = input("\nChoose model size (default: m): ").strip().lower()
    if model_size not in ['n', 's', 'm', 'l', 'x']:
        model_size = 'm'
        
    data_dir = input("Enter data directory (default: ../../data/augmented): ").strip()
    if not data_dir:
        data_dir = "../../data/augmented"
        
    # Initialize detector
    detector = CardDetectorYOLO(model_size=model_size, data_dir=data_dir)
    
    print("\nOptions:")
    print("1. Train new model")
    print("2. Resume training")
    print("3. Validate existing model")
    print("4. Test existing model")
    print("5. Create training script")
    print("6. Export model")
    
    choice = input("\nChoose option (1-6): ").strip()
    
    try:
        if choice == '1':
            print("\n🚀 Starting new training...")
            detector.train_model(pretrained=True)
            
        elif choice == '2':
            print("\n🔄 Resuming training...")
            detector.train_model(resume=True)
            
        elif choice == '3':
            model_path = input("Enter model path: ").strip()
            detector.validate_model(model_path if model_path else None)
            
        elif choice == '4':
            model_path = input("Enter model path: ").strip()
            detector.test_model(model_path if model_path else None)
            
        elif choice == '5':
            script_name = input("Script name (default: train_card_detector.py): ").strip()
            if not script_name:
                script_name = "train_card_detector.py"
            detector.create_training_script(script_name)
            
        elif choice == '6':
            model_path = input("Enter model path to export: ").strip()
            export_format = input("Export format (onnx/torchscript/tflite): ").strip().lower()
            if export_format not in ['onnx', 'torchscript', 'tflite']:
                export_format = 'onnx'
            detector.export_model(model_path, export_format)
            
        else:
            print("Invalid choice")
            
    except Exception as e:
        print(f"❌ Operation failed: {e}")
        
if __name__ == "__main__":
    main()