#!/usr/bin/env python3
"""
Complete Training Pipeline for Card Detection
Integrates data collection, augmentation, and transfer learning
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from data_collector import CardDataCollector
from data_augmentation import CardDataAugmentor
from transfer_learning import CardDetectorYOLO

class CardDetectionTrainingPipeline:
    """Complete training pipeline for card detection"""
    
    def __init__(self, base_dir: str = "../../data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Pipeline components
        self.data_collector = CardDataCollector(str(self.base_dir))
        self.data_augmentor = None
        self.model_trainer = None
        
        # Pipeline configuration
        self.config = {
            'data_collection': {
                'min_images_per_card': 10,
                'target_images_per_card': 20,
                'collection_sessions': []
            },
            'data_augmentation': {
                'augmentations_per_image': 5,
                'validation_split': 0.2,
                'test_split': 0.1
            },
            'model_training': {
                'model_size': 'm',
                'epochs': 100,
                'batch_size': 16,
                'patience': 50
            },
            'pipeline_history': []
        }
        
    def run_complete_pipeline(self, 
                            skip_collection: bool = False,
                            skip_augmentation: bool = False,
                            model_size: str = 'm'):
        """Run the complete training pipeline"""
        
        print("🃏 Starting Complete Card Detection Training Pipeline")
        print("====================================================\n")
        
        pipeline_start = datetime.now()
        
        try:
            # Step 1: Data Collection (if not skipped)
            if not skip_collection:
                print("📸 Step 1: Data Collection")
                print("-" * 30)
                self._run_data_collection()
            else:
                print("⏭️  Step 1: Data Collection (SKIPPED)")
                
            # Step 2: Data Analysis
            print("\n📊 Step 2: Data Analysis")
            print("-" * 30)
            self._analyze_collected_data()
            
            # Step 3: Data Augmentation (if not skipped)
            if not skip_augmentation:
                print("\n🔄 Step 3: Data Augmentation")
                print("-" * 30)
                self._run_data_augmentation()
            else:
                print("\n⏭️  Step 3: Data Augmentation (SKIPPED)")
                
            # Step 4: Model Training
            print("\n🚀 Step 4: Model Training")
            print("-" * 30)
            self._run_model_training(model_size)
            
            # Step 5: Model Evaluation
            print("\n📈 Step 5: Model Evaluation")
            print("-" * 30)
            self._run_model_evaluation()
            
            # Step 6: Pipeline Summary
            print("\n📋 Step 6: Pipeline Summary")
            print("-" * 30)
            self._generate_pipeline_summary(pipeline_start)
            
            print("\n✅ Complete pipeline finished successfully!")
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            raise
            
    def _run_data_collection(self):
        """Run data collection step"""
        
        print("Starting interactive data collection...")
        print("Follow the on-screen instructions to collect card images.")
        print("Aim for at least 10-20 images per card type.\n")
        
        # Check existing data
        existing_images = len(list((self.base_dir / "raw_images").glob("*.jpg")))
        print(f"Existing images: {existing_images}")
        
        if existing_images > 0:
            choice = input("Continue with existing data or collect more? (continue/collect): ").strip().lower()
            if choice == 'collect':
                self.data_collector.start_camera_collection()
        else:
            self.data_collector.start_camera_collection()
            
        # Generate collection report
        self.data_collector.generate_collection_report()
        
    def _analyze_collected_data(self):
        """Analyze collected data and provide recommendations"""
        
        # Count images and annotations
        raw_images = list((self.base_dir / "raw_images").glob("*.jpg"))
        annotations = list((self.base_dir / "annotations").glob("*.json"))
        
        print(f"Raw images: {len(raw_images)}")
        print(f"Annotations: {len(annotations)}")
        
        if len(annotations) == 0:
            print("⚠️  No annotations found! Please collect and label data first.")
            return False
            
        # Analyze card distribution
        card_counts = {}
        total_cards = 0
        
        for annotation_file in annotations:
            with open(annotation_file, 'r') as f:
                data = json.load(f)
                
            for card in data.get('cards', []):
                card_name = f"{card['rank']}_{card['suit']}"
                card_counts[card_name] = card_counts.get(card_name, 0) + 1
                total_cards += 1
                
        print(f"\nTotal labeled cards: {total_cards}")
        print(f"Unique card types: {len(card_counts)}")
        
        # Check data quality
        min_count = min(card_counts.values()) if card_counts else 0
        max_count = max(card_counts.values()) if card_counts else 0
        avg_count = total_cards / len(card_counts) if card_counts else 0
        
        print(f"Cards per type - Min: {min_count}, Max: {max_count}, Avg: {avg_count:.1f}")
        
        # Recommendations
        print("\n📋 Data Quality Assessment:")
        
        if total_cards < 200:
            print("⚠️  Low total card count - consider collecting more data")
        else:
            print("✅ Good total card count")
            
        if len(card_counts) < 20:
            print("⚠️  Limited card variety - try to collect more card types")
        else:
            print("✅ Good card variety")
            
        if min_count < 5:
            print("⚠️  Some cards have very few examples - augmentation will help")
        else:
            print("✅ Good minimum examples per card")
            
        return True
        
    def _run_data_augmentation(self):
        """Run data augmentation step"""
        
        augmented_dir = self.base_dir / "augmented"
        
        # Check if augmentation already exists
        if augmented_dir.exists() and len(list(augmented_dir.glob("**/images/*.jpg"))) > 0:
            choice = input("Augmented data exists. Regenerate? (y/n): ").strip().lower()
            if choice != 'y':
                print("Using existing augmented data")
                return
                
        # Initialize augmentor
        self.data_augmentor = CardDataAugmentor(
            str(self.base_dir),
            str(augmented_dir)
        )
        
        # Run augmentation
        print("Generating augmented dataset...")
        self.data_augmentor.augment_dataset(
            num_augmentations_per_image=self.config['data_augmentation']['augmentations_per_image'],
            validation_split=self.config['data_augmentation']['validation_split'],
            test_split=self.config['data_augmentation']['test_split']
        )
        
        # Convert to YOLO format
        print("Converting to YOLO format...")
        self.data_augmentor.create_yolo_format()
        
        print("✅ Data augmentation completed")
        
    def _run_model_training(self, model_size: str = 'm'):
        """Run model training step"""
        
        augmented_dir = self.base_dir / "augmented"
        
        if not augmented_dir.exists():
            print("❌ Augmented data not found. Please run augmentation first.")
            return
            
        # Initialize model trainer
        self.model_trainer = CardDetectorYOLO(
            model_size=model_size,
            data_dir=str(augmented_dir)
        )
        
        # Update hyperparameters from config
        self.model_trainer.hyperparams.update(self.config['model_training'])
        
        print(f"Training YOLOv8{model_size.upper()} model...")
        
        # Train the model
        results = self.model_trainer.train_model(
            pretrained=True,
            device='auto'
        )
        
        print("✅ Model training completed")
        return results
        
    def _run_model_evaluation(self):
        """Run model evaluation step"""
        
        if self.model_trainer is None:
            print("❌ No trained model available for evaluation")
            return
            
        print("Validating model...")
        val_results = self.model_trainer.validate_model()
        
        print("Testing model...")
        test_results = self.model_trainer.test_model()
        
        # Print summary
        print("\n📊 Model Performance Summary:")
        print(f"Validation mAP50: {val_results.box.map50:.4f}")
        print(f"Test mAP50: {test_results.box.map50:.4f}")
        print(f"Validation Precision: {val_results.box.mp:.4f}")
        print(f"Test Precision: {test_results.box.mp:.4f}")
        print(f"Validation Recall: {val_results.box.mr:.4f}")
        print(f"Test Recall: {test_results.box.mr:.4f}")
        
        return val_results, test_results
        
    def _generate_pipeline_summary(self, start_time: datetime):
        """Generate complete pipeline summary"""
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Count final dataset
        augmented_dir = self.base_dir / "augmented"
        
        summary = {
            'pipeline_run': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_minutes': duration.total_seconds() / 60
            },
            'dataset_stats': {},
            'model_performance': {},
            'recommendations': []
        }
        
        # Dataset statistics
        if augmented_dir.exists():
            for split in ['train', 'val', 'test']:
                split_dir = augmented_dir / split / 'images'
                if split_dir.exists():
                    image_count = len(list(split_dir.glob('*.jpg')))
                    summary['dataset_stats'][split] = image_count
                    
        # Save summary
        summary_file = self.base_dir / f"pipeline_summary_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"\n📋 Pipeline Summary:")
        print(f"Duration: {duration.total_seconds()/60:.1f} minutes")
        print(f"Dataset: {summary['dataset_stats']}")
        print(f"Summary saved: {summary_file}")
        
        # Generate recommendations
        self._generate_recommendations(summary)
        
    def _generate_recommendations(self, summary: Dict):
        """Generate recommendations for improving the model"""
        
        print("\n💡 Recommendations for Improvement:")
        
        # Dataset recommendations
        total_train = summary['dataset_stats'].get('train', 0)
        
        if total_train < 1000:
            print("📸 Collect more training data (aim for 1000+ images)")
            
        if total_train < 500:
            print("🔄 Increase augmentation multiplier to generate more training data")
            
        # Model recommendations
        print("🎯 Try different model sizes:")
        print("   - Use 'n' or 's' for faster inference")
        print("   - Use 'l' or 'x' for better accuracy")
        
        print("⚙️  Hyperparameter tuning suggestions:")
        print("   - Increase epochs if loss is still decreasing")
        print("   - Adjust learning rate if training is unstable")
        print("   - Modify batch size based on GPU memory")
        
        print("📊 Data collection tips:")
        print("   - Vary lighting conditions (bright, dim, mixed)")
        print("   - Use different backgrounds and surfaces")
        print("   - Include cards at various angles and distances")
        print("   - Add images with multiple cards")
        print("   - Include partially visible or overlapping cards")
        
def main():
    """Main function with command line interface"""
    
    parser = argparse.ArgumentParser(description='Card Detection Training Pipeline')
    parser.add_argument('--data-dir', default='../../data', help='Base data directory')
    parser.add_argument('--model-size', default='m', choices=['n', 's', 'm', 'l', 'x'], 
                       help='YOLO model size')
    parser.add_argument('--skip-collection', action='store_true', 
                       help='Skip data collection step')
    parser.add_argument('--skip-augmentation', action='store_true', 
                       help='Skip data augmentation step')
    parser.add_argument('--interactive', action='store_true', default=True,
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = CardDetectionTrainingPipeline(args.data_dir)
    
    if args.interactive:
        # Interactive mode
        print("🃏 Card Detection Training Pipeline")
        print("==================================\n")
        
        print("Options:")
        print("1. Run complete pipeline")
        print("2. Data collection only")
        print("3. Data augmentation only")
        print("4. Model training only")
        print("5. Generate data report")
        
        choice = input("\nChoose option (1-5): ").strip()
        
        if choice == '1':
            skip_collection = input("Skip data collection? (y/n): ").strip().lower() == 'y'
            skip_augmentation = input("Skip data augmentation? (y/n): ").strip().lower() == 'y'
            model_size = input("Model size (n/s/m/l/x, default m): ").strip().lower() or 'm'
            
            pipeline.run_complete_pipeline(
                skip_collection=skip_collection,
                skip_augmentation=skip_augmentation,
                model_size=model_size
            )
            
        elif choice == '2':
            pipeline._run_data_collection()
            
        elif choice == '3':
            pipeline._run_data_augmentation()
            
        elif choice == '4':
            model_size = input("Model size (n/s/m/l/x, default m): ").strip().lower() or 'm'
            pipeline._run_model_training(model_size)
            
        elif choice == '5':
            pipeline._analyze_collected_data()
            
        else:
            print("Invalid choice")
            
    else:
        # Command line mode
        pipeline.run_complete_pipeline(
            skip_collection=args.skip_collection,
            skip_augmentation=args.skip_augmentation,
            model_size=args.model_size
        )
        
if __name__ == "__main__":
    main()