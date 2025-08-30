import cv2
import numpy as np
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random
from datetime import datetime
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CardDataAugmentor:
    """Data augmentation for card detection training"""
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create augmentation pipeline
        self.augmentation_pipeline = A.Compose([
            # Geometric transformations
            A.Rotate(limit=30, p=0.7),
            A.HorizontalFlip(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=15,
                p=0.7
            ),
            A.Perspective(scale=(0.05, 0.1), p=0.3),
            
            # Color and lighting augmentations
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.8
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.6
            ),
            A.CLAHE(clip_limit=2.0, p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            
            # Noise and blur
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.MotionBlur(blur_limit=7, p=0.2),
            
            # Weather and environmental effects
            A.RandomShadow(p=0.2),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.1),
            
            # Quality degradation
            A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
            A.Downscale(scale_min=0.7, scale_max=0.9, p=0.2),
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))
        
        # Lighter augmentation for validation data
        self.light_augmentation = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.5
            ),
            A.Rotate(limit=10, p=0.3),
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels'],
            min_visibility=0.5
        ))
        
    def augment_dataset(self, 
                       num_augmentations_per_image: int = 5,
                       validation_split: float = 0.2,
                       test_split: float = 0.1):
        """Augment entire dataset and split into train/val/test"""
        
        print("Starting dataset augmentation...")
        
        # Get all annotation files
        annotation_files = list(self.input_dir.glob("annotations/*.json"))
        
        if not annotation_files:
            print("No annotation files found!")
            return
            
        print(f"Found {len(annotation_files)} annotated images")
        
        # Shuffle and split data
        random.shuffle(annotation_files)
        
        n_total = len(annotation_files)
        n_test = int(n_total * test_split)
        n_val = int(n_total * validation_split)
        n_train = n_total - n_test - n_val
        
        train_files = annotation_files[:n_train]
        val_files = annotation_files[n_train:n_train + n_val]
        test_files = annotation_files[n_train + n_val:]
        
        print(f"Split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
        
        # Create output directories
        for split in ['train', 'val', 'test']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'annotations').mkdir(parents=True, exist_ok=True)
            
        # Process each split
        self._process_split(train_files, 'train', num_augmentations_per_image)
        self._process_split(val_files, 'val', 2)  # Fewer augmentations for validation
        self._process_split(test_files, 'test', 0)  # No augmentation for test
        
        # Generate dataset statistics
        self._generate_dataset_stats()
        
        print("Dataset augmentation completed!")
        
    def _process_split(self, annotation_files: List[Path], split: str, num_augmentations: int):
        """Process a data split with augmentations"""
        
        print(f"\nProcessing {split} split...")
        
        for i, annotation_file in enumerate(annotation_files):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(annotation_files)}")
                
            # Load annotation
            with open(annotation_file, 'r') as f:
                annotation = json.load(f)
                
            # Load image
            image_path = self.input_dir / "raw_images" / annotation['filename']
            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}")
                continue
                
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Warning: Could not load image: {image_path}")
                continue
                
            # Convert BGR to RGB for albumentations
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Prepare bounding boxes and labels
            bboxes = []
            class_labels = []
            
            for card in annotation['cards']:
                bbox = card['bbox']  # [x, y, w, h]
                # Convert to [x_min, y_min, x_max, y_max] for albumentations
                x_min, y_min, w, h = bbox
                x_max = x_min + w
                y_max = y_min + h
                
                bboxes.append([x_min, y_min, x_max, y_max])
                class_labels.append(f"{card['rank']}_{card['suit']}")
                
            # Save original image (for test split or as base)
            original_filename = f"{annotation_file.stem}_original.jpg"
            original_path = self.output_dir / split / 'images' / original_filename
            cv2.imwrite(str(original_path), image)
            
            # Save original annotation
            self._save_annotation(
                annotation, original_filename, 
                self.output_dir / split / 'annotations' / f"{annotation_file.stem}_original.json"
            )
            
            # Apply augmentations
            for aug_idx in range(num_augmentations):
                try:
                    # Choose augmentation pipeline
                    if split == 'val':
                        pipeline = self.light_augmentation
                    else:
                        pipeline = self.augmentation_pipeline
                        
                    # Apply augmentation
                    augmented = pipeline(
                        image=image_rgb,
                        bboxes=bboxes,
                        class_labels=class_labels
                    )
                    
                    # Convert back to BGR for saving
                    aug_image = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
                    aug_bboxes = augmented['bboxes']
                    aug_labels = augmented['class_labels']
                    
                    # Save augmented image
                    aug_filename = f"{annotation_file.stem}_aug_{aug_idx:03d}.jpg"
                    aug_path = self.output_dir / split / 'images' / aug_filename
                    cv2.imwrite(str(aug_path), aug_image)
                    
                    # Create augmented annotation
                    aug_annotation = annotation.copy()
                    aug_annotation['filename'] = aug_filename
                    aug_annotation['augmented'] = True
                    aug_annotation['augmentation_id'] = aug_idx
                    
                    # Update card bounding boxes
                    aug_cards = []
                    for bbox, label in zip(aug_bboxes, aug_labels):
                        rank, suit = label.split('_')
                        x_min, y_min, x_max, y_max = bbox
                        w = x_max - x_min
                        h = y_max - y_min
                        
                        aug_cards.append({
                            'rank': rank,
                            'suit': suit,
                            'bbox': [int(x_min), int(y_min), int(w), int(h)],
                            'confidence': 1.0
                        })
                        
                    aug_annotation['cards'] = aug_cards
                    
                    # Save augmented annotation
                    aug_annotation_path = self.output_dir / split / 'annotations' / f"{annotation_file.stem}_aug_{aug_idx:03d}.json"
                    self._save_annotation(aug_annotation, aug_filename, aug_annotation_path)
                    
                except Exception as e:
                    print(f"Warning: Augmentation failed for {annotation_file.name}, aug {aug_idx}: {e}")
                    continue
                    
    def _save_annotation(self, annotation: Dict, filename: str, output_path: Path):
        """Save annotation file"""
        annotation_copy = annotation.copy()
        annotation_copy['filename'] = filename
        annotation_copy['augmentation_timestamp'] = datetime.now().isoformat()
        
        with open(output_path, 'w') as f:
            json.dump(annotation_copy, f, indent=2)
            
    def _generate_dataset_stats(self):
        """Generate statistics about the augmented dataset"""
        stats = {
            'generation_timestamp': datetime.now().isoformat(),
            'splits': {}
        }
        
        for split in ['train', 'val', 'test']:
            split_dir = self.output_dir / split
            
            # Count images and annotations
            images = list((split_dir / 'images').glob('*.jpg'))
            annotations = list((split_dir / 'annotations').glob('*.json'))
            
            # Count cards by type
            card_counts = {}
            total_cards = 0
            
            for annotation_file in annotations:
                with open(annotation_file, 'r') as f:
                    data = json.load(f)
                    
                for card in data.get('cards', []):
                    card_name = f"{card['rank']}_{card['suit']}"
                    card_counts[card_name] = card_counts.get(card_name, 0) + 1
                    total_cards += 1
                    
            stats['splits'][split] = {
                'num_images': len(images),
                'num_annotations': len(annotations),
                'total_cards': total_cards,
                'unique_card_types': len(card_counts),
                'card_distribution': card_counts
            }
            
        # Save stats
        stats_file = self.output_dir / 'dataset_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
            
        # Print summary
        print("\n=== Dataset Statistics ===")
        for split, split_stats in stats['splits'].items():
            print(f"{split.upper()}:")
            print(f"  Images: {split_stats['num_images']}")
            print(f"  Total cards: {split_stats['total_cards']}")
            print(f"  Unique card types: {split_stats['unique_card_types']}")
            
    def create_yolo_format(self, class_names_file: str = "classes.txt"):
        """Convert annotations to YOLO format"""
        
        print("Converting to YOLO format...")
        
        # Get all unique class names
        all_classes = set()
        
        for split in ['train', 'val', 'test']:
            annotation_dir = self.output_dir / split / 'annotations'
            for annotation_file in annotation_dir.glob('*.json'):
                with open(annotation_file, 'r') as f:
                    data = json.load(f)
                    
                for card in data.get('cards', []):
                    all_classes.add(f"{card['rank']}_{card['suit']}")
                    
        # Create class mapping
        class_list = sorted(list(all_classes))
        class_to_id = {cls: idx for idx, cls in enumerate(class_list)}
        
        # Save class names file
        with open(self.output_dir / class_names_file, 'w') as f:
            for cls in class_list:
                f.write(f"{cls}\n")
                
        # Convert annotations for each split
        for split in ['train', 'val', 'test']:
            yolo_dir = self.output_dir / split / 'labels'
            yolo_dir.mkdir(exist_ok=True)
            
            annotation_dir = self.output_dir / split / 'annotations'
            
            for annotation_file in annotation_dir.glob('*.json'):
                with open(annotation_file, 'r') as f:
                    data = json.load(f)
                    
                # Get image dimensions
                img_w, img_h = data['image_size']
                
                # Convert to YOLO format
                yolo_lines = []
                
                for card in data.get('cards', []):
                    class_name = f"{card['rank']}_{card['suit']}"
                    class_id = class_to_id[class_name]
                    
                    # Convert bbox to YOLO format (normalized center coordinates)
                    x, y, w, h = card['bbox']
                    
                    # Convert to center coordinates and normalize
                    center_x = (x + w / 2) / img_w
                    center_y = (y + h / 2) / img_h
                    norm_w = w / img_w
                    norm_h = h / img_h
                    
                    yolo_lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}")
                    
                # Save YOLO annotation
                yolo_file = yolo_dir / f"{annotation_file.stem}.txt"
                with open(yolo_file, 'w') as f:
                    f.write('\n'.join(yolo_lines))
                    
        print(f"YOLO format conversion completed. {len(class_list)} classes.")
        
def main():
    """Main function for data augmentation"""
    
    print("Card Data Augmentation Tool")
    
    input_dir = input("Enter input data directory (default: ../../data): ").strip()
    if not input_dir:
        input_dir = "../../data"
        
    output_dir = input("Enter output directory (default: ../../data/augmented): ").strip()
    if not output_dir:
        output_dir = "../../data/augmented"
        
    num_aug = input("Number of augmentations per training image (default: 5): ").strip()
    try:
        num_aug = int(num_aug) if num_aug else 5
    except ValueError:
        num_aug = 5
        
    augmentor = CardDataAugmentor(input_dir, output_dir)
    
    print("\nOptions:")
    print("1. Augment dataset")
    print("2. Convert to YOLO format")
    print("3. Both")
    
    choice = input("Choose option (1-3): ").strip()
    
    if choice in ['1', '3']:
        augmentor.augment_dataset(num_augmentations_per_image=num_aug)
        
    if choice in ['2', '3']:
        augmentor.create_yolo_format()
        
    print("\nAugmentation completed!")

if __name__ == "__main__":
    main()