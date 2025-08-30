import cv2
import os
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
import asyncio
from pathlib import Path

class CardDataCollector:
    """Data collection utility for gathering diverse card images"""
    
    def __init__(self, data_dir: str = "../../data"):
        self.data_dir = Path(data_dir)
        self.setup_directories()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.collected_images = []
        
        # Card classes for labeling
        self.card_ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        self.card_suits = ['hearts', 'diamonds', 'clubs', 'spades']
        
    def setup_directories(self):
        """Create necessary directories for data organization"""
        directories = [
            self.data_dir / "raw_images",
            self.data_dir / "processed_images", 
            self.data_dir / "annotations",
            self.data_dir / "train",
            self.data_dir / "val",
            self.data_dir / "test",
            self.data_dir / "augmented"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    def start_camera_collection(self, camera_id: int = 0):
        """Start interactive camera-based data collection"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
            
        print("\n=== Card Data Collection Started ===")
        print("Instructions:")
        print("- Position cards clearly in frame")
        print("- Press SPACE to capture image")
        print("- Press 'q' to quit")
        print("- Try different lighting, angles, backgrounds")
        print("- Collect 50-100 images per card type")
        
        image_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Display frame with overlay
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Images collected: {image_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, "SPACE: Capture | Q: Quit", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Card Data Collection', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space to capture
                self.capture_and_label_image(frame, image_count)
                image_count += 1
                
            elif key == ord('q'):  # Quit
                break
                
        cap.release()
        cv2.destroyAllWindows()
        self.save_collection_metadata()
        print(f"\nCollection completed! {image_count} images saved.")
        
    def capture_and_label_image(self, frame: np.ndarray, image_id: int):
        """Capture image and prompt for labeling"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"card_{self.session_id}_{image_id:04d}_{timestamp}.jpg"
        filepath = self.data_dir / "raw_images" / filename
        
        # Save image
        cv2.imwrite(str(filepath), frame)
        
        # Interactive labeling
        print(f"\nImage {image_id + 1} captured: {filename}")
        
        # Get card information
        cards_in_image = []
        while True:
            print("\nLabel cards in this image (or press Enter to skip):")
            rank = input(f"Card rank ({'/'.join(self.card_ranks)}): ").strip().upper()
            
            if not rank:
                break
                
            if rank not in self.card_ranks:
                print("Invalid rank. Please try again.")
                continue
                
            suit = input(f"Card suit ({'/'.join(self.card_suits)}): ").strip().lower()
            
            if suit not in self.card_suits:
                print("Invalid suit. Please try again.")
                continue
                
            # Simple bounding box (you can enhance this with mouse selection)
            print("Bounding box (x, y, width, height) or press Enter for full image:")
            bbox_input = input("Bbox (x,y,w,h): ").strip()
            
            if bbox_input:
                try:
                    bbox = [int(x) for x in bbox_input.split(',')]
                    if len(bbox) != 4:
                        raise ValueError
                except ValueError:
                    print("Invalid bbox format. Using full image.")
                    bbox = [0, 0, frame.shape[1], frame.shape[0]]
            else:
                bbox = [0, 0, frame.shape[1], frame.shape[0]]
                
            cards_in_image.append({
                'rank': rank,
                'suit': suit,
                'bbox': bbox,
                'confidence': 1.0  # Manual labeling is 100% confident
            })
            
            more_cards = input("More cards in this image? (y/n): ").strip().lower()
            if more_cards != 'y':
                break
                
        # Save annotation
        annotation = {
            'filename': filename,
            'filepath': str(filepath),
            'timestamp': timestamp,
            'image_size': [frame.shape[1], frame.shape[0]],
            'cards': cards_in_image,
            'collection_session': self.session_id
        }
        
        annotation_file = self.data_dir / "annotations" / f"{filename.replace('.jpg', '.json')}"
        with open(annotation_file, 'w') as f:
            json.dump(annotation, f, indent=2)
            
        self.collected_images.append(annotation)
        
    def batch_label_existing_images(self, image_dir: str):
        """Label existing images in a directory"""
        image_dir = Path(image_dir)
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        
        print(f"Found {len(image_files)} images to label")
        
        for i, image_path in enumerate(image_files):
            print(f"\nLabeling image {i+1}/{len(image_files)}: {image_path.name}")
            
            # Load and display image
            image = cv2.imread(str(image_path))
            if image is None:
                continue
                
            cv2.imshow('Image to Label', image)
            cv2.waitKey(1)
            
            # Get labels
            cards_in_image = []
            while True:
                print("\nLabel cards in this image (or press Enter to skip):")
                rank = input(f"Card rank ({'/'.join(self.card_ranks)}): ").strip().upper()
                
                if not rank:
                    break
                    
                if rank not in self.card_ranks:
                    print("Invalid rank. Please try again.")
                    continue
                    
                suit = input(f"Card suit ({'/'.join(self.card_suits)}): ").strip().lower()
                
                if suit not in self.card_suits:
                    print("Invalid suit. Please try again.")
                    continue
                    
                cards_in_image.append({
                    'rank': rank,
                    'suit': suit,
                    'bbox': [0, 0, image.shape[1], image.shape[0]],
                    'confidence': 1.0
                })
                
                more_cards = input("More cards in this image? (y/n): ").strip().lower()
                if more_cards != 'y':
                    break
                    
            # Save annotation
            annotation = {
                'filename': image_path.name,
                'filepath': str(image_path),
                'timestamp': datetime.now().isoformat(),
                'image_size': [image.shape[1], image.shape[0]],
                'cards': cards_in_image,
                'collection_session': 'batch_labeling'
            }
            
            annotation_file = self.data_dir / "annotations" / f"{image_path.stem}.json"
            with open(annotation_file, 'w') as f:
                json.dump(annotation, f, indent=2)
                
        cv2.destroyAllWindows()
        
    def save_collection_metadata(self):
        """Save metadata about the collection session"""
        metadata = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'total_images': len(self.collected_images),
            'images': self.collected_images,
            'card_distribution': self.get_card_distribution()
        }
        
        metadata_file = self.data_dir / f"collection_metadata_{self.session_id}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
    def get_card_distribution(self) -> Dict[str, int]:
        """Get distribution of collected cards"""
        distribution = {}
        
        for image_data in self.collected_images:
            for card in image_data['cards']:
                card_name = f"{card['rank']}_{card['suit']}"
                distribution[card_name] = distribution.get(card_name, 0) + 1
                
        return distribution
        
    def generate_collection_report(self):
        """Generate a report of collected data"""
        total_images = len(list((self.data_dir / "raw_images").glob("*.jpg")))
        total_annotations = len(list((self.data_dir / "annotations").glob("*.json")))
        
        print("\n=== Data Collection Report ===")
        print(f"Total images: {total_images}")
        print(f"Total annotations: {total_annotations}")
        
        # Load all annotations to get card distribution
        all_cards = {}
        for annotation_file in (self.data_dir / "annotations").glob("*.json"):
            with open(annotation_file, 'r') as f:
                data = json.load(f)
                for card in data.get('cards', []):
                    card_name = f"{card['rank']}_{card['suit']}"
                    all_cards[card_name] = all_cards.get(card_name, 0) + 1
                    
        print("\nCard distribution:")
        for card, count in sorted(all_cards.items()):
            print(f"  {card}: {count} images")
            
        # Recommendations
        print("\n=== Recommendations ===")
        if total_images < 500:
            print("- Collect more images (aim for 500-1000 total)")
        if len(all_cards) < 52:
            print("- Missing some card types - try to collect all 52 cards")
        
        min_count = min(all_cards.values()) if all_cards else 0
        if min_count < 10:
            print("- Some cards have very few examples - aim for 10+ per card")
            
        print("- Vary lighting conditions (bright, dim, natural, artificial)")
        print("- Vary backgrounds (table, fabric, different colors)")
        print("- Vary angles (straight, tilted, perspective)")
        print("- Include multiple cards in some images")
        print("- Include partially visible cards")

def main():
    """Main function for interactive data collection"""
    collector = CardDataCollector()
    
    print("Card Data Collection Tool")
    print("1. Start camera collection")
    print("2. Label existing images")
    print("3. Generate collection report")
    
    choice = input("Choose option (1-3): ").strip()
    
    if choice == '1':
        collector.start_camera_collection()
    elif choice == '2':
        image_dir = input("Enter path to image directory: ").strip()
        if os.path.exists(image_dir):
            collector.batch_label_existing_images(image_dir)
        else:
            print("Directory not found")
    elif choice == '3':
        collector.generate_collection_report()
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()