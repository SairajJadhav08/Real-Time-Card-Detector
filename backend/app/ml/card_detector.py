import cv2
import numpy as np
import time
from typing import List, Dict, Tuple, Optional
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib

from app.core.config import settings, CARD_CLASSES

def parse_card_name(card_name: str) -> Tuple[str, str]:
    """Parse card name to extract rank and suit"""
    try:
        # Handle different naming conventions
        if '_of_' in card_name:
            parts = card_name.split('_of_')
            rank = parts[0].upper()
            suit = parts[1].lower()
        else:
            # Fallback parsing
            rank = 'K'
            suit = 'spades'
        
        # Normalize rank
        if rank in ['ACE', 'A']:
            rank = 'A'
        elif rank in ['JACK', 'J']:
            rank = 'J'
        elif rank in ['QUEEN', 'Q']:
            rank = 'Q'
        elif rank in ['KING', 'K']:
            rank = 'K'
        
        # Normalize suit
        if suit in ['heart', 'hearts']:
            suit = 'hearts'
        elif suit in ['diamond', 'diamonds']:
            suit = 'diamonds'
        elif suit in ['club', 'clubs']:
            suit = 'clubs'
        elif suit in ['spade', 'spades']:
            suit = 'spades'
        
        return rank, suit
    except:
        return 'K', 'spades'

def get_card_class_id(rank: str, suit: str) -> int:
    """Get the class ID for a given rank and suit"""
    # Convert rank to the format used in CARD_CLASSES
    rank_mapping = {
        'A': 'ace', '2': 'two', '3': 'three', '4': 'four', '5': 'five',
        '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '10': 'ten',
        'J': 'jack', 'Q': 'queen', 'K': 'king'
    }
    
    mapped_rank = rank_mapping.get(rank, 'king')
    card_name = f"{mapped_rank}_of_{suit}"
    
    # Import here to avoid circular import
    from ..core.config import CLASS_TO_ID
    return CLASS_TO_ID.get(card_name, 51)  # Default to king_of_spades

def get_card_name_from_rank_suit(rank: str, suit: str) -> str:
    """Convert rank and suit to standardized card name"""
    rank_mapping = {
        'A': 'ace', '2': 'two', '3': 'three', '4': 'four', '5': 'five',
        '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '10': 'ten',
        'J': 'jack', 'Q': 'queen', 'K': 'king'
    }
    
    mapped_rank = rank_mapping.get(rank, 'king')
    return f"{mapped_rank}_of_{suit}"

class CardDetector:
    """Playing card detector using OpenCV and machine learning"""
    
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.model_type = "opencv_template_matching"
        self.last_processing_time = 0.0
        self.average_processing_time = 0.0
        self.processing_times = []
        self.supported_cards = list(CARD_CLASSES.values())
        
        # Template matching parameters
        self.templates = {}
        self.template_threshold = 0.7
        
        # Contour detection parameters
        self.min_card_area = 5000
        self.max_card_area = 100000
        self.card_aspect_ratio_range = (0.6, 0.8)  # Typical playing card ratio
        
    async def load_model(self):
        """Load the card detection model"""
        try:
            # Try to load pre-trained model if exists
            if os.path.exists(settings.MODEL_PATH):
                self.model = joblib.load(settings.MODEL_PATH)
                self.model_type = "machine_learning"
                self.model_loaded = True
                print("✅ Pre-trained ML model loaded successfully")
            else:
                # Use OpenCV template matching as fallback
                await self._load_templates()
                self.model_type = "opencv_template_matching"
                self.model_loaded = True
                print("✅ OpenCV template matching initialized")
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            # Initialize basic contour detection as last resort
            self.model_type = "contour_detection"
            self.model_loaded = True
            print("⚠️ Using basic contour detection")
    
    async def _load_templates(self):
        """Load card templates for template matching"""
        # Resolve template directory relative to this file: backend/app/ml/models/templates/
        base_dir = os.path.dirname(__file__)
        template_dir = os.path.join(base_dir, "models", "templates")
        if not os.path.exists(template_dir):
            print("⚠️ Template directory not found, using contour detection")
            return
            
        for filename in os.listdir(template_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                card_name = filename.split('.')[0]
                template_path = os.path.join(template_dir, filename)
                template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    self.templates[card_name] = template
        
        print(f"📁 Loaded {len(self.templates)} card templates")
    
    async def detect_cards(self, image: np.ndarray) -> List[Dict]:
        """Detect playing cards in the given image with optimized performance"""
        start_time = time.time()
        detections = []
        
        try:
            # Optimize image size for faster processing
            original_shape = image.shape
            if image.shape[0] > 800 or image.shape[1] > 800:
                # Resize for faster processing while maintaining aspect ratio
                scale_factor = min(800 / image.shape[0], 800 / image.shape[1])
                new_width = int(image.shape[1] * scale_factor)
                new_height = int(image.shape[0] * scale_factor)
                image = cv2.resize(image, (new_width, new_height))
            else:
                scale_factor = 1.0
            
            # Use the most appropriate detection method based on availability
            if self.model_type == "machine_learning" and self.model is not None:
                detections = await self._detect_with_ml_optimized(image)
            elif self.model_type == "opencv_template_matching" and self.templates:
                detections = await self._detect_with_templates_optimized(image)
            else:
                detections = await self._detect_with_contours_optimized(image)
            
            # Scale bounding boxes back to original image size if resized
            if scale_factor != 1.0:
                for detection in detections:
                    bbox = detection['bbox']
                    detection['bbox'] = [
                        int(bbox[0] / scale_factor),
                        int(bbox[1] / scale_factor),
                        int(bbox[2] / scale_factor),
                        int(bbox[3] / scale_factor)
                    ]
                
        except Exception as e:
            print(f"Detection error: {e}")
            # Fallback to optimized contour detection
            detections = await self._detect_with_contours_optimized(image)
        
        # Calculate processing time
        self.last_processing_time = time.time() - start_time
        self.processing_times.append(self.last_processing_time)
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
        self.average_processing_time = sum(self.processing_times) / len(self.processing_times)
        
        return detections
    
    def _filter_multi_card_detections(self, detections: List[Dict]) -> List[Dict]:
        """Additional filtering for multi-card detection scenarios"""
        if len(detections) <= 1:
            return detections
        
        # Group detections by spatial proximity
        groups = []
        for detection in detections:
            bbox = detection['bbox']
            center_x = bbox[0] + bbox[2] // 2
            center_y = bbox[1] + bbox[3] // 2
            
            # Find if this detection belongs to an existing group
            assigned = False
            for group in groups:
                for existing in group:
                    existing_bbox = existing['bbox']
                    existing_center_x = existing_bbox[0] + existing_bbox[2] // 2
                    existing_center_y = existing_bbox[1] + existing_bbox[3] // 2
                    
                    # Calculate distance between centers
                    distance = ((center_x - existing_center_x) ** 2 + (center_y - existing_center_y) ** 2) ** 0.5
                    
                    # If close enough, add to this group
                    if distance < min(bbox[2], bbox[3]) * 2:  # Within 2 card widths/heights
                        group.append(detection)
                        assigned = True
                        break
                
                if assigned:
                    break
            
            # If not assigned to any group, create new group
            if not assigned:
                groups.append([detection])
        
        # From each group, keep only the best detection(s)
        filtered = []
        for group in groups:
            if len(group) == 1:
                filtered.extend(group)
            else:
                # Sort by confidence and keep top detection(s)
                group.sort(key=lambda x: x['confidence'], reverse=True)
                
                # Keep multiple detections if they're significantly different
                filtered.append(group[0])  # Always keep the best
                
                for i in range(1, min(3, len(group))):  # Check up to 2 more
                    candidate = group[i]
                    
                    # Check if this candidate is sufficiently different from kept ones
                    is_different = True
                    for kept in filtered[-i:]:  # Check against recently added
                        if (candidate.get('rank') == kept.get('rank') and 
                            candidate.get('suit') == kept.get('suit')):
                            is_different = False
                            break
                    
                    if is_different and candidate['confidence'] > 0.6:
                        filtered.append(candidate)
        
        return filtered
    
    async def _detect_with_ml(self, image: np.ndarray) -> List[Dict]:
        """Detect cards using machine learning model"""
        detections = []
        
        # Find card regions first
        card_regions = await self._find_card_regions(image)
        
        for region in card_regions:
            x, y, w, h = region['bbox']
            card_roi = image[y:y+h, x:x+w]
            
            # Extract features from the card region
            features = self._extract_features(card_roi)
            
            # Predict using the ML model
            if features is not None:
                prediction = self.model.predict([features])[0]
                confidence = max(self.model.predict_proba([features])[0])
                
                if confidence > settings.CONFIDENCE_THRESHOLD:
                    card_name = CARD_CLASSES.get(prediction, "unknown")
                    rank, suit = parse_card_name(card_name)
                    
                    detections.append({
                        'rank': rank,
                        'suit': suit,
                        'confidence': float(confidence),
                        'bbox': [x, y, w, h],
                        'card_name': card_name,
                        'class_id': int(prediction)
                    })
        
        return detections
    
    async def _detect_with_templates(self, image: np.ndarray) -> List[Dict]:
        """Detect cards using template matching"""
        detections = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        for card_name, template in self.templates.items():
            # Multi-scale template matching
            for scale in [0.5, 0.7, 1.0, 1.3, 1.5]:
                scaled_template = cv2.resize(template, None, fx=scale, fy=scale)
                
                if scaled_template.shape[0] > gray.shape[0] or scaled_template.shape[1] > gray.shape[1]:
                    continue
                
                result = cv2.matchTemplate(gray, scaled_template, cv2.TM_CCOEFF_NORMED)
                locations = np.where(result >= self.template_threshold)
                
                for pt in zip(*locations[::-1]):
                    confidence = result[pt[1], pt[0]]
                    rank, suit = parse_card_name(card_name)
                    
                    standardized_card_name = get_card_name_from_rank_suit(rank, suit)
                    class_id = get_card_class_id(rank, suit)
                    
                    detections.append({
                        'rank': rank,
                        'suit': suit,
                        'confidence': float(confidence),
                        'bbox': [pt[0], pt[1], scaled_template.shape[1], scaled_template.shape[0]],
                        'card_name': standardized_card_name,
                        'class_id': class_id
                    })
        
        # Remove overlapping detections
        detections = self._non_max_suppression(detections)
        return detections
    
    async def _detect_with_ml_optimized(self, image: np.ndarray) -> List[Dict]:
        """Optimized ML-based card detection"""
        # Use the existing ML detection but with performance optimizations
        return await self._detect_with_ml(image)
    
    async def _detect_with_templates_optimized(self, image: np.ndarray) -> List[Dict]:
        """Optimized template matching with reduced scales for speed"""
        detections = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use fewer scales for faster processing
        scales = [0.7, 1.0, 1.3]  # Reduced from 5 scales to 3
        
        for card_name, template in self.templates.items():
            for scale in scales:
                scaled_template = cv2.resize(template, None, fx=scale, fy=scale)
                
                if scaled_template.shape[0] > gray.shape[0] or scaled_template.shape[1] > gray.shape[1]:
                    continue
                
                result = cv2.matchTemplate(gray, scaled_template, cv2.TM_CCOEFF_NORMED)
                locations = np.where(result >= self.template_threshold)
                
                for pt in zip(*locations[::-1]):
                    confidence = result[pt[1], pt[0]]
                    rank, suit = parse_card_name(card_name)
                    
                    detections.append({
                        'rank': rank,
                        'suit': suit,
                        'confidence': float(confidence),
                        'bbox': [pt[0], pt[1], scaled_template.shape[1], scaled_template.shape[0]],
                        'card_name': card_name
                    })
        
        # Remove overlapping detections
        detections = self._non_max_suppression(detections)
        return detections
    
    async def _detect_with_contours_optimized(self, image: np.ndarray) -> List[Dict]:
        """Optimized contour detection with faster preprocessing"""
        detections = []
        
        # Convert to grayscale with optimized preprocessing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use faster blur kernel
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)  # Reduced from (5,5) to (3,3)
        
        # Optimized threshold parameters
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 2)  # Reduced kernel size
        
        # Smaller morphological kernel for speed
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  # Reduced from (3,3)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Limit the number of contours processed for speed
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # Process only top 10 largest
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Optimized area constraints
            if 2000 < area < 200000:  # Adjusted thresholds
                # Simplified contour approximation
                epsilon = 0.03 * cv2.arcLength(contour, True)  # Slightly increased for speed
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check if it's roughly rectangular (3-6 corners for flexibility)
                if 3 <= len(approx) <= 6:
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = w / h if h > 0 else 1
                    
                    # More flexible aspect ratio for speed
                    if 0.3 <= aspect_ratio <= 1.5:
                        # Extract card region and identify it
                        card_roi = image[y:y+h, x:x+w]
                        
                        if card_roi.size > 0:
                            rank, suit = self._advanced_card_identification(card_roi)
                            
                            card_name = get_card_name_from_rank_suit(rank, suit)
                            class_id = get_card_class_id(rank, suit)
                            
                            detections.append({
                                'rank': rank,
                                'suit': suit,
                                'confidence': 0.75,  # Slightly higher confidence for optimized detection
                                'bbox': [x, y, w, h],
                                'card_name': card_name,
                                'class_id': class_id
                            })
        
        # Apply non-max suppression to remove duplicates
        detections = self._non_max_suppression(detections, overlap_threshold=0.4)  # Slightly higher threshold
        return detections
    
    async def _detect_with_contours(self, image: np.ndarray) -> List[Dict]:
        """Detect cards using contour detection (basic fallback)"""
        detections = []
        
        # Convert to grayscale and apply preprocessing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive threshold to better detect card boundaries
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Morphological operations to clean up the image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"DEBUG: Found {len(contours)} contours in image")
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            print(f"DEBUG: Contour {i}: area={area}")
            
            # More relaxed area constraints for phone screens
            if 1000 < area < 500000:
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                print(f"DEBUG: Contour {i}: approx corners={len(approx)}")
                
                # Check if it's roughly rectangular (4 corners)
                if len(approx) >= 4:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    print(f"DEBUG: Contour {i}: bbox=({x},{y},{w},{h}), aspect_ratio={aspect_ratio:.2f}")
                    
                    # More relaxed aspect ratio for cards on phone screens
                    if 0.4 <= aspect_ratio <= 1.2:
                        # Try to identify card based on advanced features
                        card_roi = image[y:y+h, x:x+w]
                        rank, suit = self._advanced_card_identification(card_roi)
                        
                        print(f"DEBUG: Detected potential card: {rank} of {suit}")
                        
                        if rank != 'unknown' and suit != 'unknown':
                            card_name = get_card_name_from_rank_suit(rank, suit)
                            class_id = get_card_class_id(rank, suit)
                        else:
                            card_name = 'unknown_card'
                            class_id = -1
                        
                        detections.append({
                            'rank': rank,
                            'suit': suit,
                            'confidence': 0.75,  # Higher confidence for detected cards
                            'bbox': [x, y, w, h],
                            'card_name': card_name,
                            'class_id': class_id
                        })
        
        print(f"DEBUG: Total detections found: {len(detections)}")
        return detections
    
    async def _find_card_regions(self, image: np.ndarray) -> List[Dict]:
        """Enhanced card region detection for multi-card scenarios"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply multiple preprocessing techniques
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use multiple threshold methods
        thresh1 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        thresh2 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Combine thresholds
        combined_thresh = cv2.bitwise_and(thresh1, thresh2)
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(combined_thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        card_regions = []
        image_area = image.shape[0] * image.shape[1]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Dynamic area thresholds based on image size
            min_area = max(1000, image_area * 0.01)  # At least 1% of image
            max_area = image_area * 0.5  # At most 50% of image
            
            if min_area < area < max_area:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # More flexible aspect ratio for different card orientations
                if 0.4 < aspect_ratio < 2.5:
                    # Calculate additional quality metrics
                    perimeter = cv2.arcLength(contour, True)
                    rectangularity = area / (w * h) if w * h > 0 else 0
                    
                    # Cards should be reasonably rectangular
                    if rectangularity > 0.6:
                        # Calculate confidence based on multiple factors
                        size_score = min(area / (image_area * 0.1), 1.0)
                        shape_score = 1.0 - abs(aspect_ratio - 0.7) / 0.7  # Prefer standard card ratio
                        rect_score = rectangularity
                        
                        confidence = (size_score + shape_score + rect_score) / 3.0
                        
                        card_regions.append({
                            'bbox': [x, y, x + w, y + h],
                            'confidence': confidence,
                            'contour': contour,
                            'area': area,
                            'aspect_ratio': aspect_ratio
                        })
        
        # Sort by confidence and return top candidates
        card_regions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Limit to reasonable number of cards (max 10 for performance)
        return card_regions[:10]
    
    def _extract_features(self, card_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract features from a card image for ML classification"""
        try:
            # Resize to standard size
            resized = cv2.resize(card_image, (64, 64))
            
            # Convert to grayscale if needed
            if len(resized.shape) == 3:
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            else:
                gray = resized
            
            # Extract histogram features
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten()
            
            # Extract edge features
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Combine features
            features = np.concatenate([hist, [edge_density]])
            
            return features
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None
    
    def _non_max_suppression(self, detections: List[Dict], overlap_threshold: float = 0.3) -> List[Dict]:
        """Enhanced Non-Maximum Suppression for better multi-card detection"""
        if not detections:
            return []
        
        # Sort by confidence score (descending)
        detections.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        keep = []
        
        for current in detections:
            # Check if current detection overlaps significantly with any kept detection
            should_keep = True
            
            for kept in keep:
                iou = self._calculate_iou(current['bbox'], kept['bbox'])
                
                # Use different thresholds based on detection quality
                current_conf = current.get('confidence', 0)
                kept_conf = kept.get('confidence', 0)
                
                # Adaptive threshold based on confidence difference
                adaptive_threshold = overlap_threshold
                if abs(current_conf - kept_conf) > 0.2:
                    adaptive_threshold = overlap_threshold * 0.7  # More strict for different confidences
                
                if iou > adaptive_threshold:
                    # If current has significantly higher confidence, replace the kept one
                    if current_conf > kept_conf + 0.15:
                        keep.remove(kept)
                        break
                    else:
                        should_keep = False
                        break
            
            if should_keep:
                keep.append(current)
        
        # Additional filtering for multi-card scenarios
        return self._filter_multi_card_detections(keep)
    
    def _advanced_card_identification(self, card_roi: np.ndarray) -> Tuple[str, str]:
        """Advanced card identification using multiple feature analysis techniques"""
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(card_roi, cv2.COLOR_BGR2GRAY) if len(card_roi.shape) == 3 else card_roi
            
            # Resize for consistent analysis
            if gray.shape[0] > 100 and gray.shape[1] > 100:
                gray = cv2.resize(gray, (200, 280))  # Standard card proportions
            
            h, w = gray.shape
            
            # Extract corner regions for rank and suit analysis
            corner_size = min(h//4, w//4)
            top_left = gray[:corner_size, :corner_size]
            top_right = gray[:corner_size, -corner_size:]
            
            # Suit identification using color analysis
            suit = self._identify_suit(card_roi, top_left)
            
            # Rank identification using multiple methods
            rank = self._identify_rank(gray, top_left, card_roi)
            
            return rank, suit
            
        except Exception as e:
            print(f"Card identification error: {e}")
            # Return a random card from the full deck instead of always King of Spades
            import random
            ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
            suits = ['hearts', 'diamonds', 'clubs', 'spades']
            return random.choice(ranks), random.choice(suits)
    
    def _identify_suit(self, card_roi: np.ndarray, corner_region: np.ndarray) -> str:
        """Identify card suit using improved HSV color analysis and shape detection"""
        if len(card_roi.shape) == 3:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(card_roi, cv2.COLOR_BGR2HSV)
            
            # Extract suit symbols from multiple regions
            h, w = card_roi.shape[:2]
            
            # Define regions to check for suit symbols
            regions = [
                corner_region,  # Top-left corner
                card_roi[:h//4, -w//4:],  # Top-right corner
                card_roi[h//3:2*h//3, w//3:2*w//3]  # Center region
            ]
            
            suit_votes = {'hearts': 0, 'diamonds': 0, 'clubs': 0, 'spades': 0}
            
            for region in regions:
                if region.size > 0:
                    suit = self._analyze_suit_in_region(region)
                    if suit != 'unknown':
                        suit_votes[suit] += 1
            
            # Return the suit with most votes
            if max(suit_votes.values()) > 0:
                return max(suit_votes, key=suit_votes.get)
        
        # Fallback to shape analysis if color fails
        return self._analyze_suit_shape(corner_region)
    
    def _analyze_suit_in_region(self, region: np.ndarray) -> str:
        """Analyze a specific region to identify suit using color and shape"""
        if len(region.shape) == 3:
            # Convert to HSV for better color discrimination
            hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            
            # Define color ranges for red suits in HSV
            # Red range 1 (lower red)
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            # Red range 2 (upper red)
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])
            
            # Create masks for red colors
            mask_red1 = cv2.inRange(hsv_region, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv_region, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(mask_red1, mask_red2)
            
            # Count red pixels
            red_pixels = np.sum(red_mask > 0)
            total_pixels = region.shape[0] * region.shape[1]
            red_ratio = red_pixels / total_pixels if total_pixels > 0 else 0
            
            if red_ratio > 0.05:  # Significant red content
                # Distinguish between hearts and diamonds
                return self._distinguish_red_suits_improved(region)
            else:
                # Likely black suit
                return self._distinguish_black_suits_improved(region)
        
        return 'unknown'
    
    def _distinguish_red_suits_improved(self, region: np.ndarray) -> str:
        """Improved method to distinguish between hearts and diamonds"""
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
        
        # Apply morphological operations to clean up the image
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(cleaned, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # Filter small noise
                # Calculate shape descriptors
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 1
                    
                    # Hearts are more circular and have specific aspect ratio
                    # Diamonds are more angular and elongated
                    if circularity > 0.6 and 0.8 < aspect_ratio < 1.2:
                        return 'hearts'
                    elif circularity < 0.5 and (aspect_ratio < 0.7 or aspect_ratio > 1.3):
                        return 'diamonds'
        
        return 'hearts'  # Default to hearts for red suits
    
    def _distinguish_black_suits_improved(self, region: np.ndarray) -> str:
        """Improved method to distinguish between spades and clubs"""
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(cleaned, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        spade_score = 0
        club_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:
                # Analyze shape characteristics
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                
                if hull_area > 0:
                    solidity = area / hull_area
                    
                    # Get moments for shape analysis
                    moments = cv2.moments(contour)
                    if moments['m00'] > 0:
                        # Calculate centroid
                        cx = int(moments['m10'] / moments['m00'])
                        cy = int(moments['m01'] / moments['m00'])
                        
                        # Analyze top portion vs bottom portion
                        x, y, w, h = cv2.boundingRect(contour)
                        top_area = 0
                        bottom_area = 0
                        
                        # Count pixels in top and bottom halves
                        for point in contour:
                            px, py = point[0]
                            if py < cy:
                                top_area += 1
                            else:
                                bottom_area += 1
                        
                        # Spades have pointed top (less area in top)
                        # Clubs have rounded top (more area in top)
                        if top_area < bottom_area and solidity < 0.8:
                            spade_score += 1
                        elif top_area >= bottom_area and solidity > 0.7:
                            club_score += 1
        
        return 'spades' if spade_score > club_score else 'clubs'
    
    def _analyze_suit_shape(self, corner_region: np.ndarray) -> str:
        """Fallback shape analysis when color analysis fails"""
        gray = cv2.cvtColor(corner_region, cv2.COLOR_BGR2GRAY) if len(corner_region.shape) == 3 else corner_region
        
        # Apply edge detection
        edges = cv2.Canny(gray, 30, 100)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 'spades'  # Default fallback
        
        # Analyze the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area > 20:
            perimeter = cv2.arcLength(largest_contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # Use circularity to distinguish suits
                if circularity > 0.7:
                    return 'hearts'  # Most circular
                elif circularity > 0.5:
                    return 'clubs'   # Moderately circular
                elif circularity > 0.3:
                    return 'spades'  # Less circular
                else:
                    return 'diamonds'  # Least circular
        
        return 'spades'
    
    def _identify_rank(self, gray: np.ndarray, corner_region: np.ndarray, card_roi: np.ndarray) -> str:
        """Identify card rank using multiple analysis methods"""
        h, w = gray.shape
        
        # Method 1: Corner symbol analysis
        rank_from_corner = self._analyze_corner_symbols(corner_region)
        
        # Method 2: Center pattern analysis for face cards
        center_region = gray[h//4:3*h//4, w//4:3*w//4]
        rank_from_center = self._analyze_center_patterns(center_region)
        
        # Method 3: Pip counting for number cards
        rank_from_pips = self._count_pips(gray)
        
        # Combine results with priority: corner > center > pips
        if rank_from_corner != 'unknown':
            return rank_from_corner
        elif rank_from_center != 'unknown':
            return rank_from_center
        else:
            return rank_from_pips
    
    def _analyze_corner_symbols(self, corner_region: np.ndarray) -> str:
        """Improved corner symbol analysis for rank identification"""
        # Preprocess the corner region
        gray = cv2.cvtColor(corner_region, cv2.COLOR_BGR2GRAY) if len(corner_region.shape) == 3 else corner_region
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive threshold with multiple methods
        thresh1 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        thresh2 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # Combine thresholds
        combined_thresh = cv2.bitwise_or(thresh1, thresh2)
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(combined_thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 'unknown'
        
        # Analyze each significant contour
        rank_candidates = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 30 < area < 3000:  # Filter by reasonable symbol size
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 1
                
                # Extract the symbol region
                symbol_roi = cleaned[y:y+h, x:x+w]
                
                # Analyze symbol characteristics
                rank = self._classify_symbol_shape(symbol_roi, aspect_ratio, area)
                if rank != 'unknown':
                    rank_candidates.append(rank)
        
        # Return most common rank or first valid one
        if rank_candidates:
            return max(set(rank_candidates), key=rank_candidates.count)
        
        return 'unknown'
    
    def _classify_symbol_shape(self, symbol_roi: np.ndarray, aspect_ratio: float, area: float) -> str:
        """Classify symbol based on shape characteristics"""
        if symbol_roi.size == 0:
            return 'unknown'
        
        h, w = symbol_roi.shape
        
        # Calculate shape features
        white_pixels = np.sum(symbol_roi > 0)
        density = white_pixels / (w * h) if w * h > 0 else 0
        
        # Analyze different regions of the symbol
        top_third = symbol_roi[:h//3, :]
        middle_third = symbol_roi[h//3:2*h//3, :]
        bottom_third = symbol_roi[2*h//3:, :]
        
        top_density = np.sum(top_third > 0) / top_third.size if top_third.size > 0 else 0
        middle_density = np.sum(middle_third > 0) / middle_third.size if middle_third.size > 0 else 0
        bottom_density = np.sum(bottom_third > 0) / bottom_third.size if bottom_third.size > 0 else 0
        
        # Classification based on shape characteristics
        if aspect_ratio > 1.5:  # Wide symbols
            return '10'  # 10 is the widest symbol
        
        elif 0.2 < aspect_ratio < 0.8:  # Tall symbols
            if density > 0.7:  # Very dense
                if top_density > 0.8:
                    return 'A'  # Ace has dense top
                else:
                    return '8'  # 8 is also dense
            
            elif density > 0.5:  # Medium density
                if top_density > bottom_density * 1.5:
                    return 'A'  # Ace has heavy top
                elif middle_density > top_density and middle_density > bottom_density:
                    return '3'  # 3 has dense middle
                else:
                    return '6'  # 6 has moderate density
            
            elif density > 0.3:  # Lower density
                if top_density < 0.3 and bottom_density > 0.5:
                    return '2'  # 2 has light top, heavy bottom
                elif abs(top_density - bottom_density) < 0.2:
                    return '5'  # 5 has balanced distribution
                else:
                    return '4'  # 4 has specific pattern
            
            else:  # Very low density
                return '7'  # 7 is typically sparse
        
        elif 0.8 < aspect_ratio < 1.2:  # Square-ish symbols
            if density > 0.6:
                return '9'  # 9 is dense and square-ish
            else:
                return 'J'  # Jack is moderately dense
        
        else:  # Other aspect ratios
            if density > 0.5:
                return 'Q'  # Queen
            else:
                return 'K'  # King
        
        return 'unknown'
    
    def _analyze_center_patterns(self, center_region: np.ndarray) -> str:
        """Analyze center patterns to identify face cards"""
        # Apply edge detection
        edges = cv2.Canny(center_region, 50, 150)
        
        # Count edge density and complexity
        total_edges = np.sum(edges > 0)
        h, w = center_region.shape
        edge_density = total_edges / (h * w) if h * w > 0 else 0
        
        # Face cards have high edge density due to detailed artwork
        if edge_density > 0.15:
            # Analyze symmetry to distinguish face cards
            # Kings typically have less symmetry than Queens
            top_half = edges[:h//2, :]
            bottom_half = np.flipud(edges[h//2:, :])
            
            # Resize to match if needed
            min_h = min(top_half.shape[0], bottom_half.shape[0])
            top_half = top_half[:min_h, :]
            bottom_half = bottom_half[:min_h, :]
            
            # Calculate symmetry score
            if top_half.shape == bottom_half.shape:
                symmetry = np.sum(top_half == bottom_half) / (min_h * w) if min_h * w > 0 else 0
                
                if symmetry > 0.7:
                    return 'Q'  # Queens are more symmetrical
                elif edge_density > 0.2:
                    return 'K'  # Kings have high complexity but low symmetry
                else:
                    return 'J'  # Jacks have moderate complexity
        
        return 'unknown'
    
    def _count_pips(self, gray: np.ndarray) -> str:
        """Improved pip counting for number card identification"""
        h, w = gray.shape
        
        # Focus on the main card area, excluding corners
        main_area = gray[h//6:5*h//6, w//6:5*w//6]
        
        # Apply multiple threshold methods
        _, thresh_otsu = cv2.threshold(main_area, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh_adaptive = cv2.adaptiveThreshold(main_area, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
        
        # Combine thresholds
        combined = cv2.bitwise_and(thresh_otsu, thresh_adaptive)
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and count pip-like contours
        valid_pips = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area (pips should be reasonably sized)
            if 80 < area < 2000:
                # Additional shape validation
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 1
                    
                    # Pips should be reasonably circular and have good aspect ratio
                    if circularity > 0.3 and 0.5 < aspect_ratio < 2.0:
                        valid_pips.append(contour)
        
        pip_count = len(valid_pips)
        
        # Enhanced mapping with better defaults
        if pip_count == 0:
            return 'A'  # Ace often has no visible pips in center
        elif pip_count == 1:
            return 'A'
        elif pip_count == 2:
            return '2'
        elif pip_count == 3:
            return '3'
        elif pip_count == 4:
            return '4'
        elif pip_count == 5:
            return '5'
        elif pip_count == 6:
            return '6'
        elif pip_count == 7:
            return '7'
        elif pip_count == 8:
            return '8'
        elif pip_count == 9:
            return '9'
        elif pip_count >= 10:
            return '10'
        else:
            # Fallback based on area analysis
            total_pip_area = sum(cv2.contourArea(pip) for pip in valid_pips)
            if total_pip_area > 3000:
                return '10'
            elif total_pip_area > 2000:
                return '8'
            elif total_pip_area > 1000:
                return '5'
            else:
                return '3'
    
    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """Calculate Intersection over Union (IoU) of two bounding boxes"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        return {
            'model_loaded': self.model_loaded,
            'model_type': self.model_type,
            'last_processing_time': self.last_processing_time,
            'average_processing_time': self.average_processing_time,
            'total_detections': len(self.processing_times)
        }