from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    """Application settings and configuration"""
    
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Playing Card Detector"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "Real-time playing card detection using computer vision"
    
    # CORS Settings
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080"
    ]
    
    # Database Settings
    DATABASE_URL: str = "sqlite:///./card_detector.db"
    DATABASE_ECHO: bool = False  # Set to True for SQL query logging
    
    # ML Model Settings
    MODEL_PATH: str = "./models/card_detector_model.pkl"
    CONFIDENCE_THRESHOLD: float = 0.4  # Lowered for better detection
    NMS_THRESHOLD: float = 0.3  # Lowered for better duplicate removal
    INPUT_SIZE: tuple = (320, 320)  # Reduced for faster processing
    
    # Card Detection Settings
    SUPPORTED_RANKS: List[str] = [
        "A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"
    ]
    SUPPORTED_SUITS: List[str] = ["hearts", "diamonds", "clubs", "spades"]
    
    # Performance Settings - Optimized for faster reaction time
    MAX_FRAME_RATE: int = 60  # Increased for smoother detection
    MAX_CONCURRENT_CONNECTIONS: int = 15  # Increased capacity
    PROCESSING_TIMEOUT: int = 3  # Reduced timeout for faster response
    MAX_DETECTION_TIME: float = 0.1  # Maximum time per detection in seconds
    ENABLE_GPU_ACCELERATION: bool = True  # Enable GPU if available
    PARALLEL_PROCESSING: bool = True  # Enable parallel processing
    
    # File Upload Settings
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_IMAGE_TYPES: List[str] = ["image/jpeg", "image/png", "image/jpg"]
    
    # Logging Settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Security Settings
    SECRET_KEY: str = "your-secret-key-here-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Development Settings
    DEBUG: bool = True
    RELOAD: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()

# Card class mappings for the model
CARD_CLASSES = {
    0: "ace_of_clubs", 1: "ace_of_diamonds", 2: "ace_of_hearts", 3: "ace_of_spades",
    4: "two_of_clubs", 5: "two_of_diamonds", 6: "two_of_hearts", 7: "two_of_spades",
    8: "three_of_clubs", 9: "three_of_diamonds", 10: "three_of_hearts", 11: "three_of_spades",
    12: "four_of_clubs", 13: "four_of_diamonds", 14: "four_of_hearts", 15: "four_of_spades",
    16: "five_of_clubs", 17: "five_of_diamonds", 18: "five_of_hearts", 19: "five_of_spades",
    20: "six_of_clubs", 21: "six_of_diamonds", 22: "six_of_hearts", 23: "six_of_spades",
    24: "seven_of_clubs", 25: "seven_of_diamonds", 26: "seven_of_hearts", 27: "seven_of_spades",
    28: "eight_of_clubs", 29: "eight_of_diamonds", 30: "eight_of_hearts", 31: "eight_of_spades",
    32: "nine_of_clubs", 33: "nine_of_diamonds", 34: "nine_of_hearts", 35: "nine_of_spades",
    36: "ten_of_clubs", 37: "ten_of_diamonds", 38: "ten_of_hearts", 39: "ten_of_spades",
    40: "jack_of_clubs", 41: "jack_of_diamonds", 42: "jack_of_hearts", 43: "jack_of_spades",
    44: "queen_of_clubs", 45: "queen_of_diamonds", 46: "queen_of_hearts", 47: "queen_of_spades",
    48: "king_of_clubs", 49: "king_of_diamonds", 50: "king_of_hearts", 51: "king_of_spades",
    52: "joker"
}

# Reverse mapping for easy lookup
CLASS_TO_ID = {v: k for k, v in CARD_CLASSES.items()}

def parse_card_name(card_name: str) -> tuple:
    """Parse card name into rank and suit"""
    if card_name == "joker":
        return "joker", "joker"
    
    parts = card_name.split("_of_")
    if len(parts) != 2:
        return "unknown", "unknown"
    
    rank = parts[0]
    suit = parts[1]
    
    # Convert rank names
    rank_mapping = {
        "ace": "A",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
        "jack": "J",
        "queen": "Q",
        "king": "K"
    }
    
    return rank_mapping.get(rank, rank), suit