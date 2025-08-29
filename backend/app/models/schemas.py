from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class DetectionRequest(BaseModel):
    """Request model for card detection"""
    image_data: str = Field(..., description="Base64 encoded image data")
    session_id: Optional[str] = Field(None, description="Session identifier")
    
class CardDetection(BaseModel):
    """Individual card detection result"""
    rank: str = Field(..., description="Card rank (A, 2-10, J, Q, K)")
    suit: str = Field(..., description="Card suit (hearts, diamonds, clubs, spades)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    bbox: List[int] = Field(..., description="Bounding box coordinates [x, y, width, height]")
    card_name: Optional[str] = Field(None, description="Full card name")

class DetectionResult(BaseModel):
    """Response model for card detection"""
    success: bool = Field(..., description="Whether detection was successful")
    detections: List[CardDetection] = Field(default=[], description="List of detected cards")
    timestamp: str = Field(..., description="Detection timestamp")
    processing_time: float = Field(..., description="Processing time in seconds")
    error_message: Optional[str] = Field(None, description="Error message if detection failed")

class DetectionHistory(BaseModel):
    """Model for detection history records"""
    id: int = Field(..., description="Detection record ID")
    card_rank: str = Field(..., description="Card rank")
    card_suit: str = Field(..., description="Card suit")
    confidence: float = Field(..., description="Detection confidence")
    bbox: List[int] = Field(..., description="Bounding box coordinates")
    timestamp: str = Field(..., description="Detection timestamp")
    session_id: Optional[str] = Field(None, description="Session identifier")

class DetectionStats(BaseModel):
    """Model for detection statistics"""
    total_detections: int = Field(..., description="Total number of detections")
    unique_cards: int = Field(..., description="Number of unique cards detected")
    most_detected_card: str = Field(..., description="Most frequently detected card")
    average_confidence: float = Field(..., description="Average detection confidence")
    detections_today: int = Field(..., description="Number of detections today")

class CardFrequency(BaseModel):
    """Model for card detection frequency"""
    card_rank: str = Field(..., description="Card rank")
    card_suit: str = Field(..., description="Card suit")
    count: int = Field(..., description="Number of times detected")
    avg_confidence: float = Field(..., description="Average confidence for this card")

class TimelineData(BaseModel):
    """Model for detection timeline data"""
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    count: int = Field(..., description="Number of detections on this date")

class WebSocketMessage(BaseModel):
    """Base model for WebSocket messages"""
    type: str = Field(..., description="Message type")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class FrameMessage(WebSocketMessage):
    """WebSocket message for sending camera frames"""
    type: str = Field(default="frame", description="Message type")
    image: str = Field(..., description="Base64 encoded image data")
    session_id: Optional[str] = Field(None, description="Session identifier")

class DetectionMessage(WebSocketMessage):
    """WebSocket message for detection results"""
    type: str = Field(default="detection", description="Message type")
    detections: List[CardDetection] = Field(..., description="Detected cards")
    processing_time: float = Field(..., description="Processing time in seconds")

class ErrorMessage(WebSocketMessage):
    """WebSocket message for errors"""
    type: str = Field(default="error", description="Message type")
    error: str = Field(..., description="Error message")
    code: Optional[str] = Field(None, description="Error code")

class StatusMessage(WebSocketMessage):
    """WebSocket message for status updates"""
    type: str = Field(default="status", description="Message type")
    status: str = Field(..., description="Current status")
    message: Optional[str] = Field(None, description="Status message")

class ModelInfo(BaseModel):
    """Model information response"""
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_type: str = Field(..., description="Type of model being used")
    supported_cards: List[str] = Field(..., description="List of supported card names")
    average_processing_time: float = Field(..., description="Average processing time")
    performance_stats: Optional[Dict[str, Any]] = Field(None, description="Additional performance statistics")

class HistoryFilter(BaseModel):
    """Filter parameters for detection history"""
    limit: int = Field(default=50, ge=1, le=1000, description="Maximum number of records to return")
    offset: int = Field(default=0, ge=0, description="Number of records to skip")
    card_rank: Optional[str] = Field(None, description="Filter by card rank")
    card_suit: Optional[str] = Field(None, description="Filter by card suit")
    start_date: Optional[str] = Field(None, description="Start date filter (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date filter (YYYY-MM-DD)")
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum confidence threshold")

class ExportRequest(BaseModel):
    """Request model for data export"""
    format: str = Field(default="json", description="Export format (json, csv)")
    filter: Optional[HistoryFilter] = Field(None, description="Filter parameters")

class HealthCheck(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")
    database_connected: bool = Field(..., description="Database connection status")
    model_loaded: bool = Field(..., description="ML model status")
    uptime: Optional[float] = Field(None, description="Service uptime in seconds")

class ConfigUpdate(BaseModel):
    """Model for updating configuration"""
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Detection confidence threshold")
    nms_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Non-maximum suppression threshold")
    max_frame_rate: Optional[int] = Field(None, ge=1, le=60, description="Maximum frame rate for processing")
    template_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Template matching threshold")

class BatchDetectionRequest(BaseModel):
    """Request model for batch detection"""
    images: List[str] = Field(..., description="List of base64 encoded images")
    session_id: Optional[str] = Field(None, description="Session identifier")
    save_to_history: bool = Field(default=True, description="Whether to save results to history")

class BatchDetectionResult(BaseModel):
    """Response model for batch detection"""
    success: bool = Field(..., description="Whether batch detection was successful")
    results: List[DetectionResult] = Field(..., description="Detection results for each image")
    total_processing_time: float = Field(..., description="Total processing time for all images")
    successful_detections: int = Field(..., description="Number of successful detections")
    failed_detections: int = Field(..., description="Number of failed detections")

# Response models for common API responses
class SuccessResponse(BaseModel):
    """Generic success response"""
    success: bool = Field(default=True, description="Operation success status")
    message: str = Field(..., description="Success message")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional response data")

class ErrorResponse(BaseModel):
    """Generic error response"""
    success: bool = Field(default=False, description="Operation success status")
    error: str = Field(..., description="Error message")
    code: Optional[str] = Field(None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")