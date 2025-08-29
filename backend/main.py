import asyncio
import base64
import json
import logging
import os
import time
from datetime import datetime, timedelta
from io import BytesIO
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from app.core.config import settings
from app.ml.card_detector import CardDetector
from app.models.database import Database
from app.models.schemas import (
    DetectionRequest,
    DetectionResult,
    CardDetection,
    DetectionHistory,
    DetectionStats,
    CardFrequency,
    TimelineData,
    WebSocketMessage,
    FrameMessage,
    DetectionMessage,
    ErrorMessage,
    StatusMessage,
    ModelInfo,
    HistoryFilter,
    ExportRequest,
    HealthCheck,
    ConfigUpdate,
    BatchDetectionRequest,
    BatchDetectionResult,
    SuccessResponse,
    ErrorResponse
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Playing Card Detector API",
    description="Real-time playing card detection using OpenCV and Machine Learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
# settings is already imported from config
card_detector = CardDetector()
database = Database()

# Performance tracking
performance_stats = {
    "total_detections": 0,
    "total_processing_time": 0.0,
    "average_processing_time": 0.0,
    "last_detection_time": None
}


class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast(self, message: dict):
        disconnected_clients = []
        for client_id, connection in self.active_connections.items():
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)


manager = ConnectionManager()
processing_locks: Dict[str, asyncio.Lock] = {}
last_frame_time: Dict[str, float] = {}


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        # Initialize database
        await database.init_db()
        logger.info("Database initialized successfully")
        
        # Load card detector models
        await card_detector.load_model()
        logger.info("Card detector models loaded successfully")
        
        logger.info("Application startup completed")
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    try:
        await database.close()
        logger.info("Application shutdown completed")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")


def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 image to OpenCV format"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        
        # Convert to PIL Image
        pil_image = Image.open(BytesIO(image_data))
        
        # Convert to OpenCV format (BGR)
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return opencv_image
    except Exception as e:
        logger.error(f"Error decoding base64 image: {e}")
        raise ValueError(f"Invalid image data: {e}")


async def process_detection(image: np.ndarray, client_id: str = None) -> DetectionResult:
    """Process card detection on image"""
    start_time = time.time()
    
    try:
        # Run detection
        raw_detections = await card_detector.detect_cards(image)
        # Normalize detections into CardDetection models
        detections: List[CardDetection] = []
        for det in raw_detections:
            try:
                detections.append(CardDetection(**det))
            except Exception as e:
                logger.error(f"Invalid detection format skipped: {det} error: {e}")
        
        processing_time = time.time() - start_time
        
        # Update performance stats
        performance_stats["total_detections"] += 1
        performance_stats["total_processing_time"] += processing_time
        performance_stats["average_processing_time"] = (
            performance_stats["total_processing_time"] / performance_stats["total_detections"]
        )
        performance_stats["last_detection_time"] = datetime.now().isoformat()
        
        # Save detections to database
        for detection in detections:
            await database.save_detection(
                card_rank=detection.rank,
                card_suit=detection.suit,
                confidence=detection.confidence,
                bbox=detection.bbox,
            )
        
        result = DetectionResult(
            success=True,
            detections=detections,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Processed detection: {len(detections)} cards found in {processing_time:.3f}s")
        return result
        
    except Exception as e:
        logger.error(f"Detection processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


# WebSocket endpoint
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    # Initialize per-client lock and timing
    if client_id not in processing_locks:
        processing_locks[client_id] = asyncio.Lock()
    if client_id not in last_frame_time:
        last_frame_time[client_id] = 0.0
    
    try:
        # Send initial status
        await manager.send_personal_message({
            "type": "status",
            "status": "connected",
            "message": "WebSocket connection established",
            "timestamp": datetime.now().isoformat()
        }, client_id)
        
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "frame":
                try:
                    # Basic server-side frame throttling
                    now = time.time()
                    min_interval = 1.0 / max(settings.MAX_FRAME_RATE, 1)
                    if now - last_frame_time.get(client_id, 0.0) < min_interval:
                        # Drop frame silently to protect server
                        continue
                    last_frame_time[client_id] = now

                    # Decode and process image with per-client lock to avoid overlap
                    image_data = message.get("image")
                    if not image_data:
                        raise ValueError("No image data provided")
                    
                    image = decode_base64_image(image_data)
                    async with processing_locks[client_id]:
                        result = await process_detection(image, client_id)
                    
                    # Send detection result
                    await manager.send_personal_message({
                        "type": "detection",
                        "detections": [det.dict() for det in result.detections],
                        "processing_time": result.processing_time,
                        "timestamp": result.timestamp
                    }, client_id)
                    
                except Exception as e:
                    logger.error(f"Frame processing error for {client_id}: {e}")
                    await manager.send_personal_message({
                        "type": "error",
                        "message": str(e),
                        "timestamp": datetime.now().isoformat()
                    }, client_id)
            
            elif message.get("type") == "ping":
                # Respond to ping
                await manager.send_personal_message({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }, client_id)
    
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        manager.disconnect(client_id)


# REST API Endpoints

@app.get("/", response_model=HealthCheck)
async def root():
    """Root endpoint with health check"""
    return HealthCheck(
        status="healthy",
        message="Playing Card Detector API is running",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        await database.get_stats()
        database_connected = True
        
        # Check if model is loaded
        model_loaded = hasattr(card_detector, 'model') and card_detector.model is not None
        
        return HealthCheck(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            version="1.0.0",
            database_connected=database_connected,
            model_loaded=model_loaded
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@app.post("/api/detect", response_model=DetectionResult)
async def detect_cards_endpoint(request: DetectionRequest):
    """Single image detection endpoint"""
    try:
        image = decode_base64_image(request.image_data)
        result = await process_detection(image)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Detection endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Detection failed")


@app.post("/api/detect/batch", response_model=BatchDetectionResult)
async def batch_detect_cards(request: BatchDetectionRequest):
    """Batch detection endpoint"""
    try:
        results = []
        total_processing_time = 0.0
        
        for i, image_data in enumerate(request.images):
            try:
                image = decode_base64_image(image_data)
                result = await process_detection(image)
                results.append(result)
                total_processing_time += result.processing_time
            except Exception as e:
                logger.error(f"Error processing image {i}: {e}")
                # Continue with other images
                continue
        
        return BatchDetectionResult(
            results=results,
            total_images=len(request.images),
            successful_detections=len(results),
            total_processing_time=total_processing_time,
            average_processing_time=total_processing_time / len(results) if results else 0.0
        )
    except Exception as e:
        logger.error(f"Batch detection error: {e}")
        raise HTTPException(status_code=500, detail="Batch detection failed")


@app.get("/api/history", response_model=List[DetectionHistory])
async def get_detection_history(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    rank: Optional[str] = Query(None),
    suit: Optional[str] = Query(None),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None)
):
    """Get detection history with filtering"""
    try:
        history = await database.get_detection_history(
            limit=limit,
            offset=offset,
            rank=rank,
            suit=suit,
            min_confidence=min_confidence,
            start_date=start_date,
            end_date=end_date
        )
        return history
    except Exception as e:
        logger.error(f"History endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch history")


@app.get("/api/stats", response_model=DetectionStats)
async def get_detection_stats(days: int = Query(30, ge=1, le=365)):
    """Get detection statistics"""
    try:
        stats = await database.get_stats(days=days)
        return stats
    except Exception as e:
        logger.error(f"Stats endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch statistics")


@app.get("/api/card-frequency")
async def get_card_frequency(days: int = Query(30, ge=1, le=365)):
    """Get card detection frequency"""
    try:
        frequency = await database.get_card_frequency(days=days)
        return {
            "cards": frequency,
            "total_unique_cards": len(frequency),
            "most_frequent": frequency[0].card_name if frequency else None,
            "least_frequent": frequency[-1].card_name if frequency else None
        }
    except Exception as e:
        logger.error(f"Card frequency endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch card frequency")


@app.get("/api/timeline")
async def get_detection_timeline(days: int = Query(7, ge=1, le=30)):
    """Get detection timeline data"""
    try:
        # Generate hourly timeline for the specified days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        timeline_data = []
        current_date = start_date
        
        while current_date <= end_date:
            for hour in range(24):
                # Get detections for this hour (simplified)
                hour_start = current_date.replace(hour=hour, minute=0, second=0, microsecond=0)
                hour_end = hour_start + timedelta(hours=1)
                
                # This is a simplified version - in a real app, you'd query the database
                timeline_data.append(TimelineData(
                    date=current_date.strftime("%Y-%m-%d"),
                    hour=hour,
                    count=0,  # Would be actual count from database
                    avg_confidence=0.0  # Would be actual average from database
                ))
            
            current_date += timedelta(days=1)
        
        return {"timeline": timeline_data}
    except Exception as e:
        logger.error(f"Timeline endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch timeline")


@app.delete("/api/history", response_model=SuccessResponse)
async def clear_detection_history():
    """Clear all detection history"""
    try:
        await database.clear_history()
        return SuccessResponse(message="Detection history cleared successfully")
    except Exception as e:
        logger.error(f"Clear history error: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear history")


@app.post("/api/export", response_model=SuccessResponse)
async def export_detection_data(request: ExportRequest):
    """Export detection data"""
    try:
        file_path = await database.export_data(
            format=request.format,
            start_date=request.start_date,
            end_date=request.end_date,
            filename=request.filename
        )
        return SuccessResponse(message=f"Data exported to {file_path}")
    except Exception as e:
        logger.error(f"Export error: {e}")
        raise HTTPException(status_code=500, detail="Failed to export data")


@app.get("/api/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model"""
    try:
        stats = card_detector.get_performance_stats()
        
        return ModelInfo(
            model_type=stats.get("model_type", "OpenCV Template Matching"),
            model_path=stats.get("model_path", "N/A"),
            input_size=stats.get("input_size", [640, 480]),
            classes=stats.get("classes", []),
            inference_time=stats.get("avg_inference_time", 0.0),
            total_detections=performance_stats["total_detections"],
            success_rate=stats.get("success_rate", 0.0)
        )
    except Exception as e:
        logger.error(f"Model info error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model information")


@app.post("/api/config", response_model=SuccessResponse)
async def update_config(config: ConfigUpdate):
    """Update application configuration"""
    try:
        # Update settings (this is a simplified version)
        logger.info(f"Config update requested: {config.dict()}")
        return SuccessResponse(message="Configuration updated successfully")
    except Exception as e:
        logger.error(f"Config update error: {e}")
        raise HTTPException(status_code=500, detail="Failed to update configuration")


@app.get("/api/performance")
async def get_performance_stats():
    """Get real-time performance statistics"""
    try:
        return {
            "total_detections": performance_stats["total_detections"],
            "average_processing_time": performance_stats["average_processing_time"],
            "last_detection_time": performance_stats["last_detection_time"],
            "active_connections": len(manager.active_connections),
            "model_stats": card_detector.get_performance_stats()
        }
    except Exception as e:
        logger.error(f"Performance stats error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get performance statistics")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )