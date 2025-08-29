from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import base64
import json
from datetime import datetime
from typing import List, Optional
import asyncio
from io import BytesIO
from PIL import Image

from app.core.config import settings
from app.ml.card_detector import CardDetector
from app.models.database import Database
from app.models.schemas import DetectionResult, DetectionHistory

app = FastAPI(
    title="Playing Card Detector API",
    description="Real-time playing card detection using computer vision",
    version="1.0.0"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
card_detector = CardDetector()
database = Database()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    """Initialize database and load ML models on startup"""
    await database.init_db()
    await card_detector.load_model()
    print("🚀 Playing Card Detector API started successfully!")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Playing Card Detector API",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/detect", response_model=DetectionResult)
async def detect_card(file: UploadFile = File(...)):
    """Detect playing cards in uploaded image"""
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        image_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(image_array.shape) == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Detect cards
        detections = await card_detector.detect_cards(image_array)
        
        # Save to database
        for detection in detections:
            await database.save_detection(
                card_rank=detection['rank'],
                card_suit=detection['suit'],
                confidence=detection['confidence'],
                bbox=detection['bbox']
            )
        
        return DetectionResult(
            success=True,
            detections=detections,
            timestamp=datetime.now().isoformat(),
            processing_time=card_detector.last_processing_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    """WebSocket endpoint for real-time card detection"""
    await manager.connect(websocket)
    try:
        while True:
            # Receive base64 encoded image from frontend
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message['type'] == 'frame':
                # Decode base64 image
                image_data = base64.b64decode(message['image'].split(',')[1])
                image = Image.open(BytesIO(image_data))
                image_array = np.array(image)
                
                # Convert RGB to BGR for OpenCV
                if len(image_array.shape) == 3:
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                
                # Detect cards
                detections = await card_detector.detect_cards(image_array)
                
                # Send results back
                response = {
                    'type': 'detection',
                    'detections': detections,
                    'timestamp': datetime.now().isoformat(),
                    'processing_time': card_detector.last_processing_time
                }
                
                await manager.send_personal_message(json.dumps(response), websocket)
                
                # Save significant detections to database
                for detection in detections:
                    if detection['confidence'] > 0.5:  # Save medium to high-confidence detections
                        await database.save_detection(
                            card_rank=detection['rank'],
                            card_suit=detection['suit'],
                            confidence=detection['confidence'],
                            bbox=detection['bbox']
                        )
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@app.get("/api/history", response_model=List[DetectionHistory])
async def get_detection_history(limit: int = 50, offset: int = 0):
    """Get detection history from database"""
    try:
        history = await database.get_detection_history(limit=limit, offset=offset)
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")

@app.get("/api/stats")
async def get_detection_stats():
    """Get detection statistics"""
    try:
        stats = await database.get_detection_stats()
        return {
            "total_detections": stats['total'],
            "unique_cards": stats['unique_cards'],
            "most_detected_card": stats['most_detected'],
            "average_confidence": stats['avg_confidence'],
            "detections_today": stats['today_count']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch stats: {str(e)}")

@app.delete("/api/history")
async def clear_detection_history():
    """Clear all detection history"""
    try:
        await database.clear_history()
        return {"message": "Detection history cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {str(e)}")

@app.get("/api/model/info")
async def get_model_info():
    """Get information about the loaded model"""
    return {
        "model_loaded": card_detector.model_loaded,
        "model_type": card_detector.model_type,
        "supported_cards": card_detector.supported_cards,
        "average_processing_time": card_detector.average_processing_time
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )