import sqlite3
import aiosqlite
from datetime import datetime, date
from typing import List, Dict, Optional
import json
import asyncio
from pathlib import Path

from app.core.config import settings

class Database:
    """Database manager for card detection data"""
    
    def __init__(self):
        self.db_path = "card_detector.db"
        self.connection = None
        
    async def init_db(self):
        """Initialize database and create tables"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Create detections table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS detections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        card_rank TEXT NOT NULL,
                        card_suit TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        bbox TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        session_id TEXT,
                        image_hash TEXT
                    )
                """)
                
                # Create detection_stats table for analytics
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS detection_stats (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date DATE NOT NULL,
                        total_detections INTEGER DEFAULT 0,
                        unique_cards INTEGER DEFAULT 0,
                        average_confidence REAL DEFAULT 0.0,
                        most_detected_card TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes for better performance
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_detections_timestamp 
                    ON detections(timestamp)
                """)
                
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_detections_card 
                    ON detections(card_rank, card_suit)
                """)
                
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_stats_date 
                    ON detection_stats(date)
                """)
                
                await db.commit()
                print("✅ Database initialized successfully")
                
        except Exception as e:
            print(f"❌ Database initialization error: {e}")
            raise
    
    async def save_detection(self, card_rank: str, card_suit: str, confidence: float, 
                           bbox: List[int], session_id: Optional[str] = None, 
                           image_hash: Optional[str] = None) -> int:
        """Save a card detection to the database"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                bbox_json = json.dumps(bbox)
                
                cursor = await db.execute("""
                    INSERT INTO detections (card_rank, card_suit, confidence, bbox, session_id, image_hash)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (card_rank, card_suit, confidence, bbox_json, session_id, image_hash))
                
                detection_id = cursor.lastrowid
                await db.commit()
                
                # Update daily stats
                await self._update_daily_stats(db)
                
                return detection_id
                
        except Exception as e:
            print(f"Error saving detection: {e}")
            raise
    
    async def get_detection_history(self, limit: int = 50, offset: int = 0, 
                                  rank: Optional[str] = None, 
                                  suit: Optional[str] = None,
                                  min_confidence: Optional[float] = None,
                                  start_date: Optional[str] = None,
                                  end_date: Optional[str] = None) -> List[Dict]:
        """Get detection history with optional filtering"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                query = """
                    SELECT id, card_rank, card_suit, confidence, bbox, timestamp, session_id
                    FROM detections
                """
                params = []
                
                # Add filters
                conditions = []
                if rank:
                    conditions.append("card_rank = ?")
                    params.append(rank)
                if suit:
                    conditions.append("card_suit = ?")
                    params.append(suit)
                if min_confidence:
                    conditions.append("confidence >= ?")
                    params.append(min_confidence)
                if start_date:
                    conditions.append("DATE(timestamp) >= ?")
                    params.append(start_date)
                if end_date:
                    conditions.append("DATE(timestamp) <= ?")
                    params.append(end_date)
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                
                query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                
                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()
                    
                    detections = []
                    for row in rows:
                        detections.append({
                            'id': row[0],
                            'card_rank': row[1],
                            'card_suit': row[2],
                            'confidence': row[3],
                            'bbox': json.loads(row[4]),
                            'timestamp': row[5],
                            'session_id': row[6]
                        })
                    
                    return detections
                    
        except Exception as e:
            print(f"Error fetching detection history: {e}")
            raise
    
    async def get_detection_stats(self) -> Dict:
        """Get detection statistics"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Total detections
                async with db.execute("SELECT COUNT(*) FROM detections") as cursor:
                    total_count = (await cursor.fetchone())[0]
                
                # Unique cards detected
                async with db.execute("""
                    SELECT COUNT(DISTINCT card_rank || '_' || card_suit) FROM detections
                """) as cursor:
                    unique_cards = (await cursor.fetchone())[0]
                
                # Most detected card
                async with db.execute("""
                    SELECT card_rank, card_suit, COUNT(*) as count
                    FROM detections
                    GROUP BY card_rank, card_suit
                    ORDER BY count DESC
                    LIMIT 1
                """) as cursor:
                    most_detected_row = await cursor.fetchone()
                    most_detected = f"{most_detected_row[0]} of {most_detected_row[1]}" if most_detected_row else "None"
                
                # Average confidence
                async with db.execute("SELECT AVG(confidence) FROM detections") as cursor:
                    avg_confidence = (await cursor.fetchone())[0] or 0.0
                
                # Today's detections
                today = date.today().isoformat()
                async with db.execute("""
                    SELECT COUNT(*) FROM detections 
                    WHERE DATE(timestamp) = ?
                """, (today,)) as cursor:
                    today_count = (await cursor.fetchone())[0]
                
                return {
                    'total': total_count,
                    'unique_cards': unique_cards,
                    'most_detected': most_detected,
                    'avg_confidence': round(avg_confidence, 3),
                    'today_count': today_count
                }
                
        except Exception as e:
            print(f"Error fetching detection stats: {e}")
            raise
    
    async def get_stats(self, days: Optional[int] = None) -> Dict:
        """Get detection statistics with optional date filtering"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Build date filter
                date_filter = ""
                params = []
                if days:
                    date_filter = "WHERE timestamp >= datetime('now', '-{} days')".format(days)
                
                # Total detections
                query = f"SELECT COUNT(*) FROM detections {date_filter}"
                async with db.execute(query, params) as cursor:
                    total_count = (await cursor.fetchone())[0]
                
                # Unique cards detected
                query = f"SELECT COUNT(DISTINCT card_rank || '_' || card_suit) FROM detections {date_filter}"
                async with db.execute(query, params) as cursor:
                    unique_cards = (await cursor.fetchone())[0]
                
                # Most detected card
                query = f"""
                    SELECT card_rank, card_suit, COUNT(*) as count
                    FROM detections {date_filter}
                    GROUP BY card_rank, card_suit
                    ORDER BY count DESC
                    LIMIT 1
                """
                async with db.execute(query, params) as cursor:
                    most_detected_row = await cursor.fetchone()
                    most_detected = f"{most_detected_row[0]} of {most_detected_row[1]}" if most_detected_row else "None"
                
                # Average confidence
                query = f"SELECT AVG(confidence) FROM detections {date_filter}"
                async with db.execute(query, params) as cursor:
                    avg_confidence = (await cursor.fetchone())[0] or 0.0
                
                return {
                    'total_detections': total_count,
                    'unique_cards': unique_cards,
                    'most_detected_card': most_detected,
                    'average_confidence': round(avg_confidence, 3),
                    'detections_today': 0  # Will be calculated separately if needed
                }
                
        except Exception as e:
            print(f"Error fetching stats: {e}")
            raise
    
    async def get_card_frequency(self, limit: int = 10, days: Optional[int] = None) -> List[Dict]:
        """Get most frequently detected cards"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Build date filter
                date_filter = ""
                params = []
                if days:
                    date_filter = "WHERE timestamp >= datetime('now', '-{} days')".format(days)
                
                query = f"""
                    SELECT card_rank, card_suit, COUNT(*) as count, AVG(confidence) as avg_confidence
                    FROM detections {date_filter}
                    GROUP BY card_rank, card_suit
                    ORDER BY count DESC
                    LIMIT ?
                """
                params.append(limit)
                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()
                    
                    frequency_data = []
                    for row in rows:
                        frequency_data.append({
                            'card_rank': row[0],
                            'card_suit': row[1],
                            'count': row[2],
                            'avg_confidence': round(row[3], 3)
                        })
                    
                    return frequency_data
                    
        except Exception as e:
            print(f"Error fetching card frequency: {e}")
            raise
    
    async def get_detection_timeline(self, days: int = 7) -> List[Dict]:
        """Get detection timeline for the last N days"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute("""
                    SELECT DATE(timestamp) as date, COUNT(*) as count
                    FROM detections
                    WHERE timestamp >= datetime('now', '-{} days')
                    GROUP BY DATE(timestamp)
                    ORDER BY date
                """.format(days)) as cursor:
                    rows = await cursor.fetchall()
                    
                    timeline_data = []
                    for row in rows:
                        timeline_data.append({
                            'date': row[0],
                            'count': row[1]
                        })
                    
                    return timeline_data
                    
        except Exception as e:
            print(f"Error fetching detection timeline: {e}")
            raise
    
    async def clear_history(self, older_than_days: Optional[int] = None):
        """Clear detection history"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                if older_than_days:
                    await db.execute("""
                        DELETE FROM detections 
                        WHERE timestamp < datetime('now', '-{} days')
                    """.format(older_than_days))
                else:
                    await db.execute("DELETE FROM detections")
                
                await db.commit()
                print("✅ Detection history cleared")
                
        except Exception as e:
            print(f"Error clearing history: {e}")
            raise
    
    async def _update_daily_stats(self, db):
        """Update daily statistics"""
        try:
            today = date.today().isoformat()
            
            # Check if stats for today already exist
            async with db.execute("""
                SELECT id FROM detection_stats WHERE date = ?
            """, (today,)) as cursor:
                existing = await cursor.fetchone()
            
            # Calculate today's stats
            async with db.execute("""
                SELECT COUNT(*), COUNT(DISTINCT card_rank || '_' || card_suit), AVG(confidence)
                FROM detections
                WHERE DATE(timestamp) = ?
            """, (today,)) as cursor:
                stats_row = await cursor.fetchone()
                total_detections, unique_cards, avg_confidence = stats_row
            
            # Get most detected card today
            async with db.execute("""
                SELECT card_rank, card_suit, COUNT(*) as count
                FROM detections
                WHERE DATE(timestamp) = ?
                GROUP BY card_rank, card_suit
                ORDER BY count DESC
                LIMIT 1
            """, (today,)) as cursor:
                most_detected_row = await cursor.fetchone()
                most_detected = f"{most_detected_row[0]} of {most_detected_row[1]}" if most_detected_row else "None"
            
            if existing:
                # Update existing record
                await db.execute("""
                    UPDATE detection_stats
                    SET total_detections = ?, unique_cards = ?, average_confidence = ?, most_detected_card = ?
                    WHERE date = ?
                """, (total_detections, unique_cards, avg_confidence or 0.0, most_detected, today))
            else:
                # Insert new record
                await db.execute("""
                    INSERT INTO detection_stats (date, total_detections, unique_cards, average_confidence, most_detected_card)
                    VALUES (?, ?, ?, ?, ?)
                """, (today, total_detections, unique_cards, avg_confidence or 0.0, most_detected))
            
            await db.commit()
            
        except Exception as e:
            print(f"Error updating daily stats: {e}")
    
    async def export_data(self, format: str = "json") -> str:
        """Export detection data"""
        try:
            detections = await self.get_detection_history(limit=10000)
            
            if format.lower() == "json":
                return json.dumps(detections, indent=2, default=str)
            elif format.lower() == "csv":
                import csv
                import io
                
                output = io.StringIO()
                if detections:
                    writer = csv.DictWriter(output, fieldnames=detections[0].keys())
                    writer.writeheader()
                    for detection in detections:
                        # Convert bbox list to string for CSV
                        detection_copy = detection.copy()
                        detection_copy['bbox'] = str(detection_copy['bbox'])
                        writer.writerow(detection_copy)
                
                return output.getvalue()
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            print(f"Error exporting data: {e}")
            raise