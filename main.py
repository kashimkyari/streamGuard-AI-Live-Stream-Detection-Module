import threading
import time
import subprocess
import cv2
import numpy as np
import os
import logging
from datetime import datetime, timezone, timedelta
import requests
from collections import defaultdict, deque
import hashlib
import tempfile
import m3u8
import io
import wave
import re
import base64
import json

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Updated SQLAlchemy imports (SQLAlchemy 2.0+)
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker, scoped_session, joinedload

# AI model imports with error handling
try:
    from ultralytics import YOLO
    yolo_available = True
except ImportError as e:
    logging.error(f"Failed to import YOLO: {e}")
    yolo_available = False

# PyAV for m3u8 handling (required)
try:
    import av
    pyav_available = True
except ImportError as e:
    logging.error(f"Failed to import PyAV: {e}")
    pyav_available = False
    raise ImportError("PyAV is required for this application")

# Whisper for audio transcription (required)
try:
    import whisper
    whisper_available = True
except ImportError as e:
    logging.error(f"Failed to import whisper: {e}")
    whisper_available = False
    raise ImportError("Whisper is required for audio transcription")

# Import models from your existing models.py
from models import Stream, ChaturbateStream, StripchatStream, FlaggedObject, ChatKeyword, DetectionLog, Assignment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    logger.error("DATABASE_URL environment variable not set")
    exit(1)

engine = create_engine(DATABASE_URL)
Session = scoped_session(sessionmaker(bind=engine))

# Load AI models with logging
yolo_model = None
whisper_model = None

if yolo_available:
    try:
        logger.info("Loading YOLOv8n model...")
        yolo_model = YOLO("yolov8n.pt")
        logger.info("YOLOv8n model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load YOLOv8n model: {e}")
        yolo_available = False

if whisper_available:
    try:
        logger.info("Loading Whisper model...")
        whisper_model = whisper.load_model("base")
        logger.info("Whisper model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        whisper_available = False

# Configuration
CHECK_INTERVAL = int(os.getenv('CHECK_INTERVAL', 10))
NEW_STREAM_CHECK_INTERVAL = int(os.getenv('NEW_STREAM_CHECK_INTERVAL', 30))
MAX_CONCURRENT_STREAMS = int(os.getenv('MAX_CONCURRENT_STREAMS', 4))
AUDIO_SEGMENT_DURATION = int(os.getenv('AUDIO_SEGMENT_DURATION', 30))
TRANSCRIPTION_INTERVAL = int(os.getenv('TRANSCRIPTION_INTERVAL', 3))
DETECTION_COOLDOWN = int(os.getenv('DETECTION_COOLDOWN', 300))
DETECTION_CONFIDENCE_THRESHOLD = float(os.getenv('DETECTION_CONFIDENCE_THRESHOLD', 0.5))
MAX_DETECTIONS_PER_MINUTE = int(os.getenv('MAX_DETECTIONS_PER_MINUTE', 1))
CHAT_CHECK_INTERVAL = int(os.getenv('CHAT_CHECK_INTERVAL', 10))

logger.info(f"Configuration loaded - Check interval: {CHECK_INTERVAL}s, Detection cooldown: {DETECTION_COOLDOWN}s, Chat check interval: {CHAT_CHECK_INTERVAL}s")

# Global variables for detection management
last_visual_alerts = {}
last_audio_alerts = {}
last_chat_alerts = {}
detection_managers = {}
audio_transcribers = {}

def get_session():
    return Session()

def get_stream_info(stream_url):
    """Get platform and streamer info from stream URL."""
    try:
        with Session() as session:
            stream = session.execute(select(Stream).filter_by(room_url=stream_url)).scalar_one_or_none()
            if stream:
                return stream.type.lower(), stream.streamer_username, stream.id

            cb_stream = session.execute(select(ChaturbateStream).filter_by(chaturbate_m3u8_url=stream_url)).scalar_one_or_none()
            if cb_stream:
                stream = session.execute(select(Stream).filter_by(id=cb_stream.id)).scalar_one_or_none()
                return 'chaturbate', stream.streamer_username if stream else 'unknown', stream.id if stream else None

            sc_stream = session.execute(select(StripchatStream).filter_by(stripchat_m3u8_url=stream_url)).scalar_one_or_none()
            if sc_stream:
                stream = session.execute(select(Stream).filter_by(id=sc_stream.id)).scalar_one_or_none()
                return 'stripchat', stream.streamer_username if stream else 'unknown', stream.id if stream else None

            logger.warning(f"No stream found for URL: {stream_url}")
            return 'unknown', 'unknown', None
    except Exception as e:
        logger.error(f"Error retrieving stream info for {stream_url}: {e}")
        return 'unknown', 'unknown', None

def get_stream_assignment(stream_url):
    """Get assignment info for a stream."""
    try:
        with Session() as session:
            stream = session.execute(select(Stream).filter_by(room_url=stream_url)).scalar_one_or_none()
            if not stream:
                cb_stream = session.execute(select(ChaturbateStream).filter_by(chaturbate_m3u8_url=stream_url)).scalar_one_or_none()
                if cb_stream:
                    stream = session.execute(select(Stream).filter_by(id=cb_stream.id)).scalar_one_or_none()
                else:
                    sc_stream = session.execute(select(StripchatStream).filter_by(stripchat_m3u8_url=stream_url)).scalar_one_or_none()
                    if sc_stream:
                        stream = session.execute(select(Stream).filter_by(id=sc_stream.id)).scalar_one_or_none()
            
            if not stream:
                logger.warning(f"No stream found for URL: {stream_url}")
                return None, None

            query = session.execute(
                select(Assignment)
                .options(joinedload(Assignment.agent), joinedload(Assignment.stream))
                .filter_by(stream_id=stream.id)
            ).scalars().all()
            
            if not query:
                logger.info(f"No assignments found for stream: {stream_url}")
                return None, None
                
            assignment = query[0]
            agent_id = assignment.agent_id
            return assignment.id, agent_id
    except Exception as e:
        logger.error(f"Error retrieving stream assignment for {stream_url}: {e}")
        return None, None

def update_stream_status(stream_id, status):
    """Update the status of a stream in the database."""
    try:
        with get_session() as session:
            stmt = select(Stream).where(Stream.id == stream_id)
            stream = session.execute(stmt).scalar_one_or_none()
            if stream:
                old_status = stream.status
                stream.status = status
                session.commit()
                logger.info(f"Stream {stream_id} status updated: {old_status} -> {status}")
            else:
                logger.warning(f"Stream {stream_id} not found in database")
    except Exception as e:
        logger.error(f"Error updating stream {stream_id} status: {e}")

def log_detection(stream_id, event_type, detection, frame=None, stream_url=None):
    """Log a detection event to the detection_logs table."""
    try:
        with get_session() as session:
            stmt = select(Stream).where(Stream.id == stream_id)
            stream = session.execute(stmt).scalar_one_or_none()
            if not stream:
                logger.warning(f"Stream {stream_id} not found for detection logging")
                return
            
            room_url = stream.room_url
            image_data = None
            details = detection
            assignment_id = None
            agent_id = None
            
            if event_type == 'object_detection':
                platform, streamer, _ = get_stream_info(stream_url)
                assignment_id, agent_id = get_stream_assignment(stream_url)
                image_b64 = None
                if frame is not None:
                    success, buffer = cv2.imencode('.jpg', frame)
                    if success:
                        image_data = buffer.tobytes()
                        image_b64 = base64.b64encode(buffer).decode('utf-8')
                    else:
                        logger.error(f"Failed to encode frame for stream {stream_id}")
                
                details = {
                    'detections': [{
                        'class': detection['detected_item'],
                        'confidence': detection['confidence'],
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'bbox': detection.get('bbox', [])
                    }],
                    'streamer_name': streamer,
                    'platform': platform,
                    'assigned_agent': agent_id,
                    'annotated_image': image_b64
                }
            elif event_type == 'audio_detection':
                image_data = None
            elif event_type == 'chat_keyword_detection':
                platform, streamer, _ = get_stream_info(stream_url)
                assignment_id, agent_id = get_stream_assignment(stream_url)
                image_data = None
                details = {
                    'type': detection.get('type'),
                    'keyword': detection.get('keyword'),
                    'message': detection.get('message'),
                    'username': detection.get('username'),
                    'timestamp': detection.get('timestamp'),
                    'streamer_name': streamer,
                    'platform': platform,
                    'assigned_agent': agent_id
                }
            
            detection_log = DetectionLog(
                room_url=room_url,
                event_type=event_type,
                details=details,
                detection_image=image_data,
                timestamp=datetime.now(timezone.utc),
                assigned_agent=agent_id,
                assignment_id=assignment_id,
                read=False
            )
            session.add(detection_log)
            session.commit()
            logger.info(f"Detection logged for stream {stream_id}: {event_type} - {json.dumps({k: v for k, v in details.items() if k != 'annotated_image'}, default=str)}")
    except Exception as e:
        logger.error(f"Error logging detection for stream {stream_id}: {e}")
        if 'session' in locals():
            session.rollback()

def get_flagged_objects():
    """Retrieve the list of flagged object names and their confidence thresholds."""
    try:
        with get_session() as session:
            stmt = select(FlaggedObject)
            objects = session.execute(stmt).scalars().all()
            flagged = {obj.object_name.lower(): float(obj.confidence_threshold) for obj in objects}
            logger.debug(f"Retrieved {len(flagged)} flagged objects: {flagged}")
            return flagged
    except Exception as e:
        logger.error(f"Error retrieving flagged objects: {e}")
        return {}

def get_flagged_keywords():
    """Retrieve the list of flagged keywords."""
    try:
        with get_session() as session:
            stmt = select(ChatKeyword)
            keywords = session.execute(stmt).scalars().all()
            flagged = [keyword.keyword.lower() for keyword in keywords]
            logger.debug(f"Retrieved {len(flagged)} flagged keywords: {flagged}")
            return flagged
    except Exception as e:
        logger.error(f"Error retrieving flagged keywords: {e}")
        return []

def check_stream_status(url):
    """Check if a stream is online by sending a HEAD request to the m3u8 URL."""
    try:
        response = requests.head(url, timeout=5)
        is_online = response.status_code == 200
        logger.debug(f"Stream status check for {url}: {response.status_code} - {'Online' if is_online else 'Offline'}")
        return is_online
    except requests.RequestException as e:
        logger.debug(f"Stream status check failed for {url}: {e}")
        return False

def fetch_chaturbate_room_uid(streamer_username):
    """Fetch Chaturbate room and broadcaster UIDs."""
    url = f"https://chaturbate.com/api/chatvideocontext/{streamer_username}/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer': f'https://chaturbate.com/{streamer_username}/',
        'Connection': 'keep-alive',
    }
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, headers=headers, timeout=10, verify=False)
            response.raise_for_status()
            data = response.json()
            broadcaster_uid = data.get('broadcaster_uid')
            room_uid = data.get('room_uid')
            logger.debug(f"Fetched Chaturbate UIDs for {streamer_username}: broadcaster_uid={broadcaster_uid}, room_uid={room_uid}")
            return broadcaster_uid, room_uid
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed for Chaturbate room UID fetch: {e}")
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt)
    logger.error(f"Failed to fetch Chaturbate room UID for {streamer_username} after {max_attempts} attempts")
    return None, None

def fetch_chaturbate_chat(room_url, streamer, broadcaster_uid):
    """Fetch Chaturbate chat messages."""
    if not broadcaster_uid:
        logger.warning(f"No broadcaster UID for {room_url}")
        return []
    url = "https://chaturbate.com/push_service/room_history/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer': f'https://chaturbate.com/{streamer}/',
        'X-Requested-With': 'XMLHttpRequest',
        'Content-Type': 'multipart/form-data; boundary=----WebKitFormBoundary428c342290b0a909',
        'Origin': 'https://chaturbate.com',
        'Connection': 'keep-alive',
    }
    data = (
        '------WebKitFormBoundary428c342290b0a909\r\n'
        f'Content-Disposition: form-data; name="topics"\r\n\r\n'
        f'{{"RoomMessageTopic#RoomMessageTopic:{broadcaster_uid}":{{"broadcaster_uid":"{broadcaster_uid}"}}}}\r\n'
        '------WebKitFormBoundary428c342290b0a909\r\n'
        'Content-Disposition: form-data; name="csrfmiddlewaretoken"\r\n\r\n'
        'NdFODN04i4jCUKVTPs3JyAwxsVnuxiy0\r\n'
        '------WebKitFormBoundary428c342290b0a909--\r\n'
    )
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            response = requests.post(url, headers=headers, data=data, timeout=15, verify=False)
            response.raise_for_status()
            chat_data = response.json()
            messages = []
            for key, msg_data in chat_data.items():
                if f"RoomMessageTopic#RoomMessageTopic:{broadcaster_uid}" in msg_data:
                    msg = msg_data[f"RoomMessageTopic#RoomMessageTopic:{broadcaster_uid}"]
                    messages.append({
                        "username": msg.get("from_user", {}).get("username", "unknown"),
                        "message": msg.get("message", ""),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
            logger.info(f"Fetched {len(messages)} Chaturbate chat messages for {streamer} at {room_url}")
            return messages
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed for Chaturbate chat fetch: {e}")
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt)
    logger.error(f"Failed to fetch Chaturbate chat for {streamer} at {room_url} after {max_attempts} attempts")
    return []

def fetch_stripchat_chat(room_url, streamer):
    """Fetch Stripchat chat messages."""
    url = f"https://stripchat.com/api/front/v2/models/username/{streamer}/chat"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer': f'https://stripchat.com/{streamer}',
        'content-type': 'application/json',
        'front-version': '11.1',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'Connection': 'keep-alive',
    }
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, headers=headers, timeout=15, verify=False)
            response.raise_for_status()
            chat_data = response.json()
            messages = []
            for msg in chat_data.get('messages', []):
                message_type = msg.get("type", "")
                details = msg.get("details", {})
                body = details.get("body", "")
                if message_type == "text" or (message_type == "tip" and body):
                    messages.append({
                        "username": msg.get("userData", {}).get("username", "unknown"),
                        "message": body,
                        "timestamp": msg.get("createdAt", datetime.now(timezone.utc).isoformat())
                    })
            logger.info(f"Fetched {len(messages)} Stripchat chat messages for {streamer} at {room_url}")
            return messages
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed for Stripchat chat fetch: {e}")
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt)
    logger.error(f"Failed to fetch Stripchat chat for {streamer} at {room_url} after {max_attempts} attempts")
    return []

def fetch_chat_messages(stream_id, room_url):
    """Fetch chat messages based on platform."""
    try:
        platform, streamer, _ = get_stream_info(room_url)
        logger.info(f"Fetching chat for {streamer} on {platform} for stream {stream_id}")
        
        if platform == "chaturbate":
            broadcaster_uid, _ = fetch_chaturbate_room_uid(streamer)
            return fetch_chaturbate_chat(room_url, streamer, broadcaster_uid)
        elif platform == "stripchat":
            return fetch_stripchat_chat(room_url, streamer)
        else:
            logger.warning(f"Unsupported platform {platform} for {room_url}")
            return []
    except Exception as e:
        logger.error(f"Chat fetch error for {room_url}: {e}")
        return []

def process_chat_messages(messages, room_url, detection_manager):
    """Process chat messages for flagged keywords."""
    try:
        keywords = get_flagged_keywords()
        if not keywords:
            logger.debug(f"No flagged keywords found for {room_url}")
            return []

        detected = []
        now = datetime.now(timezone.utc)

        for msg in messages:
            text = msg.get("message", "").lower()
            user = msg.get("username", "unknown")
            timestamp = msg.get("timestamp", now.isoformat())

            for keyword in keywords:
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, text):
                    detection = {
                        "type": "keyword",
                        "keyword": keyword,
                        "message": text,
                        "username": user,
                        "timestamp": timestamp
                    }
                    should_log, reason = detection_manager.should_log_detection(keyword, 1.0, None)
                    if should_log:
                        detected.append(detection)
                        logger.info(f"Chat keyword detected in {room_url}: {keyword} from {user}")
                    else:
                        logger.debug(f"Chat detection skipped for {room_url}: {keyword} - {reason}")

        return detected
    except Exception as e:
        logger.error(f"Error processing chat messages for {room_url}: {e}")
        return []

class DetectionManager:
    """Manages detections to prevent spam and duplicate alerts."""
    
    def __init__(self, stream_id):
        self.stream_id = stream_id
        self.last_detection_time = {}
        self.detection_history = defaultdict(deque)
        self.detection_zones = {}
        self.lock = threading.Lock()
    
    def _get_area_hash(self, box):
        """Create a hash of the detection area to identify similar locations."""
        if box is None:
            return hashlib.md5("chat_detection".encode()).hexdigest()[:8]
        x, y, w, h = map(int, box.xywh[0])
        zone_x = x // 100
        zone_y = y // 100
        zone_w = max(1, w // 100)
        zone_h = max(1, h // 100)
        return hashlib.md5(f"{zone_x}_{zone_y}_{zone_w}_{zone_h}".encode()).hexdigest()[:8]
    
    def should_log_detection(self, object_name, confidence, box):
        """Determine if a detection should be logged based on various criteria."""
        with self.lock:
            current_time = time.time()
            
            if confidence < DETECTION_CONFIDENCE_THRESHOLD:
                return False, "confidence_too_low"
            
            if object_name in self.last_detection_time:
                time_since_last = current_time - self.last_detection_time[object_name]
                if time_since_last < DETECTION_COOLDOWN:
                    return False, f"cooldown_active_{int(DETECTION_COOLDOWN - time_since_last)}s_remaining"
            
            area_hash = self._get_area_hash(box)
            if object_name in self.detection_zones:
                if self.detection_zones[object_name] == area_hash:
                    if object_name in self.last_detection_time:
                        time_since_last = current_time - self.last_detection_time[object_name]
                        if time_since_last < DETECTION_COOLDOWN * 2:
                            return False, "same_area_too_recent"
            
            history = self.detection_history[object_name]
            while history and current_time - history[0] > 60:
                history.popleft()
            
            if len(history) >= MAX_DETECTIONS_PER_MINUTE:
                return False, "rate_limit_exceeded"
            
            self.last_detection_time[object_name] = current_time
            self.detection_zones[object_name] = area_hash
            history.append(current_time)
            
            return True, "detection_logged"
    
    def get_detection_stats(self):
        """Get detection statistics for this stream."""
        with self.lock:
            stats = {}
            current_time = time.time()
            
            for obj_name, history in self.detection_history.items():
                recent_count = sum(1 for t in history if current_time - t < 300)
                last_detection = self.last_detection_time.get(obj_name, 0)
                time_since_last = current_time - last_detection if last_detection else 0
                
                stats[obj_name] = {
                    'total_detections': len(history),
                    'recent_detections_5min': recent_count,
                    'seconds_since_last': int(time_since_last),
                    'cooldown_active': time_since_last < DETECTION_COOLDOWN
                }
            
            return stats

class AudioTranscriber:
    """Handles audio extraction and transcription from streams using AV and Whisper."""
    
    def __init__(self, stream_id):
        self.stream_id = stream_id
        self.last_transcription = time.time()
        self.lock = threading.Lock()
        self.transcription_thread = None
    
    def extract_audio_with_av(self, m3u8_url):
        """Extract audio from m3u8 stream using PyAV."""
        temp_audio_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
            
            input_container = av.open(m3u8_url, options={
                'timeout': '10000000',
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            audio_stream = None
            for stream in input_container.streams:
                if stream.type == 'audio':
                    audio_stream = stream
                    break
            
            if not audio_stream:
                logger.warning(f"No audio stream found in {m3u8_url}")
                return None
            
            output_container = av.open(temp_audio_path, 'w')
            output_stream = output_container.add_stream('pcm_s16le', rate=16000, layout='mono')
            
            start_time = time.time()
            packets_processed = 0
            
            for packet in input_container.demux(audio_stream):
                if time.time() - start_time > AUDIO_SEGMENT_DURATION:
                    break
                
                try:
                    for frame in packet.decode():
                        if frame.sample_rate != 16000 or frame.layout.name != 'mono':
                            resampler = av.AudioResampler(
                                format='s16',
                                layout='mono',
                                rate=16000
                            )
                            frame = resampler.resample(frame)[0]
                        
                        for packet in output_stream.encode(frame):
                            output_container.mux(packet)
                        
                        packets_processed += 1
                        
                except Exception as e:
                    logger.debug(f"Error processing audio packet for stream {self.stream_id}: {e}")
                    continue
            
            for packet in output_stream.encode():
                output_container.mux(packet)
            
            output_container.close()
            input_container.close()
            
            if packets_processed == 0:
                logger.warning(f"No audio packets processed for stream {self.stream_id}")
                return None
            
            if os.path.exists(temp_audio_path) and os.path.getsize(temp_audio_path) > 1024:
                logger.debug(f"Audio extracted successfully for stream {self.stream_id}: {packets_processed} packets")
                return temp_audio_path
            else:
                logger.warning(f"Audio extraction failed for stream {self.stream_id}: file too small or empty")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting audio with AV for stream {self.stream_id}: {e}")
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass
            return None
    
    def transcribe_audio_with_whisper(self, audio_path):
        """Transcribe audio using Whisper."""
        try:
            if not whisper_model:
                logger.warning("Whisper model not available for transcription")
                return None
            
            result = whisper_model.transcribe(
                audio_path,
                language=None,
                task='transcribe',
                verbose=False
            )
            
            text = result.get('text', '').strip()
            if text:
                logger.info(f"Whisper transcription for stream {self.stream_id}: {text[:100]}...")
                return text
            else:
                logger.debug(f"No transcription text returned for stream {self.stream_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error transcribing audio with Whisper for stream {self.stream_id}: {e}")
            return None
    
    def process_audio_transcription(self, m3u8_url):
        """Main method to extract and transcribe audio."""
        with self.lock:
            current_time = time.time()
            
            if current_time - self.last_transcription < TRANSCRIPTION_INTERVAL:
                return None
            
            if self.transcription_thread and self.transcription_thread.is_alive():
                logger.debug(f"Transcription already running for stream {self.stream_id}")
                return None
            
            self.last_transcription = current_time
        
        def transcription_worker():
            """Worker function for transcription to avoid blocking main thread."""
            audio_path = None
            try:
                audio_path = self.extract_audio_with_av(m3u8_url)
                if not audio_path:
                    logger.debug(f"No audio extracted for stream {self.stream_id}")
                    return
                
                transcription = self.transcribe_audio_with_whisper(audio_path)
                
                if transcription:
                    try:
                        self.check_flagged_keywords(transcription, m3u8_url)
                    except AttributeError as e:
                        logger.error(f"AttributeError in check_flagged_keywords for stream {self.stream_id}: {e}")
                else:
                    logger.debug(f"No transcription result for stream {self.stream_id}")
                    
            except Exception as e:
                logger.error(f"Error in transcription worker for stream {self.stream_id}: {e}")
            finally:
                if audio_path and os.path.exists(audio_path):
                    try:
                        os.unlink(audio_path)
                    except Exception as e:
                        logger.warning(f"Failed to delete temporary audio file {audio_path}: {e}")
        
        self.transcription_thread = threading.Thread(target=transcription_worker, daemon=True)
        self.transcription_thread.start()
    
    def check_flagged_keywords(self, transcription, stream_url):
        """Check transcription for flagged keywords using whole-word matching."""
        try:
            with Session() as session:
                stmt = select(ChatKeyword)
                keywords = session.execute(stmt).scalars().all()
                
                flagged_keywords = []
                transcription_lower = transcription.lower()
                
                for keyword_obj in keywords:
                    keyword = keyword_obj.keyword.lower()
                    pattern = r'\b' + re.escape(keyword) + r'\b'
                    if re.search(pattern, transcription_lower):
                        flagged_keywords.append(keyword_obj.keyword)
                
                if flagged_keywords:
                    logger.info(f"Flagged keywords detected in stream {self.stream_id}: {flagged_keywords}")
                    for keyword in flagged_keywords:
                        platform, streamer, stream_id = get_stream_info(stream_url)
                        assignment_id, agent_id = get_stream_assignment(stream_url)
                        now = datetime.now(timezone.utc)
                        
                        if keyword in last_audio_alerts.get(stream_url, {}):
                            last_alert = last_audio_alerts[stream_url][keyword]
                            if (now - last_alert).total_seconds() < DETECTION_COOLDOWN:
                                continue
                        
                        last_audio_alerts.setdefault(stream_url, {})[keyword] = now
                        
                        detection = {
                            'type': 'keyword',
                            'keyword': keyword,
                            'transcript': transcription,
                            'timestamp': now.isoformat(),
                            'streamer_name': streamer,
                            'platform': platform,
                            'assigned_agent': agent_id
                        }
                        log_detection(self.stream_id, 'audio_detection', detection, stream_url=stream_url)
                            
        except Exception as e:
            logger.error(f"Error checking flagged keywords for stream {self.stream_id}: {e}")

def process_stream(stream_id, m3u8_url, stop_event):
    """Process a single stream using PyAV for both video and audio, and check chat messages."""
    logger.info(f"Starting stream processing for stream {stream_id}: {m3u8_url}")
    
    if stream_id not in detection_managers:
        detection_managers[stream_id] = DetectionManager(stream_id)
    if stream_id not in audio_transcribers:
        audio_transcribers[stream_id] = AudioTranscriber(stream_id)
    
    detection_manager = detection_managers[stream_id]
    audio_transcriber = audio_transcribers[stream_id]
    flagged_objects = get_flagged_objects()
    
    container = None
    try:
        container = av.open(m3u8_url, options={
            'timeout': '10000000',
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'reconnect': '1',
            'reconnect_streamed': '1',
            'reconnect_delay_max': '5'
        })
        
        video_stream = None
        for stream in container.streams:
            if stream.type == 'video':
                video_stream = stream
                break
        
        if not video_stream:
            logger.error(f"No video stream found for stream {stream_id}")
            update_stream_status(stream_id, 'offline')
            return
        
        last_check = time.time()
        last_audio_check = time.time()
        last_chat_check = time.time()
        frame_count = 0
        detection_count = 0
        skipped_detections = 0
        chat_detection_count = 0
        chat_skipped_detections = 0
        
        logger.info(f"Starting video processing for stream {stream_id}")
        
        for packet in container.demux(video_stream):
            if stop_event.is_set():
                break
            
            try:
                if time.time() - last_check >= CHECK_INTERVAL:
                    if not check_stream_status(m3u8_url):
                        logger.info(f"Stream {stream_id} went offline")
                        update_stream_status(stream_id, 'offline')
                        stop_event.set()
                        break
                    last_check = time.time()
                    
                    stats = detection_manager.get_detection_stats()
                    if stats:
                        logger.info(f"Stream {stream_id} detection stats: {stats}")
                
                if time.time() - last_audio_check >= TRANSCRIPTION_INTERVAL:
                    audio_transcriber.process_audio_transcription(m3u8_url)
                    last_audio_check = time.time()
                
                if time.time() - last_chat_check >= CHAT_CHECK_INTERVAL:
                    messages = fetch_chat_messages(stream_id, m3u8_url)
                    chat_detections = process_chat_messages(messages, m3u8_url, detection_manager)
                    for detection in chat_detections:
                        keyword = detection['keyword']
                        now = datetime.now(timezone.utc)
                        if keyword in last_chat_alerts.get(m3u8_url, {}):
                            last_alert = last_chat_alerts[m3u8_url][keyword]
                            if (now - last_alert).total_seconds() < DETECTION_COOLDOWN:
                                chat_skipped_detections += 1
                                logger.debug(f"Chat detection skipped for stream {stream_id}: {keyword} - cooldown_active")
                                continue
                        
                        last_chat_alerts.setdefault(m3u8_url, {})[keyword] = now
                        log_detection(stream_id, 'chat_keyword_detection', detection, stream_url=m3u8_url)
                        chat_detection_count += 1
                    last_chat_check = time.time()
                
                for frame in packet.decode():
                    if stop_event.is_set():
                        break
                    
                    frame_count += 1
                    
                    if frame_count % 30 != 0:
                        continue
                    
                    img = frame.to_ndarray(format='bgr24')
                    annotated_img = img.copy()  # Create a copy for annotation
                    
                    if yolo_available and yolo_model and flagged_objects:
                        try:
                            results = yolo_model(img, device='cpu', verbose=False)
                            detections = []
                            now = datetime.now(timezone.utc)
                            
                            for result in results:
                                if result.boxes is not None:
                                    for box in result.boxes:
                                        label = result.names[int(box.cls)].lower()
                                        confidence = float(box.conf)
                                        
                                        if label in flagged_objects and confidence >= flagged_objects[label]:
                                            should_log, reason = detection_manager.should_log_detection(label, confidence, box)
                                            
                                            if should_log:
                                                detection_count += 1
                                                logger.info(f"Object detected in stream {stream_id}: {label} (confidence: {confidence:.2f})")
                                                
                                                x, y, w, h = map(int, box.xywh[0])
                                                x1, y1 = x - w // 2, y - h // 2
                                                x2, y2 = x + w // 2, y + h // 2
                                                
                                                if label in last_visual_alerts.get(m3u8_url, {}):
                                                    last_alert = last_visual_alerts[m3u8_url][label]
                                                    if (now - last_alert).total_seconds() < DETECTION_COOLDOWN:
                                                        continue
                                                
                                                last_visual_alerts.setdefault(m3u8_url, {})[label] = now
                                                
                                                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                                cv2.putText(annotated_img, f"{label} {confidence:.2f}", (x1, y1 - 10), 
                                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                                
                                                detection = {
                                                    'detected_item': label,
                                                    'confidence': confidence,
                                                    'bbox': [x1, y1, x2, y2]
                                                }
                                                detections.append(detection)
                                                
                                                log_detection(stream_id, 'object_detection', detection, annotated_img, m3u8_url)
                                            else:
                                                skipped_detections += 1
                                                logger.debug(f"Detection skipped for stream {stream_id}: {label} - {reason}")
                                                
                        except Exception as e:
                            logger.error(f"Error during YOLO detection for stream {stream_id}: {e}")
                            
            except Exception as e:
                logger.debug(f"Error processing packet for stream {stream_id}: {e}")
                continue
        
        logger.info(f"Finished processing stream {stream_id}. Frames: {frame_count}, Detections: {detection_count}, Skipped: {skipped_detections}, Chat Detections: {chat_detection_count}, Chat Skipped: {chat_skipped_detections}")
        
    except Exception as e:
        logger.error(f"Error processing stream {stream_id}: {e}")
        update_stream_status(stream_id, 'offline')
    finally:
        if container:
            try:
                container.close()
            except:
                pass
        logger.info(f"Cleanup completed for stream {stream_id}")

def get_online_streams():
    """Retrieve a list of online streams with their m3u8 URLs."""
    try:
        with get_session() as session:
            stmt = select(Stream).where(Stream.status == 'online')
            streams = session.execute(stmt).scalars().all()
            online_streams = []
            for stream in streams:
                m3u8_url = None
                if isinstance(stream, ChaturbateStream):
                    m3u8_url = stream.chaturbate_m3u8_url
                elif isinstance(stream, StripchatStream):
                    m3u8_url = stream.stripchat_m3u8_url
                
                if m3u8_url:
                    online_streams.append((stream.id, m3u8_url))
                else:
                    logger.warning(f"Stream {stream.id} has no m3u8 URL")
                    
            logger.info(f"Found {len(online_streams)} online streams")
            return online_streams
    except Exception as e:
        logger.error(f"Error retrieving online streams: {e}")
        return []

def main():
    """Main function to start processing streams and listen for new ones."""
    logger.info("Starting enhanced stream monitoring service with PyAV, Whisper, and Chat Monitoring...")
    
    if not pyav_available:
        logger.error("PyAV is required for this application. Please install with: pip install av")
        return
        
    if not yolo_available:
        logger.error("YOLO model is not available. Please install with: pip install ultralytics")
        return
        
    if not whisper_available:
        logger.error("Whisper is required for audio transcription. Please install with: pip install openai-whisper")
        return
    
    logger.info("All required dependencies available:")
    logger.info("✔ PyAV for stream processing")
    logger.info("✔ YOLO for object detection") 
    logger.info("✔ Whisper for audio transcription")
    logger.info("✔ Chat monitoring enabled")
    
    active_streams = []
    
    online_streams = get_online_streams()
    for stream_id, m3u8_url in online_streams:
        if len(active_streams) >= MAX_CONCURRENT_STREAMS:
            logger.warning(f"Max concurrent streams ({MAX_CONCURRENT_STREAMS}) reached. Skipping stream {stream_id}")
            break
            
        stop_event = threading.Event()
        thread = threading.Thread(target=process_stream, args=(stream_id, m3u8_url, stop_event), daemon=True)
        thread.start()
        active_streams.append((stream_id, thread, stop_event))
        logger.info(f"Started processing thread for stream {stream_id}")
    
    def listen_for_new_streams():
        known_ids = set(stream_id for stream_id, _ in online_streams)
        logger.info("Started listening for new streams...")
        
        while True:
            try:
                current_streams = get_online_streams()
                current_ids = set(stream_id for stream_id, _ in current_streams)
                new_streams = [s for s in current_streams if s[0] not in known_ids]
                
                for stream_id, m3u8_url in new_streams:
                    if len(active_streams) < MAX_CONCURRENT_STREAMS:
                        stop_event = threading.Event()
                        thread = threading.Thread(target=process_stream, args=(stream_id, m3u8_url, stop_event), daemon=True)
                        thread.start()
                        active_streams.append((stream_id, thread, stop_event))
                        known_ids.add(stream_id)
                        logger.info(f"Started processing new stream {stream_id}")
                    else:
                        logger.warning(f"Cannot start new stream {stream_id}: max concurrent limit reached")
                
                active_streams[:] = [(sid, t, e) for sid, t, e in active_streams if t.is_alive()]
                
            except Exception as e:
                logger.error(f"Error in new stream listener: {e}")
            
            time.sleep(NEW_STREAM_CHECK_INTERVAL)
    
    threading.Thread(target=listen_for_new_streams, daemon=True).start()
    
    logger.info("Enhanced stream monitoring service started successfully")
    logger.info(f"Detection settings - Cooldown: {DETECTION_COOLDOWN}s, Confidence threshold: {DETECTION_CONFIDENCE_THRESHOLD}, Max per minute: {MAX_DETECTIONS_PER_MINUTE}")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down stream monitoring service...")
        for stream_id, thread, stop_event in active_streams:
            stop_event.set()
        
        detection_managers.clear()
        audio_transcribers.clear()
        logger.info("Stream monitoring service stopped")

if __name__ == "__main__":
    main()