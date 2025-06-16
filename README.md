# StreamGuard AI - Live Stream Detection Module

🛡️ **Advanced AI-powered detection system for real-time monitoring of live streams**

This repository contains the core detection engine for StreamGuard AI, a comprehensive live stream monitoring solution that uses computer vision, audio transcription, and chat analysis to detect flagged content across multiple streaming platforms.

## 🌟 Features

### 🔍 Multi-Modal Detection
- **Computer Vision**: Real-time object detection using YOLOv8
- **Audio Analysis**: Speech-to-text transcription with keyword detection
- **Chat Monitoring**: Real-time chat message analysis for flagged keywords

### 🎯 Platform Support
- **Chaturbate**: Full video, audio, and chat monitoring
- **Stripchat**: Complete platform integration
- Extensible architecture for additional platforms

### ⚡ Performance & Reliability
- **Concurrent Processing**: Monitor up to 4 streams simultaneously
- **Smart Rate Limiting**: Prevents detection spam with configurable cooldowns
- **Resource Optimization**: Efficient frame processing and memory management
- **Automatic Reconnection**: Robust stream handling with failure recovery

### 🧠 Intelligent Detection Management
- **Detection Zones**: Area-based duplicate detection prevention
- **Confidence Thresholds**: Customizable per-object detection sensitivity
- **Temporal Filtering**: Time-based detection cooldowns
- **Statistical Tracking**: Comprehensive detection analytics

## 🛠️ Technology Stack

- **Computer Vision**: YOLOv8 (Ultralytics)
- **Audio Processing**: OpenAI Whisper + PyAV
- **Stream Processing**: PyAV for HLS/m3u8 streams
- **Database**: SQLAlchemy with PostgreSQL/MySQL support
- **Concurrency**: Threading with event-based coordination
- **Logging**: Comprehensive structured logging

## 📋 Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for faster inference)
- Database server (PostgreSQL/MySQL)
- Sufficient storage for model files (~100MB)

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/kashimkyari/streamGuard-AI-Live-Stream-Detection-Module.git
cd streamGuard-AI-Live-Stream-Detection-Module
```

### 2. Install Dependencies
```bash
# Install core dependencies
pip install -r requirements.txt

# Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Or install PyTorch (GPU version) - for better performance
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install Required Models
```bash
# YOLO model will be downloaded automatically on first run
# Whisper models will be downloaded automatically

# Alternatively, pre-download models:
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
python -c "import whisper; whisper.load_model('base')"
```

### 4. Environment Configuration
Create a `.env` file in the project root:

```env
# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/streamguard

# Detection Settings
CHECK_INTERVAL=10
DETECTION_COOLDOWN=300
DETECTION_CONFIDENCE_THRESHOLD=0.5
MAX_DETECTIONS_PER_MINUTE=1

# Stream Processing
MAX_CONCURRENT_STREAMS=4
AUDIO_SEGMENT_DURATION=30
TRANSCRIPTION_INTERVAL=3

# Chat Monitoring
CHAT_CHECK_INTERVAL=10
NEW_STREAM_CHECK_INTERVAL=30
```

## 🎛️ Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHECK_INTERVAL` | 10 | Stream status check interval (seconds) |
| `DETECTION_COOLDOWN` | 300 | Minimum time between same object detections (seconds) |
| `DETECTION_CONFIDENCE_THRESHOLD` | 0.5 | Minimum confidence for object detection (0.0-1.0) |
| `MAX_DETECTIONS_PER_MINUTE` | 1 | Maximum detections per object per minute |
| `MAX_CONCURRENT_STREAMS` | 4 | Maximum simultaneous streams to process |
| `AUDIO_SEGMENT_DURATION` | 30 | Audio capture duration for transcription (seconds) |
| `TRANSCRIPTION_INTERVAL` | 3 | Audio transcription frequency (seconds) |
| `CHAT_CHECK_INTERVAL` | 10 | Chat message checking frequency (seconds) |

## 🏃‍♂️ Usage

### Basic Execution
```bash
python main.py
```

### Development Mode with Verbose Logging
```bash
export LOG_LEVEL=DEBUG
python main.py
```

### Docker Deployment
```bash
# Build the container
docker build -t streamguard-detection .

# Run with environment file
docker run --env-file .env streamguard-detection
```

## 🗃️ Database Schema

The detection module requires the following database tables:

- **streams**: Main stream records
- **chaturbate_streams**: Chaturbate-specific stream data
- **stripchat_streams**: Stripchat-specific stream data
- **flagged_objects**: Object detection configuration
- **chat_keywords**: Chat keyword configuration
- **detection_logs**: Detection event records
- **assignments**: Stream-to-agent assignments

> **Note**: The complete database schema and management is handled by the [main application repository](https://github.com/kashimkyari/LiveStream_Monitoring_Vue3_Flask.git).

## 🔧 API Integration

This detection module is designed to work with the main StreamGuard application:

- **Stream Management**: Receives stream URLs from the main application
- **Detection Logging**: Stores detection events in the shared database
- **Real-time Updates**: Provides detection status updates
- **Agent Assignment**: Respects stream-to-agent assignments

## 📊 Monitoring & Logging

### Log Files
- `monitor.log`: Comprehensive application logs
- Console output: Real-time status updates

### Detection Statistics
- Per-stream detection counts
- Confidence score distributions
- Processing performance metrics
- Error rates and recovery statistics

### Health Monitoring
```python
# Check detection manager statistics
stats = detection_manager.get_detection_stats()
print(f"Detection stats: {stats}")
```

## 🎯 Detection Types

### 1. Object Detection
```python
# Example flagged objects configuration
flagged_objects = {
    'person': 0.7,      # 70% confidence threshold
    'weapon': 0.5,      # 50% confidence threshold
    'explicit': 0.8     # 80% confidence threshold
}
```

### 2. Audio Transcription
- Real-time speech-to-text conversion
- Keyword matching with regex patterns
- Multi-language support (auto-detection)

### 3. Chat Analysis
- Platform-specific chat API integration
- Real-time message processing
- Keyword pattern matching

## 🔒 Security Considerations

- **No Data Storage**: Audio and video frames are processed in memory only
- **Secure Connections**: All stream connections use HTTPS/WSS
- **Database Security**: Uses parameterized queries to prevent SQL injection
- **Rate Limiting**: Built-in protection against API abuse

## 🚨 Error Handling

The system includes comprehensive error handling:

- **Stream Failures**: Automatic reconnection with exponential backoff
- **Model Loading**: Graceful degradation if AI models fail to load
- **Database Errors**: Transaction rollback and retry mechanisms
- **Network Issues**: Timeout handling and connection recovery

## 📈 Performance Optimization

### CPU Optimization
- Frame skipping for non-critical processing
- Efficient memory management
- Optimized detection pipelines

### GPU Acceleration
```bash
# Enable GPU acceleration (if available)
export CUDA_VISIBLE_DEVICES=0
python main.py
```

## 🧪 Testing

### Unit Tests
```bash
python -m pytest tests/
```

### Integration Tests
```bash
python -m pytest tests/integration/
```

### Performance Benchmarks
```bash
python benchmark.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Projects

- **[Main Application](https://github.com/kashimkyari/LiveStream_Monitoring_Vue3_Flask.git)**: Complete StreamGuard platform with web interface
- **Frontend**: Vue.js 3 dashboard for stream management
- **Backend**: Flask API for stream scraping and application management

## 📞 Support

For support and questions:

- **Issues**: [GitHub Issues](https://github.com/kashimkyari/streamGuard-AI-Live-Stream-Detection-Module/issues)
- **Documentation**: [Wiki](https://github.com/kashimkyari/streamGuard-AI-Live-Stream-Detection-Module/wiki)
- **Main Project**: [StreamGuard Platform](https://github.com/kashimkyari/LiveStream_Monitoring_Vue3_Flask.git)

## 🏆 Acknowledgments

- **Ultralytics**: YOLOv8 object detection model
- **OpenAI**: Whisper speech recognition model
- **PyAV**: Video/audio processing library
- **SQLAlchemy**: Database ORM and management

---

**StreamGuard AI** - Protecting digital spaces through intelligent monitoring