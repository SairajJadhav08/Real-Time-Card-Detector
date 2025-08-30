# 🃏 Playing Card Detector - Real-Time AI Recognition

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18+-blue.svg)](https://reactjs.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://ultralytics.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A sophisticated real-time playing card detection system powered by **YOLOv8 transfer learning** and modern web technologies. Achieve **85-95% accuracy** with custom-trained models!

## 🌟 Features

### 🎯 **Advanced ML Detection**
- **YOLOv8 Transfer Learning** - State-of-the-art object detection
- **Custom Training Pipeline** - Train on your own card datasets
- **Real-time Performance** - Fast inference for live video streams
- **High Accuracy** - 85-95% detection accuracy with proper training
- **Robust Recognition** - Works under various lighting and angle conditions

### 🚀 **Complete Training System**
- **Interactive Data Collection** - Camera-based image capture with labeling
- **Advanced Data Augmentation** - 10+ augmentation techniques
- **Automated Training Pipeline** - One-click model improvement
- **Performance Monitoring** - Real-time training metrics and visualization
- **Model Export** - Deploy optimized models for production

### 💻 **Modern Web Interface**
- **React Frontend** - Responsive and intuitive user interface
- **FastAPI Backend** - High-performance Python API
- **Real-time WebSocket** - Live detection streaming
- **Performance Dashboard** - Model statistics and analytics
- **Settings Panel** - Easy configuration management

### 🔧 **Developer-Friendly**
- **Comprehensive Documentation** - Step-by-step guides
- **Modular Architecture** - Easy to extend and customize
- **Docker Support** - Containerized deployment
- **API Documentation** - Auto-generated OpenAPI specs

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- Webcam (for real-time detection)

### 1. Clone Repository
```bash
git clone https://github.com/SairajJadhav08/Real-Time-Card-Detector.git
cd Real-Time-Card-Detector
```

### 2. Backend Setup
```bash
cd backend
pip install -r requirements.txt
python main.py
```

### 3. Frontend Setup
```bash
cd frontend
npm install
npm start
```

### 4. Access Application
- **Frontend**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs

## 🎯 Improve Your Model (NEW!)

Our advanced ML training pipeline helps you achieve **professional-grade accuracy**:

### Quick Model Training
```bash
cd backend/app/ml
python training_pipeline.py
```

This interactive pipeline will guide you through:
1. **Data Collection** - Capture diverse card images
2. **Data Augmentation** - Expand dataset 5-10x automatically
3. **Transfer Learning** - Train YOLOv8 on your data
4. **Model Evaluation** - Performance metrics and visualization

### Expected Results
- **Before**: ~60-70% accuracy (OpenCV template matching)
- **After**: 85-95% accuracy (YOLOv8 transfer learning)
- **Training Time**: 1-3 hours
- **Data Needed**: 500-1000 card images

## 📁 Project Structure

```
Real-Time-Card-Detector/
├── backend/
│   ├── app/
│   │   ├── ml/                    # 🆕 ML Training Pipeline
│   │   │   ├── training_pipeline.py    # Complete training workflow
│   │   │   ├── data_collector.py       # Interactive data collection
│   │   │   ├── data_augmentation.py    # Dataset augmentation
│   │   │   ├── transfer_learning.py    # YOLOv8 implementation
│   │   │   ├── test_pipeline.py        # System verification
│   │   │   └── README.md              # Detailed ML guide
│   │   ├── core/                  # Core backend logic
│   │   ├── models/                # Database models
│   │   └── main.py               # FastAPI application
│   ├── requirements.txt          # Python dependencies
│   └── main.py                   # Application entry point
├── frontend/
│   ├── src/
│   │   ├── components/           # React components
│   │   ├── hooks/               # Custom React hooks
│   │   └── App.tsx              # Main application
│   ├── package.json             # Node.js dependencies
│   └── public/                  # Static assets
└── README.md                    # This file
```

## 🎮 Usage Guide

### Basic Detection
1. **Start the application** (see Quick Start)
2. **Allow camera access** when prompted
3. **Position cards** in the camera view
4. **View real-time detection** results

### Advanced Training
1. **Navigate to ML directory**: `cd backend/app/ml`
2. **Read the guide**: Check `README.md` for detailed instructions
3. **Run training pipeline**: `python training_pipeline.py`
4. **Follow interactive prompts** for data collection and training
5. **Deploy improved model** in your application

### API Usage
```python
import requests

# Upload image for detection
with open('card_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/detect',
        files={'file': f}
    )
    
detections = response.json()
print(f"Detected cards: {detections['cards']}")
```

## 🔧 Configuration

### Model Settings
- **Model Type**: Choose between OpenCV and YOLOv8
- **Confidence Threshold**: Adjust detection sensitivity
- **Input Resolution**: Balance speed vs accuracy
- **Batch Processing**: Configure for multiple images

### Training Parameters
- **Epochs**: Training duration (default: 100)
- **Batch Size**: Memory vs speed tradeoff
- **Learning Rate**: Model optimization speed
- **Augmentation Level**: Dataset expansion factor

## 📊 Performance Metrics

### Detection Accuracy
| Model Type | Accuracy | Speed (FPS) | Memory Usage |
|------------|----------|-------------|---------------|
| OpenCV Template | 60-70% | 30+ | Low |
| YOLOv8 Nano | 80-85% | 25+ | Medium |
| YOLOv8 Small | 85-90% | 20+ | Medium |
| YOLOv8 Medium | 90-95% | 15+ | High |

### Supported Cards
- **Standard 52-card deck**
- **All ranks**: A, 2-10, J, Q, K
- **All suits**: ♠️ ♥️ ♦️ ♣️
- **Multiple cards** in single image
- **Overlapping cards** (with training)

## 🛠️ Development

### Adding New Features
1. **Backend**: Extend FastAPI endpoints in `backend/app/`
2. **Frontend**: Add React components in `frontend/src/components/`
3. **ML Models**: Customize training in `backend/app/ml/`

### Testing
```bash
# Test ML pipeline
cd backend/app/ml
python test_pipeline.py

# Test API endpoints
cd backend
pytest tests/

# Test frontend
cd frontend
npm test
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open Pull Request**

### Development Guidelines
- Follow **PEP 8** for Python code
- Use **TypeScript** for frontend development
- Add **tests** for new features
- Update **documentation** for API changes

## 📚 Documentation

- **ML Training Guide**: `backend/app/ml/README.md`
- **API Documentation**: http://localhost:8000/docs (when running)
- **Frontend Components**: Check component files for JSDoc
- **Database Schema**: `backend/app/models/`

## 🐛 Troubleshooting

### Common Issues

**"No camera detected"**
- Ensure webcam is connected and not used by other applications
- Check browser permissions for camera access

**"Low detection accuracy"**
- Use the ML training pipeline to improve the model
- Ensure good lighting and clear card visibility
- Consider collecting more training data

**"Slow performance"**
- Reduce input resolution in settings
- Use smaller YOLOv8 model (nano/small)
- Close other resource-intensive applications

**"Installation errors"**
- Ensure Python 3.8+ and Node.js 16+ are installed
- Try creating a virtual environment
- Check system compatibility for PyTorch/CUDA

## 🔮 Future Enhancements

- **Multi-deck Support** - Handle multiple card decks
- **Card Counting** - Advanced game analysis
- **Mobile App** - React Native implementation
- **Cloud Deployment** - AWS/Azure integration
- **Real-time Multiplayer** - WebSocket-based gaming
- **Advanced Analytics** - Game statistics and insights

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ultralytics** for the amazing YOLOv8 framework
- **FastAPI** team for the excellent web framework
- **React** community for frontend inspiration
- **OpenCV** for computer vision foundations
- **Contributors** who help improve this project

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/SairajJadhav08/Real-Time-Card-Detector/issues)
- **Discussions**: [GitHub Discussions](https://github.com/SairajJadhav08/Real-Time-Card-Detector/discussions)
- **Email**: [Contact Developer](mailto:your-email@example.com)

---

⭐ **Star this repository** if you find it helpful!

🚀 **Ready to build amazing card detection applications?** Start with our [Quick Start Guide](#-quick-start)!
