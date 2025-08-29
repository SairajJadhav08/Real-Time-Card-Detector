# Playing Card Detector - Real-Time AI Recognition

A full-stack application that detects playing cards in real-time using computer vision, featuring an interactive React frontend and Python backend with AI/ML capabilities.

## 🎯 Features

- **Real-time card detection** using OpenCV and machine learning
- **Interactive React UI** with animations (card flipping, glowing borders)
- **Live camera feed** with instant card recognition
- **Database storage** for detection history
- **Fast API backend** optimized for real-time performance
- **Beautiful animations** using Framer Motion and Tailwind CSS

## 🏗️ Project Structure

```
playing-card-detector/
├── backend/                 # Python FastAPI backend
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py         # FastAPI application
│   │   ├── models/         # Database models
│   │   ├── api/            # API routes
│   │   ├── core/           # Core configurations
│   │   └── ml/             # Machine learning modules
│   ├── requirements.txt    # Python dependencies
│   └── Dockerfile         # Docker configuration
├── frontend/               # React frontend
│   ├── public/
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── hooks/          # Custom hooks
│   │   ├── utils/          # Utility functions
│   │   └── styles/         # CSS and styling
│   ├── package.json       # Node.js dependencies
│   └── tailwind.config.js # Tailwind configuration
├── models/                 # Pre-trained models and datasets
├── database/              # Database files and migrations
├── docs/                  # Documentation
└── scripts/               # Setup and utility scripts
```

## 🚀 Quick Start

### Prerequisites

- **Node.js** (v16 or higher)
- **Python** (3.8 or higher)
- **Git**
- **VS Code** (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd playing-card-detector
   ```

2. **Backend Setup**
   ```bash
   cd backend
   python -m venv venv
   venv\Scripts\activate  # On Windows
   pip install -r requirements.txt
   ```

3. **Frontend Setup**
   ```bash
   cd frontend
   npm install
   ```

4. **Database Setup**
   ```bash
   # SQLite database will be created automatically
   python backend/app/database/init_db.py
   ```

### Running the Application

1. **Start Backend Server**
   ```bash
   cd backend
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Start Frontend Development Server**
   ```bash
   cd frontend
   npm start
   ```

3. **Open in Browser**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

## 📊 Dataset

This project uses the **Complete Playing Card Dataset** from Kaggle:
- **Source**: https://www.kaggle.com/datasets/jaypradipshah/the-complete-playing-card-dataset
- **Contains**: 53 classes (52 cards + 1 joker)
- **Format**: High-quality images for training and testing

## 🔧 Technology Stack

### Frontend
- **React 18** - Modern UI framework
- **Tailwind CSS** - Utility-first styling
- **Framer Motion** - Smooth animations
- **Axios** - HTTP client for API calls
- **React Webcam** - Camera integration

### Backend
- **FastAPI** - High-performance Python web framework
- **OpenCV** - Computer vision library
- **TensorFlow/PyTorch** - Machine learning framework
- **SQLAlchemy** - Database ORM
- **Pydantic** - Data validation

### Database
- **SQLite** - Lightweight database for development
- **PostgreSQL** - Production database option

## 🎮 Usage

1. **Allow camera access** when prompted
2. **Hold a playing card** in front of the camera
3. **Watch real-time detection** with animated feedback
4. **View detection history** in the sidebar
5. **Enjoy smooth animations** and responsive UI

## 🔍 API Endpoints

- `POST /api/detect` - Detect cards in uploaded image
- `GET /api/history` - Get detection history
- `GET /api/stats` - Get detection statistics
- `WebSocket /ws/detect` - Real-time detection stream

## 🎨 UI Features

- **Live camera feed** with overlay detection boxes
- **Card flip animations** when new cards are detected
- **Glowing borders** around detected cards
- **Smooth transitions** between different states
- **Responsive design** for all screen sizes
- **Dark/Light theme** support

## 📈 Performance Optimization

- **Frame rate optimization** for smooth real-time detection
- **Model quantization** for faster inference
- **WebSocket connections** for low-latency communication
- **Efficient image processing** pipeline
- **Caching strategies** for repeated detections

## 🛠️ Development

### VS Code Setup

1. **Install recommended extensions**:
   - Python
   - ES7+ React/Redux/React-Native snippets
   - Tailwind CSS IntelliSense
   - Prettier

2. **Configure workspace settings**:
   ```json
   {
     "python.defaultInterpreterPath": "./backend/venv/Scripts/python.exe",
     "editor.formatOnSave": true,
     "editor.codeActionsOnSave": {
       "source.organizeImports": true
     }
   }
   ```

### Testing

```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm test
```

## 📝 License

MIT License - see LICENSE file for details

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📞 Support

For questions and support, please open an issue in the GitHub repository.