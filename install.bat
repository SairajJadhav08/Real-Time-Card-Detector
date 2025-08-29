@echo off
echo ========================================
echo Playing Card Detector - Installation Script
echo ========================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

:: Check if Node.js is installed
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js 16+ from https://nodejs.org/
    pause
    exit /b 1
)

echo ✓ Python and Node.js are installed
echo.

:: Create project directories
echo Creating project directories...
if not exist "models" mkdir models
if not exist "templates" mkdir templates
if not exist "exports" mkdir exports
if not exist "uploads" mkdir uploads
echo ✓ Directories created
echo.

:: Backend setup
echo ========================================
echo Setting up Backend (Python + FastAPI)
echo ========================================
cd backend

:: Create virtual environment
echo Creating Python virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)
echo ✓ Virtual environment created

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo ✓ Virtual environment activated

:: Install Python dependencies
echo Installing Python dependencies...
pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install Python dependencies
    echo Please check your internet connection and try again
    pause
    exit /b 1
)
echo ✓ Python dependencies installed

:: Create .env file if it doesn't exist
if not exist ".env" (
    echo Creating .env configuration file...
    (
        echo # Environment
        echo ENVIRONMENT=development
        echo DEBUG=true
        echo.
        echo # Database
        echo DATABASE_URL=sqlite:///./card_detector.db
        echo.
        echo # API
        echo CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
        echo.
        echo # ML Model Settings
        echo CONFIDENCE_THRESHOLD=0.7
        echo NMS_THRESHOLD=0.4
        echo MAX_DETECTIONS=10
        echo.
        echo # Logging
        echo LOG_LEVEL=INFO
    ) > .env
    echo ✓ .env file created
)

cd ..

:: Frontend setup
echo.
echo ========================================
echo Setting up Frontend (React + TypeScript)
echo ========================================
cd frontend

:: Install Node.js dependencies
echo Installing Node.js dependencies...
npm install
if %errorlevel% neq 0 (
    echo ERROR: Failed to install Node.js dependencies
    echo Please check your internet connection and try again
    pause
    exit /b 1
)
echo ✓ Node.js dependencies installed

cd ..

:: Create startup scripts
echo.
echo Creating startup scripts...

:: Backend startup script
(
    echo @echo off
    echo echo Starting Playing Card Detector Backend...
    echo cd backend
    echo call venv\Scripts\activate.bat
    echo python main.py
    echo pause
) > start-backend.bat

:: Frontend startup script
(
    echo @echo off
    echo echo Starting Playing Card Detector Frontend...
    echo cd frontend
    echo npm start
    echo pause
) > start-frontend.bat

:: Combined startup script
(
    echo @echo off
    echo echo ========================================
    echo echo Playing Card Detector - Starting Services
    echo echo ========================================
    echo echo.
    echo echo Starting Backend Server...
    echo start "Backend" cmd /k "cd backend ^&^& call venv\Scripts\activate.bat ^&^& python main.py"
    echo echo.
    echo echo Waiting for backend to start...
    echo timeout /t 5 /nobreak ^> nul
    echo echo.
    echo echo Starting Frontend Server...
    echo start "Frontend" cmd /k "cd frontend ^&^& npm start"
    echo echo.
    echo echo ========================================
    echo echo Services are starting...
    echo echo Backend: http://localhost:8000
    echo echo Frontend: http://localhost:3000
    echo echo API Docs: http://localhost:8000/docs
    echo echo ========================================
    echo echo.
    echo echo Press any key to exit...
    echo pause ^> nul
) > start-all.bat

echo ✓ Startup scripts created
echo.

:: Installation complete
echo ========================================
echo Installation Complete! 🎉
echo ========================================
echo.
echo Next steps:
echo 1. Run 'start-all.bat' to start both servers
echo 2. Open http://localhost:3000 in your browser
echo 3. Allow camera access when prompted
echo 4. Point camera at playing cards to test detection
echo.
echo Available scripts:
echo - start-all.bat     : Start both backend and frontend
echo - start-backend.bat : Start only the backend server
echo - start-frontend.bat: Start only the frontend server
echo.
echo Useful URLs:
echo - Frontend App: http://localhost:3000
echo - Backend API: http://localhost:8000
echo - API Documentation: http://localhost:8000/docs
echo - API ReDoc: http://localhost:8000/redoc
echo.
echo For troubleshooting, check README.md
echo.
echo Would you like to start the application now? (y/n)
set /p choice="Enter your choice: "
if /i "%choice%"=="y" (
    echo.
    echo Starting application...
    call start-all.bat
) else (
    echo.
    echo You can start the application later by running 'start-all.bat'
)

echo.
echo Installation script completed!
pause