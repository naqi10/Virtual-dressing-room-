# Real-Time Virtual Try-On Setup Guide

## Overview
This integration adds a MediaPipe-based real-time virtual try-on feature to your Virtual Dressing Room application. Users can now sign in and choose between:
1. **Real-Time Try-On**: Live camera feed with MediaPipe pose detection
2. **Dataset Pairs**: Traditional dataset-based try-on

## Directory Structure

```
Virtual-dressing-room--main/
├── static/
│   └── shirts/          # Place your shirt images here (PNG/JPG with transparency)
├── realtime_tryon.py    # MediaPipe real-time pipeline module
├── new1.py              # Updated Flask app with sign-in and realtime routes
└── requirements.txt     # Dependencies (already includes mediapipe)
```

## Setup Instructions

### 1. Add Shirt Images
Place your shirt images (PNG or JPG) in the `static/shirts/` directory:
- PNG files with transparency work best
- Recommended size: 500x500 to 1000x1000 pixels
- Name them descriptively (e.g., `blue-casual-shirt.png`, `red-formal-shirt.jpg`)

Example:
```
static/shirts/
├── shirt1.png
├── shirt2.png
├── blue-casual.png
└── red-formal.jpg
```

### 2. Install Dependencies
Make sure you have all required packages:
```bash
pip install -r requirements.txt
```

Key dependencies:
- `mediapipe==0.10.14` (for pose detection)
- `opencv-python` (for camera and image processing)
- `flask` (for web server)
- `numpy` (for array operations)

### 3. Run the Application
```bash
python new1.py
```

The server will start on `http://localhost:5000`

## Usage Flow

1. **Sign In**: 
   - Navigate to `http://localhost:5000`
   - Enter any username and password (demo mode - accepts any credentials)
   - Click "Sign In"

2. **Choose Mode**:
   - **Real-Time Try-On**: Click "Real-Time Try-On" card
   - **Dataset Pairs**: Click "Dataset Pairs" card

3. **Real-Time Try-On Controls**:
   - Click "Start Camera" to begin
   - Use arrow keys or buttons to switch between shirts
   - Position yourself in front of the camera
   - The system will automatically detect your pose and overlay the shirt

4. **Dataset Pairs**:
   - Select a pair from the dropdown or enter an index
   - Click "Try On" to see the result

## Features

### Real-Time Try-On
- ✅ MediaPipe pose detection
- ✅ Automatic shirt sizing based on body proportions
- ✅ Smooth transitions between shirts
- ✅ Arm occlusion (shirt appears behind arms)
- ✅ Keyboard controls (arrow keys)
- ✅ Gallery view with thumbnails
- ✅ Hand gesture detection for selection

### Sign-In System
- ✅ Session-based authentication
- ✅ Dashboard with mode selection
- ✅ Logout functionality

## API Endpoints

### Authentication
- `GET /` - Redirects to sign-in or dashboard
- `GET /signin` - Sign-in page
- `POST /signin` - Authenticate user
- `GET /logout` - Logout user
- `GET /dashboard` - Dashboard with mode selection

### Real-Time Try-On
- `GET /realtime` - Real-time try-on page
- `GET /video_feed` - Camera video stream (MJPEG)
- `POST /realtime/next_shirt` - Switch to next shirt
- `POST /realtime/prev_shirt` - Switch to previous shirt
- `POST /realtime/set_shirt/<index>` - Set specific shirt
- `GET /realtime/info` - Get current shirt info (JSON)

### Dataset Pairs
- `GET /predict` - Dataset pairs page
- `POST /predict` - Process dataset pair

## Troubleshooting

### Camera Not Working
- Check camera permissions in your browser
- Ensure no other application is using the camera
- Try a different browser (Chrome/Firefox recommended)

### No Shirts Appearing
- Verify shirts are in `static/shirts/` directory
- Check file formats (PNG/JPG)
- Ensure files are readable

### Pose Detection Not Working
- Ensure good lighting
- Stand at appropriate distance from camera
- Make sure your upper body is visible

### Performance Issues
- Reduce `process_width` in `realtime_tryon.py` (default: 640)
- Adjust `smooth_frames` for smoother but slower processing
- Close other applications using CPU/GPU

## Customization

### Adjust Processing Settings
Edit `realtime_tryon.py`:
```python
RealtimeTryOn(
    shirt_folder="static/shirts",
    process_width=640,      # Lower = faster, less accurate
    smooth_frames=15        # More = smoother, more lag
)
```

### Change Shirt Sizing
Modify the body proportion factors in `process_frame()` method:
```python
if body_ratio > 1.8:
    width_factor = 1.6
    height_factor = 1.8
```

### Add More Shirts
Simply add more PNG/JPG files to `static/shirts/` - they will be automatically loaded!

## Notes

- The sign-in system is currently in demo mode (accepts any credentials)
- For production, implement proper user authentication
- Camera access requires HTTPS in production (or localhost for development)
- MediaPipe works best with good lighting and clear view of upper body

