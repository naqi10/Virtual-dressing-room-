# Directory Structure Guide

## Current Structure

```
Virtual-dressing-room--main/
├── static/
│   ├── shirts/              # ← Place your shirt images here
│   ├── uploads/            # User uploaded images
│   ├── results/            # Generated try-on results
│   └── images/             # Dataset images
├── checkpoints/            # Model checkpoints
├── datasets/               # Dataset files
├── realtime_tryon.py       # MediaPipe real-time pipeline
├── new1.py                 # Main Flask application
└── requirements.txt        # Python dependencies
```

## Shirt Images Location

**Your shirt images should be placed in:**
```
static/shirts/
```

### Supported Formats
- PNG (recommended - supports transparency)
- JPG/JPEG

### Example
If you have shirts in `D:\Virtual-Shirt-Try-On-main\Shirts\`, you can:

**Option 1: Copy shirts to the new location**
```bash
# Copy all shirt images
copy "D:\Virtual-Shirt-Try-On-main\Shirts\*.png" "static\shirts\"
copy "D:\Virtual-Shirt-Try-On-main\Shirts\*.jpg" "static\shirts\"
```

**Option 2: Update the code to use your existing directory**

Edit `realtime_tryon.py` line ~20:
```python
# Change from:
shirt_folder = os.path.join("static", "shirts")

# To:
shirt_folder = r"D:\Virtual-Shirt-Try-On-main\Shirts"
```

Or edit `new1.py` in the `get_realtime_system()` function:
```python
shirt_folder = r"D:\Virtual-Shirt-Try-On-main\Shirts"  # Use your existing path
```

## Quick Setup

1. **Create shirts directory** (if not exists):
   ```bash
   mkdir static\shirts
   ```

2. **Add your shirt images** to `static/shirts/`

3. **Run the application**:
   ```bash
   python new1.py
   ```

4. **Access the app**:
   - Open browser: `http://localhost:5000`
   - Sign in with any username/password
   - Choose "Real-Time Try-On" or "Dataset Pairs"

## File Naming

Shirt files can have any name, but descriptive names help:
- ✅ `blue-casual-shirt.png`
- ✅ `red-formal-shirt.jpg`
- ✅ `shirt1.png`
- ✅ `tshirt_001.png`

The system will automatically:
- Load all PNG/JPG files from the shirts folder
- Display them in the gallery
- Allow switching between them

