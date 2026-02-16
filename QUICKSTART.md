# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Choose Your Version

#### Option A: Real-Time Detection (Recommended)
```bash
jupyter notebook "Digital_Image__Real_Time_.ipynb"
```
- Use your webcam
- Get instant alerts
- See live EAR/MAR values

#### Option B: Static Image Analysis
```bash
jupyter notebook "digital-image-drowsy.ipynb"
```
- Analyze pre-recorded images
- Calculate adaptive thresholds
- Visualize detection results

### Step 3: Run and Test
- **Real-time**: Look at the camera and try closing your eyes or yawning
- **Static**: Ensure you have images in the correct directory structure
- Press 'q' to quit real-time detection

## 📊 Expected Output

### Real-Time Mode:
```
✅ Video window with:
   - Face bounding box (green)
   - EAR value (top-left)
   - MAR value (top-left)
   - "DROWSY!" warning when detected (red)
   - Audio beep alert

✅ Console output:
   EAR: 0.28 | MAR: 0.45 | Status: Alert
   EAR: 0.15 | MAR: 0.45 | Status: DROWSY!
```

### Static Image Mode:
```
✅ Processed images with:
   - Bounding boxes around faces
   - Classification labels (Alert/Drowsy)
   - EAR and MAR values displayed

✅ Console statistics:
   Person 1 - Avg EAR: 0.28, Avg MAR: 0.42
   Person 2 - Avg EAR: 0.26, Avg MAR: 0.38
```

## ⚙️ Customization

### Adjust Sensitivity
Edit these values in the notebook:

```python
# Make detection more sensitive (detect drowsiness earlier)
EAR_THRESHOLD = 0.28  # Default: 0.25
MAR_THRESHOLD = 0.55  # Default: 0.6
CONSECUTIVE_FRAMES = 2  # Default: 3

# Make detection less sensitive (reduce false positives)
EAR_THRESHOLD = 0.22
MAR_THRESHOLD = 0.65
CONSECUTIVE_FRAMES = 5
```

### Change Alert Sound (Windows)
```python
# Higher pitch, longer duration
winsound.Beep(1500, 500)  # (frequency_Hz, duration_ms)

# Lower pitch, shorter duration
winsound.Beep(800, 200)
```

## 🐛 Troubleshooting

### Camera Not Found
```python
# Try different camera index
cap = cv2.VideoCapture(1)  # Instead of 0
```

### MediaPipe Not Detecting Face
- Ensure good lighting
- Face the camera directly
- Check if image enhancement is helping:
```python
# Increase gamma for darker images
enhanced = enhance_image(frame, gamma=1.5)
```

### Slow Performance
```python
# Reduce video resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

## 💡 Tips for Best Results

1. **Lighting**: Ensure face is well-lit from the front
2. **Position**: Keep face centered in frame
3. **Distance**: Stay 50-70cm from camera
4. **Calibration**: Run for 10-15 seconds to establish baseline
5. **Testing**: Try different expressions to test sensitivity

## 📝 Notes

- First run may take longer (MediaPipe model download)
- Webcam permission required for real-time mode
- Works best with modern webcams (720p or higher)
- Processing speed: 20-30 FPS on average hardware

## 🆘 Need Help?

- Check the main [README.md](README.md) for detailed documentation
- Review the presentation [Drowsy_Driver_detection.pdf](Drowsy_Driver_detection.pdf)
- Open an issue on GitHub for bugs or questions

---
Happy Testing! 🚗💤🎯
