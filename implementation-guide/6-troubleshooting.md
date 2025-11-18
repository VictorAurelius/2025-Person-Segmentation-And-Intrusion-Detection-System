# Hướng Dẫn Xử Lý Sự Cố

## 1. Vấn Đề Cài Đặt

### Lỗi: `pip install` thất bại

**Triệu chứng:**
```
ERROR: Could not find a version that satisfies the requirement opencv-python
```

**Nguyên nhân:**
- pip quá cũ
- Python version không tương thích
- Kết nối mạng

**Giải pháp:**

```bash
# 1. Cập nhật pip
python -m pip install --upgrade pip

# 2. Kiểm tra Python version (cần >= 3.8)
python --version

# 3. Thử cài riêng lẻ
pip install opencv-python==4.8.0

# 4. Dùng headless version
pip install opencv-python-headless

# 5. Cài từ wheel file
pip download opencv-python
pip install opencv_python-4.8.0-*.whl
```

---

### Lỗi: Import errors

**Triệu chứng:**
```python
ModuleNotFoundError: No module named 'cv2'
```

**Giải pháp:**

```bash
# 1. Đảm bảo virtual environment đã kích hoạt
which python  # Phải trỏ đến venv/bin/python

# 2. Kiểm tra opencv đã cài chưa
pip list | grep opencv

# 3. Cài lại opencv
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python

# 4. Test import
python -c "import cv2; print('OK')"
```

---

### Lỗi: Thiếu build tools (Windows)

**Triệu chứng:**
```
error: Microsoft Visual C++ 14.0 is required
```

**Giải pháp:**

1. Tải và cài Visual C++ Build Tools:
   https://visualstudio.microsoft.com/visual-cpp-build-tools/

2. Hoặc cài Visual Studio Community

3. Hoặc dùng pre-built wheels

---

## 2. Vấn Đề Runtime

### Lỗi: Video không mở được

**Triệu chứng:**
```
[ERROR] Cannot open video source
```

**Các bước kiểm tra:**

```bash
# 1. Kiểm tra file có tồn tại không
ls -lh data/input/test_video.mp4

# 2. Kiểm tra loại file
file data/input/test_video.mp4

# 3. Test phát video trong VLC
vlc data/input/test_video.mp4

# 4. Kiểm tra với Python
python -c "import cv2; cap = cv2.VideoCapture('data/input/test_video.mp4'); print('Opened' if cap.isOpened() else 'Failed')"
```

**Giải pháp:**

```bash
# Nếu format không support, convert
ffmpeg -i input.avi -c:v libx264 output.mp4

# Hoặc dùng opencv-contrib-python
pip install opencv-contrib-python
```

---

### Lỗi: Webcam không mở được

**Triệu chứng:**
```
[ERROR] Cannot open camera 0
```

**Giải pháp:**

```bash
# 1. Liệt kê cameras có sẵn (Linux)
ls /dev/video*

# 2. Thử index khác
python src/main.py --source 1
python src/main.py --source 2

# 3. Kiểm tra permissions (Linux/Mac)
# Linux
sudo usermod -a -G video $USER
# Logout và login lại

# Mac: System Preferences → Security & Privacy → Camera
# Cho phép Terminal/Python

# 4. Kiểm tra camera bị app khác dùng
# Đóng Zoom, Skype, etc.
```

---

### Lỗi: ROI file not found

**Triệu chứng:**
```
FileNotFoundError: data/roi/restricted_area.json
```

**Giải pháp:**

```bash
# 1. Tạo thư mục
mkdir -p data/roi

# 2. Tạo file ROI tối thiểu
cat > data/roi/restricted_area.json << 'EOF'
{
  "restricted_areas": [
    {
      "name": "Test Area",
      "type": "rectangle",
      "x": 100,
      "y": 100,
      "width": 200,
      "height": 200,
      "color": [255, 0, 0]
    }
  ]
}
EOF

# 3. Hoặc dùng ROI selector tool
python tools/roi_selector.py --video data/input/video.mp4
```

---

## 3. Vấn Đề Phát Hiện

### Vấn Đề: Không phát hiện được chuyển động

**Triệu chứng:**
- Video chạy nhưng không có bounding boxes
- Không có alerts dù rõ ràng có người

**Nguyên nhân:**
- Threshold quá cao
- Min object area quá lớn
- Background model chưa học xong

**Giải pháp:**

```yaml
# Trong config.yaml, giảm các giá trị này:

motion:
  threshold: 10  # Giảm từ 16

intrusion:
  min_object_area: 500  # Giảm từ 1000
  overlap_threshold: 0.2  # Giảm từ 0.3
```

**Debug script:**

```python
import cv2
import numpy as np

cap = cv2.VideoCapture("data/input/test_video.mp4")
bg_subtractor = cv2.createBackgroundSubtractorMOG2(threshold=10)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    fg_mask = bg_subtractor.apply(frame)

    # Hiển thị mask để debug
    cv2.imshow("Original", frame)
    cv2.imshow("Foreground Mask", fg_mask)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

### Vấn Đề: Quá nhiều false alerts

**Triệu chứng:**
- Alerts liên tục
- Shadows trigger alerts
- Nhiễu nhỏ cũng alert

**Giải pháp:**

```yaml
# 1. Tăng thresholds
motion:
  threshold: 25  # Tăng từ 16
  detect_shadows: true  # Bật shadow detection

intrusion:
  overlap_threshold: 0.5  # Tăng từ 0.3
  time_threshold: 2.0  # Tăng từ 1.0
  min_object_area: 2000  # Tăng từ 1000
```

**Morphological filtering mạnh hơn:**

```yaml
morphology:
  kernel_size: 7  # Tăng từ 5
  iterations: 3   # Tăng từ 2
```

---

### Vấn Đề: Edges không detect được

**Triệu chứng:**
- Object boundaries không rõ
- Edge map trống hoặc quá nhiễu

**Giải pháp:**

```yaml
# Điều chỉnh Canny thresholds
edge:
  method: "canny"
  low_threshold: 30   # Giảm để có nhiều edges hơn
  high_threshold: 100  # Giảm tương ứng (ratio 1:3)
```

**Hoặc thử Sobel:**

```yaml
edge:
  method: "sobel"
  low_threshold: 50
  high_threshold: 150
```

---

### Vấn Đề: Segmentation errors

**Triệu chứng:**
- Người bị tách thành nhiều vùng
- Background lẫn vào vùng người

**Giải pháp:**

```yaml
# 1. Tăng morphological operations
morphology:
  kernel_size: 7
  iterations: 3

# 2. Điều chỉnh adaptive threshold
threshold:
  block_size: 15  # Tăng từ 11
  C: 3  # Tăng từ 2
```

---

## 4. Vấn Đề Performance

### Vấn Đề: Xử lý quá chậm (FPS < 10)

**Triệu chứng:**
- Video lag
- FPS thấp
- CPU 100%

**Giải pháp:**

#### 1. Giảm Resolution

```python
# Thêm vào preprocessing trong main.py
frame = cv2.resize(frame, (640, 360))
```

#### 2. Skip Frames

```python
# Xử lý mỗi frame thứ 2
if frame_count % 2 != 0:
    continue
```

#### 3. Đơn giản hóa Pipeline

```yaml
# Dùng frame_diff thay MOG2 (nhanh hơn nhiều)
motion:
  method: "frame_diff"

# Tắt edge detection (nếu không cần thiết)
# Comment out edge detection trong pipeline
```

#### 4. Tắt Display

```bash
python src/main.py --no-display
```

#### 5. Đơn giản hóa ROI

- Dùng rectangle thay polygon
- Giảm số lượng ROI

---

### Vấn Đề: Out of Memory

**Triệu chứng:**
```
MemoryError: Unable to allocate array
```

**Giải pháp:**

```yaml
# 1. Giảm history trong MOG2
motion:
  history: 200  # Giảm từ 500

# 2. Xử lý video segments
# Chia video thành các đoạn ngắn

# 3. Tắt save_video nếu không cần
output:
  save_video: false
```

**Script xử lý segments:**

```python
import cv2

def process_segment(video_path, start_frame, end_frame):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for i in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        # Process frame...

    cap.release()

# Xử lý từng đoạn 1000 frames
total_frames = 3000
for start in range(0, total_frames, 1000):
    process_segment("video.mp4", start, start + 1000)
```

---

## 5. Vấn Đề Configuration

### Lỗi: YAML syntax error

**Triệu chứng:**
```
yaml.scanner.ScannerError: mapping values are not allowed here
```

**Nguyên nhân:**
- Thụt lề sai (tab/spaces không đồng nhất)
- Thiếu dấu hai chấm
- Giá trị string không quoted

**Giải pháp:**

```bash
# 1. Validate YAML online
# Paste nội dung vào: yamllint.com

# 2. Kiểm tra indentation
# YAML PHẢI dùng spaces, KHÔNG dùng tabs

# 3. Quote strings có ký tự đặc biệt
source: "path/with spaces/video.mp4"  # ✅ Đúng
source: path/with spaces/video.mp4    # ❌ Sai
```

---

### Lỗi: JSON ROI syntax error

**Triệu chứng:**
```
json.decoder.JSONDecodeError: Expecting property name
```

**Giải pháp:**

```bash
# Validate JSON online: jsonlint.com

# Kiểm tra:
# - Dấu phẩy cuối cùng (không được có)
# - Keys phải trong quotes
# - Quotes phải là double (")
```

**Ví dụ sai → đúng:**

```json
// ❌ SAI
{
  "name": "Area 1",  // Có comment (không được)
  "type": "polygon"   // Thiếu dấu phẩy
  'points': [[100,100]]  // Single quotes
  "color": [255, 0, 0],  // Dấu phẩy cuối (không được)
}

// ✅ ĐÚNG
{
  "name": "Area 1",
  "type": "polygon",
  "points": [[100, 100]],
  "color": [255, 0, 0]
}
```

---

## 6. Vấn Đề Platform-Specific

### macOS

#### Camera Permission

```bash
# System Preferences → Security & Privacy → Camera
# Tích chọn Terminal hoặc IDE đang dùng
```

#### File Permissions

```bash
chmod +x tools/roi_selector.py
chmod -R 755 data/
```

---

### Windows

#### Path Separators

```python
# ❌ SAI - Single backslash
path = "data\input\video.mp4"

# ✅ ĐÚNG - Forward slash hoặc double backslash
path = "data/input/video.mp4"
path = "data\\input\\video.mp4"
```

#### Antivirus

- Whitelist Python và thư mục project
- Tắt tạm real-time protection khi test

---

### Linux

#### Camera Device

```bash
# Liệt kê devices
ls -l /dev/video*

# Thử các device
python src/main.py --source /dev/video0
python src/main.py --source /dev/video1
```

#### Permissions

```bash
# Thêm user vào group video
sudo usermod -a -G video $USER

# Logout và login lại
# Hoặc
newgrp video
```

---

## 7. Debug Tools

### Enable Debug Mode

```bash
python src/main.py --debug
```

Hiển thị:
- Chi tiết mỗi bước xử lý
- Timing thông tin
- Shape của các arrays
- Warnings

### Test Individual Modules

```bash
# Test motion detection
python tests/test_motion.py

# Test thresholding
python tests/test_threshold.py

# Test intrusion detection
python tests/test_intrusion.py
```

### Check System Info

```bash
# OpenCV build info
python -c "import cv2; print(cv2.getBuildInformation())"

# Python packages
pip list

# System resources
free -h  # Linux
top      # CPU/Memory usage
```

---

## 8. Common Commands for Diagnosis

```bash
# Test video file
python -c "import cv2; cap=cv2.VideoCapture('video.mp4'); print('OK' if cap.isOpened() else 'FAIL')"

# Test camera
python -c "import cv2; cap=cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'FAIL')"

# Check OpenCV version
python -c "import cv2; print(cv2.__version__)"

# Test imports
python -c "import cv2, numpy, yaml, scipy; print('All OK')"

# Disk space
df -h

# Memory usage
free -h
```

---

## 9. Khi Vẫn Gặp Vấn Đề

### 1. Thu Thập Thông Tin

```bash
# Generate debug report
python tools/debug_report.py > debug.txt
```

### 2. Simplify Setup

- Bắt đầu lại với config tối thiểu
- Test với video ngắn (10 giây)
- Dùng ROI đơn giản (rectangle)
- Tắt các features không cần thiết

### 3. Compare With Working Example

```bash
# Chạy với example video và config
python src/main.py --config examples/working_config.yaml \
                   --source examples/sample_video.mp4
```

### 4. Check Documentation

- README.md trong project
- Implementation guides
- Code comments

### 5. Clean Reinstall

```bash
# Remove virtual environment
rm -rf venv

# Recreate
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 10. Prevention Tips

### 1. Version Control

```bash
git init
git add .
git commit -m "Working version"
```

### 2. Backup Configs

```bash
cp config/config.yaml config/config.backup
```

### 3. Test Incrementally

- Test từng module riêng trước khi integrate
- Test với data đơn giản trước data phức tạp

### 4. Document Changes

- Ghi lại tham số đã thử
- Note kết quả của từng config

---

**Ngày tạo**: Tháng 1/2025
**Phiên bản**: 1.0
