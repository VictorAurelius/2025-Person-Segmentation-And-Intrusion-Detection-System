# Hướng Dẫn Chạy Hệ Thống

## Quy Trình Thực Hiện Từng Bước

### Bước 1: Kích Hoạt Environment

```bash
cd final-project/code

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

Xác nhận đã kích hoạt: terminal hiển thị `(venv)`

---

### Bước 2: Xác Minh Setup

```bash
# Kiểm tra các file cần thiết
ls data/input/           # Có video chưa?
ls data/roi/             # Có file ROI chưa?
cat config/config.yaml   # Cấu hình đúng chưa?
```

**Checklist:**
- [ ] Video trong `data/input/`
- [ ] ROI trong `data/roi/restricted_area.json`
- [ ] `config.yaml` đã cấu hình đúng

---

### Bước 3: Chạy Chương Trình Chính

#### Cách 1: Chạy Cơ Bản

```bash
python src/main.py
```

Hệ thống sẽ:
1. Load cấu hình từ `config/config.yaml`
2. Mở video source
3. Load ROI definitions
4. Xử lý từng frame
5. Hiển thị kết quả realtime
6. Lưu output video và logs

#### Cách 2: Command-line Arguments

```bash
# Dùng config tùy chỉnh
python src/main.py --config custom_config.yaml

# Override video source
python src/main.py --source data/input/video_khac.mp4

# Dùng webcam
python src/main.py --source 0

# Tắt hiển thị (headless mode)
python src/main.py --no-display

# Bật debug logging
python src/main.py --debug

# Kết hợp nhiều options
python src/main.py --source 0 --debug --no-display
```

---

### Bước 4: Các Phím Điều Khiển Khi Đang Chạy

Trong cửa sổ hiển thị video:

| Phím | Chức năng |
|------|-----------|
| **q** | Thoát chương trình |
| **p** | Pause/Resume xử lý |
| **r** | Reset background model |
| **ESC** | Thoát |

**Ví dụ sử dụng:**
- Nhấn **'p'** để tạm dừng, xem kỹ frame hiện tại
- Nhấn **'r'** khi có thay đổi lớn về ánh sáng hoặc scene
- Nhấn **'q'** để thoát và lưu kết quả

---

### Bước 5: Output Mong Đợi

#### A. Console Output

```
[INFO] Loading configuration...
[INFO] Initializing video source...
[INFO] Video: 1280x720 @ 30 FPS
[INFO] Loading ROI definitions...
[INFO] ROI: 2 restricted areas loaded
[INFO] Starting detection pipeline...
[INFO] Processing frame 100/900 (11.1%) - FPS: 28.5
[ALERT] Intrusion detected at 00:05 in Area 1!
[INFO] Processing frame 200/900 (22.2%) - FPS: 29.1
...
[INFO] Processing complete. Saved to data/output/result.mp4
[INFO] Total alerts: 3
[INFO] Alert log: data/output/alerts.log
```

#### B. Cửa Sổ Hiển Thị

Bạn sẽ thấy:
- **Video gốc** với overlays
- **ROI areas** (polygon/rectangle màu đỏ/xanh)
- **Bounding boxes**:
  - Màu xanh lá: Object detected (không xâm nhập)
  - Màu đỏ: Intrusion detected
- **Alert banner** ở trên cùng khi có xâm nhập
- **Info panel** ở dưới cùng:
  - Frame number
  - FPS hiện tại
  - Số alerts
  - Số intrusions đang active

---

### Bước 6: Xem Lại Kết Quả

#### Output Video

```bash
# Mở với player mặc định

# Mac
open data/output/result.mp4

# Linux
xdg-open data/output/result.mp4

# Windows
start data/output/result.mp4

# Hoặc dùng VLC
vlc data/output/result.mp4
```

#### Alert Log

```bash
cat data/output/alerts.log
```

**Ví dụ nội dung log:**
```
2025-01-06 14:32:15 | Cửa Chính | 2.3s | Frame 150 | Center: (320, 240) | Area: 5234px | Screenshot: alert_0001.jpg
2025-01-06 14:32:18 | Bãi Xe | 1.8s | Frame 240 | Center: (450, 320) | Area: 4892px | Screenshot: alert_0002.jpg
```

#### Screenshots

```bash
ls data/output/screenshots/
# alert_0001.jpg, alert_0002.jpg, ...
```

---

## Chế Độ Nâng Cao

### 1. Interactive Mode (Điều chỉnh realtime)

```bash
python src/main.py --interactive
```

**Tính năng:**
- Điều chỉnh parameters trong khi chạy
- Trackbars cho threshold, sensitivity
- Test nhanh các giá trị khác nhau

**Sử dụng:**
- Dùng trackbars để điều chỉnh
- Quan sát thay đổi realtime
- Ghi lại giá trị tốt nhất

### 2. Batch Processing

Xử lý nhiều video cùng lúc:

```bash
# Tạo script batch_process.py
python tools/batch_process.py --input-dir data/input/ --output-dir data/output/
```

**Lợi ích:**
- Xử lý hàng loạt video
- Tự động áp dụng cùng config
- Tạo report tổng hợp

### 3. Real-time Camera Monitoring

```bash
# Sử dụng webcam
python src/main.py --source 0

# Sử dụng IP camera
python src/main.py --source "rtsp://192.168.1.100:554/stream"
```

**Lưu ý:**
- Camera phải được kết nối và có quyền truy cập
- IP camera cần credentials đúng
- Test kết nối trước khi chạy

---

## Monitoring Performance

### 1. Hiển Thị FPS

```bash
python src/main.py --show-fps
```

Hiển thị FPS realtime trên video.

### 2. Resource Profiling

```bash
python src/main.py --profile
```

Thu thập thông tin:
- CPU usage
- Memory usage
- Processing time per frame
- Bottlenecks

### 3. Stats Summary

Sau khi hoàn tất, hệ thống hiển thị:

```
================================================================================
PROCESSING SUMMARY
================================================================================
Total frames processed: 900
Total time: 31.25 seconds
Average FPS: 28.8
Total alerts: 5
Alert log: data/output/alerts.log
Screenshots: data/output/screenshots
================================================================================
```

---

## Troubleshooting Trong Lúc Chạy

### Vấn Đề 1: Không Hiển Thị Video

**Triệu chứng:** Console chạy nhưng không có cửa sổ

**Giải pháp:**
```bash
# Kiểm tra config
grep show_realtime config/config.yaml
# Phải là: show_realtime: true

# Hoặc force show
python src/main.py  # Mặc định sẽ show
```

### Vấn Đề 2: Quá Nhiều False Alerts

**Triệu chứng:** Alert liên tục dù không có xâm nhập thật

**Giải pháp:**
```bash
# Dừng chương trình (Ctrl+C)
# Chỉnh config:
# - Tăng overlap_threshold (0.3 → 0.5)
# - Tăng time_threshold (1.0 → 2.0)
# - Tăng min_object_area (1000 → 2000)
# Chạy lại
```

### Vấn Đề 3: Bỏ Sót Intrusions

**Triệu chứng:** Có người vào nhưng không alert

**Giải pháp:**
```bash
# Chỉnh config:
# - Giảm motion.threshold (16 → 10)
# - Giảm overlap_threshold (0.3 → 0.2)
# - Giảm time_threshold (1.0 → 0.5)
# - Giảm min_object_area
```

### Vấn Đề 4: Xử Lý Chậm

**Triệu chứng:** FPS < 10

**Giải pháp:**
```bash
# Tắt display để nhanh hơn
python src/main.py --no-display

# Hoặc giảm resolution trong video preprocessing
# Hoặc dùng frame_diff thay vì MOG2
```

### Vấn Đề 5: Camera Không Mở Được

**Triệu chứng:** Lỗi "Cannot open video source"

**Giải pháp:**
```bash
# Test camera trước
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"

# Thử camera index khác: 0, 1, 2
python src/main.py --source 1

# Kiểm tra permissions (macOS/Linux)
```

---

## Dừng Hệ Thống

### Cách 1: Graceful Shutdown

- Nhấn phím **'q'** trong cửa sổ video
- Hoặc **Ctrl+C** trong terminal

Hệ thống sẽ:
1. Hoàn tất xử lý frame hiện tại
2. Đóng video writer
3. Giải phóng resources
4. Hiển thị summary
5. Thoát sạch

### Cách 2: Force Stop

```bash
# Ctrl+C (2 lần)
# Hoặc
killall python
```

⚠️ **Cảnh báo:** Force stop có thể làm mất dữ liệu output chưa lưu

---

## Best Practices

### 1. Luôn Test Với Video Ngắn Trước

```bash
# Test 30 giây đầu
python src/main.py --source data/input/test_short.mp4
```

### 2. Backup Config Khi Tìm Ra Settings Tốt

```bash
cp config/config.yaml config/config_working.yaml
```

### 3. Kiểm Tra Disk Space Trước Khi Chạy

```bash
df -h
```

### 4. Dùng Screen/Tmux Cho Processing Lâu

```bash
# Chạy trong screen session
screen -S intrusion
python src/main.py --source video_dai.mp4
# Detach: Ctrl+A, D
# Reattach: screen -r intrusion
```

### 5. Log Output Cho Debugging

```bash
python src/main.py 2>&1 | tee run.log
```

---

## Các Bước Tiếp Theo

✅ Hệ thống đang chạy!

Nếu gặp vấn đề: [6. Xử Lý Sự Cố](6-troubleshooting.md)

---

**Ngày tạo**: Tháng 1/2025
**Phiên bản**: 1.0
