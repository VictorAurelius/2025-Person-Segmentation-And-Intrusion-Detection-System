# Hướng Dẫn Cấu Hình Hệ Thống

## Tổng Quan

File cấu hình `config/config.yaml` chứa tất cả các tham số để điều chỉnh hệ thống phát hiện xâm nhập. Việc cấu hình đúng là then chốt để đạt được độ chính xác cao.

---

## File Cấu Hình Chính: `config/config.yaml`

### 1. Cấu Hình Video Input

```yaml
video:
  source: "data/input/test_video.mp4"  # Đường dẫn video hoặc camera index
  fps: 30                              # Frame rate
```

**Các tùy chọn cho `source`:**
- **File path**: `"data/input/video.mp4"`
- **Webcam**: `0` (camera mặc định) hoặc `1, 2, ...` cho camera phụ
- **IP camera**: `"rtsp://username:password@ip:port/stream"`

**Ví dụ:**
```yaml
# Video file
source: "data/input/cua_chinh.mp4"

# Webcam
source: 0

# IP camera
source: "rtsp://admin:123456@192.168.1.100:554/stream1"
```

---

### 2. Cấu Hình Motion Detection

```yaml
motion:
  method: "MOG2"        # Phương pháp: "MOG2", "KNN", "frame_diff"
  history: 500          # Số frame để học background
  threshold: 16         # Độ nhạy (càng thấp càng nhạy)
  detect_shadows: true  # Phát hiện và đánh dấu bóng đổ
```

**Giải thích các tham số:**

#### `method` - Phương pháp phát hiện
- **MOG2**: Mixture of Gaussians 2 - Chính xác cao, tốt với ánh sáng thay đổi
- **KNN**: K-Nearest Neighbors - Nhanh hơn MOG2
- **frame_diff**: Frame differencing - Nhanh nhất, đơn giản nhất

#### `history` - Lịch sử background
- Số frame dùng để học background model
- **Giá trị cao** (500-1000): Background ổn định hơn, chậm thích nghi
- **Giá trị thấp** (200-300): Thích nghi nhanh với thay đổi

#### `threshold` - Ngưỡng phát hiện
- Độ nhạy trong phát hiện chuyển động
- **Giá trị thấp** (8-12): Nhạy cao, nhiều false positives
- **Giá trị trung bình** (16-25): Cân bằng
- **Giá trị cao** (30-50): Ít nhạy, ít false positives

#### `detect_shadows` - Phát hiện bóng
- `true`: Phát hiện và loại bỏ bóng đổ
- `false`: Coi bóng như chuyển động

**Presets theo điều kiện ánh sáng:**

```yaml
# Ban ngày (ngoài trời)
motion:
  method: "MOG2"
  history: 500
  threshold: 20
  detect_shadows: true

# Thiếu sáng (trong nhà)
motion:
  method: "MOG2"
  history: 300
  threshold: 12
  detect_shadows: true

# Ban đêm
motion:
  method: "MOG2"
  history: 200
  threshold: 10
  detect_shadows: false
```

---

### 3. Cấu Hình Adaptive Thresholding

```yaml
threshold:
  method: "gaussian"    # Phương pháp: "gaussian", "mean", "otsu"
  block_size: 11        # Kích thước vùng lân cận (phải là số lẻ)
  C: 2                  # Hằng số trừ đi từ mean
```

**Giải thích các tham số:**

#### `method` - Phương pháp threshold
- **gaussian**: Dùng Gaussian-weighted mean - Tốt nhất cho gradient mượt
- **mean**: Dùng mean đơn giản - Nhanh hơn
- **otsu**: Tự động tính threshold - Không cần block_size

#### `block_size` - Kích thước block
- Kích thước vùng lân cận để tính threshold local
- **Nhỏ** (7-11): Thích nghi local nhiều hơn, có thể nhiễu
- **Trung bình** (11-15): Cân bằng
- **Lớn** (15-25): Mượt hơn, ít nhiễu

#### `C` - Hằng số điều chỉnh
- Giá trị trừ đi từ mean để fine-tune threshold
- **C dương** (2-5): Threshold thấp hơn → nhiều foreground
- **C âm**: Threshold cao hơn → ít foreground

**Presets theo tình huống:**

```yaml
# Độ tương phản cao
threshold:
  method: "gaussian"
  block_size: 11
  C: 2

# Độ tương phản thấp
threshold:
  method: "gaussian"
  block_size: 15
  C: 3

# Nhiễu cao
threshold:
  method: "gaussian"
  block_size: 21
  C: 5
```

---

### 4. Cấu Hình Edge Detection

```yaml
edge:
  method: "canny"       # Phương pháp: "canny", "sobel", "prewitt", "scharr"
  low_threshold: 50     # Ngưỡng dưới (Canny)
  high_threshold: 150   # Ngưỡng trên (Canny)
```

**Giải thích các tham số:**

#### `method` - Phương pháp phát hiện biên
- **canny**: Chính xác nhất, có hysteresis thresholding
- **sobel**: Nhanh, gradient-based
- **prewitt**: Tương tự Sobel
- **scharr**: Chính xác hơn Sobel một chút

#### `low_threshold` và `high_threshold` (cho Canny)
- **Ratio khuyến nghị**: 1:2 hoặc 1:3
- **Nhiều edge**: `low: 30`, `high: 100`
- **Cân bằng**: `low: 50`, `high: 150`
- **Ít edge (chỉ edge mạnh)**: `low: 100`, `high: 200`

**Ví dụ:**

```yaml
# Phát hiện nhiều edge
edge:
  method: "canny"
  low_threshold: 30
  high_threshold: 100

# Chỉ edge rõ nét
edge:
  method: "canny"
  low_threshold: 100
  high_threshold: 200

# Dùng Sobel (nhanh hơn)
edge:
  method: "sobel"
  low_threshold: 50   # Dùng cho magnitude threshold
  high_threshold: 150
```

---

### 5. Cấu Hình Morphological Operations

```yaml
morphology:
  kernel_size: 5   # Kích thước structuring element
  iterations: 2    # Số lần áp dụng
```

**Mục đích:**
- **Opening** (erosion → dilation): Loại bỏ nhiễu
- **Closing** (dilation → erosion): Lấp lỗ hổng
- **Làm mượt** boundary

**Điều chỉnh:**
- **kernel_size nhỏ** (3-5): Ảnh hưởng nhẹ
- **kernel_size lớn** (7-9): Ảnh hưởng mạnh hơn
- **iterations nhiều**: Hiệu ứng mạnh hơn

---

### 6. Cấu Hình Intrusion Detection

```yaml
intrusion:
  roi_file: "data/roi/restricted_area.json"
  overlap_threshold: 0.3   # Tỷ lệ chồng lấn tối thiểu (30%)
  time_threshold: 1.0      # Thời gian tối thiểu trong ROI (giây)
  min_object_area: 1000    # Diện tích tối thiểu (pixels)
```

**Giải thích các tham số:**

#### `roi_file` - File định nghĩa ROI
- Đường dẫn đến file JSON chứa các restricted areas

#### `overlap_threshold` - Ngưỡng chồng lấn
- Tỷ lệ object phải overlap với ROI để kích hoạt alert
- **0.2-0.3**: Lenient (cảnh báo sớm)
- **0.4-0.5**: Balanced
- **0.6-0.8**: Strict (chỉ cảnh báo khi trong ROI hoàn toàn)

#### `time_threshold` - Ngưỡng thời gian
- Thời gian object phải ở trong ROI trước khi alert
- **0.5-1.0 giây**: Cảnh báo nhanh
- **1.5-2.0 giây**: Xác nhận thật sự xâm nhập
- **> 2.0 giây**: Rất chắc chắn

#### `min_object_area` - Diện tích tối thiểu
- Lọc các object quá nhỏ (nhiễu, bóng)
- **500-1000**: Cho phép object nhỏ
- **1500-2000**: Chỉ object trung bình/lớn
- **> 3000**: Chỉ object rất lớn

**Ví dụ cấu hình:**

```yaml
# Phát hiện nghiêm ngặt
intrusion:
  roi_file: "data/roi/restricted_area.json"
  overlap_threshold: 0.5
  time_threshold: 2.0
  min_object_area: 2000

# Phát hiện lenient (cảnh báo sớm)
intrusion:
  roi_file: "data/roi/restricted_area.json"
  overlap_threshold: 0.2
  time_threshold: 0.5
  min_object_area: 800
```

---

### 7. Cấu Hình Alert System

```yaml
alert:
  visual: true                          # Hiển thị alert trên video
  audio: true                           # Phát âm thanh cảnh báo
  log_file: "data/output/alerts.log"   # File log
  save_screenshots: true                # Lưu screenshot khi alert
```

**Các tùy chọn:**
- **visual**: Hiển thị bounding box đỏ và banner
- **audio**: Phát beep sound (platform-dependent)
- **log_file**: Ghi lại tất cả alerts
- **save_screenshots**: Lưu frame khi có alert

---

### 8. Cấu Hình Output

```yaml
output:
  save_video: true                      # Lưu video đã xử lý
  output_path: "data/output/result.mp4" # Đường dẫn lưu
  show_realtime: true                   # Hiển thị realtime
```

**Tùy chọn:**
- **save_video = false**: Chỉ xử lý, không lưu (nhanh hơn)
- **show_realtime = false**: Headless mode (xử lý background)

---

## Presets Cấu Hình Hoàn Chỉnh

### Preset 1: Ngoài Trời Ban Ngày

```yaml
# Video Input
video:
  source: "data/input/outdoor_day.mp4"
  fps: 30

# Motion Detection
motion:
  method: "MOG2"
  history: 500
  threshold: 20
  detect_shadows: true

# Adaptive Thresholding
threshold:
  method: "gaussian"
  block_size: 11
  C: 2

# Edge Detection
edge:
  method: "canny"
  low_threshold: 50
  high_threshold: 150

# Morphology
morphology:
  kernel_size: 5
  iterations: 2

# Intrusion Detection
intrusion:
  roi_file: "data/roi/restricted_area.json"
  overlap_threshold: 0.3
  time_threshold: 1.0
  min_object_area: 1500

# Alert
alert:
  visual: true
  audio: true
  log_file: "data/output/alerts.log"
  save_screenshots: true

# Output
output:
  save_video: true
  output_path: "data/output/result.mp4"
  show_realtime: true
```

### Preset 2: Trong Nhà Thiếu Sáng

```yaml
# Motion Detection
motion:
  method: "MOG2"
  history: 300
  threshold: 12
  detect_shadows: true

# Adaptive Thresholding
threshold:
  method: "gaussian"
  block_size: 15
  C: 3

# Edge Detection
edge:
  method: "canny"
  low_threshold: 30
  high_threshold: 100

# Intrusion
intrusion:
  overlap_threshold: 0.3
  time_threshold: 1.5
  min_object_area: 1000
```

### Preset 3: Ban Đêm

```yaml
# Motion Detection
motion:
  method: "MOG2"
  history: 200
  threshold: 10
  detect_shadows: false

# Adaptive Thresholding
threshold:
  method: "gaussian"
  block_size: 21
  C: 5

# Edge Detection
edge:
  method: "canny"
  low_threshold: 20
  high_threshold: 80

# Intrusion
intrusion:
  overlap_threshold: 0.4
  time_threshold: 2.0
  min_object_area: 1200
```

---

## Chiến Lược Parameter Tuning

### 1. Bắt Đầu Với Defaults
- Dùng config mặc định trước
- Chạy trên video test
- Quan sát kết quả

### 2. Điều Chỉnh Motion Detection
- **Quá nhiều false positives**: Tăng `threshold`
- **Bỏ sót chuyển động**: Giảm `threshold`
- **Background thay đổi nhiều**: Giảm `history`

### 3. Điều Chỉnh Intrusion Detection
- **Cảnh báo quá sớm**: Tăng `time_threshold`
- **Bỏ sót xâm nhập**: Giảm `overlap_threshold`
- **Nhiễu trigger alerts**: Tăng `min_object_area`

### 4. Fine-tuning
- Điều chỉnh từng tham số một
- Test trên nhiều video khác nhau
- Ghi lại kết quả

---

## Validation Cấu Hình

### Script Kiểm Tra

```python
import yaml

def validate_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Kiểm tra các key bắt buộc
    required = ['video', 'motion', 'threshold', 'edge', 'intrusion', 'alert', 'output']
    for key in required:
        assert key in config, f"Thiếu section: {key}"

    # Kiểm tra các giá trị
    assert config['threshold']['block_size'] % 2 == 1, "block_size phải là số lẻ"
    assert 0 < config['intrusion']['overlap_threshold'] <= 1, "overlap_threshold phải từ 0-1"
    assert config['motion']['method'] in ['MOG2', 'KNN', 'frame_diff'], "method không hợp lệ"

    print("✅ Cấu hình hợp lệ")

# Cách dùng
validate_config('config/config.yaml')
```

---

## Các Bước Tiếp Theo

✅ Đã cấu hình hệ thống!

Tiếp theo: [5. Chạy Hệ Thống](5-running-system.md)

---

**Ngày tạo**: Tháng 1/2025
**Phiên bản**: 1.0
