# Hướng Dẫn Định Nghĩa ROI (Region of Interest)

## Tổng Quan

ROI (Region of Interest) xác định khu vực cấm, nơi hệ thống sẽ phát hiện xâm nhập. Định nghĩa ROI chính xác là yếu tố quan trọng để hệ thống hoạt động hiệu quả.

---

## Phương Pháp 1: Dùng Công Cụ ROI Selector (Khuyến nghị)

### Bước 1: Chuẩn Bị Video

Đảm bảo bạn có video test trong `data/input/`:
```bash
ls data/input/
# Kết quả: test_video.mp4 (hoặc video của bạn)
```

### Bước 2: Chạy ROI Selector

```bash
cd final-project/code
python tools/roi_selector.py --video data/input/test_video.mp4
```

### Bước 3: Định Nghĩa ROI Tương Tác

**Phím điều khiển:**
- **Click chuột trái**: Thêm điểm vào ROI hiện tại
- **Click chuột phải**: Hoàn thành ROI hiện tại (cần tối thiểu 3 điểm)
- **Phím 'c'**: Xóa các điểm hiện tại (bắt đầu lại)
- **Phím 'd'**: Xóa ROI vừa lưu
- **Phím 's'**: Lưu tất cả ROI và thoát
- **Phím 'q'**: Thoát không lưu

**Quy trình:**
1. Click các điểm để định nghĩa đường biên polygon
2. Click chuột phải khi hoàn thành
3. Lặp lại cho các ROI bổ sung
4. Nhấn 's' để lưu

### Bước 4: Xác Nhận Kết Quả

Kiểm tra file được tạo:
```bash
cat data/roi/restricted_area.json
```

Kết quả mong đợi:
```json
{
  "restricted_areas": [
    {
      "name": "Area 1",
      "type": "polygon",
      "points": [[120, 150], [450, 160], [430, 380], [100, 370]],
      "color": [255, 0, 0]
    }
  ]
}
```

---

## Phương Pháp 2: Chỉnh Sửa JSON Thủ Công

### ROI Dạng Polygon

Chỉnh sửa file `data/roi/restricted_area.json`:

```json
{
  "restricted_areas": [
    {
      "name": "Cửa Chính",
      "type": "polygon",
      "points": [
        [100, 100],   // Góc trên-trái
        [400, 100],   // Góc trên-phải
        [400, 300],   // Góc dưới-phải
        [100, 300]    // Góc dưới-trái
      ],
      "color": [255, 0, 0]  // Màu Blue trong BGR
    }
  ]
}
```

**Định dạng Points:**
- Mảng các tọa độ [x, y]
- Gốc tọa độ (0, 0) ở góc trên-trái
- x tăng sang phải
- y tăng xuống dưới
- Định nghĩa các đỉnh theo thứ tự (thuận hoặc ngược chiều kim đồng hồ)

### ROI Dạng Rectangle

```json
{
  "restricted_areas": [
    {
      "name": "Bãi Đỗ Xe",
      "type": "rectangle",
      "x": 500,         // Tọa độ x góc trên-trái
      "y": 200,         // Tọa độ y góc trên-trái
      "width": 200,     // Chiều rộng (pixels)
      "height": 150,    // Chiều cao (pixels)
      "color": [0, 0, 255]  // Màu Red trong BGR
    }
  ]
}
```

### Nhiều ROI

```json
{
  "restricted_areas": [
    {
      "name": "Khu Vực A",
      "type": "polygon",
      "points": [[100, 100], [300, 100], [300, 250], [100, 250]],
      "color": [255, 0, 0]
    },
    {
      "name": "Khu Vực B",
      "type": "rectangle",
      "x": 400,
      "y": 300,
      "width": 150,
      "height": 100,
      "color": [0, 255, 0]
    },
    {
      "name": "Khu Vực C",
      "type": "polygon",
      "points": [[600, 50], [750, 80], [720, 200], [580, 180]],
      "color": [0, 0, 255]
    }
  ]
}
```

---

## Phương Pháp 3: Tìm Tọa Độ Từ Ảnh

### Bước 1: Trích Xuất Frame Đầu Tiên

```python
import cv2

cap = cv2.VideoCapture("data/input/test_video.mp4")
ret, frame = cap.read()
cap.release()

if ret:
    # Lưu frame
    cv2.imwrite("data/roi/first_frame.jpg", frame)

    # Hiển thị với tọa độ
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Điểm: [{x}, {y}]")

    cv2.namedWindow("Click để lấy tọa độ")
    cv2.setMouseCallback("Click để lấy tọa độ", mouse_callback)
    cv2.imshow("Click để lấy tọa độ", frame)
    print("Click trên frame để lấy tọa độ. Nhấn 'q' để thoát.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

### Bước 2: Ghi Chú Tọa Độ

Click vào các góc của khu vực cấm và ghi lại tọa độ:
```
Điểm: [120, 150]
Điểm: [450, 160]
Điểm: [430, 380]
Điểm: [100, 370]
```

### Bước 3: Tạo JSON

Sử dụng các tọa độ đã ghi trong file `restricted_area.json`.

---

## Mã Màu (Định Dạng BGR)

Các màu phổ biến để hiển thị ROI:

```python
# Định dạng BGR (không phải RGB!)
[255, 0, 0]     # Blue (Xanh dương)
[0, 255, 0]     # Green (Xanh lá)
[0, 0, 255]     # Red (Đỏ)
[255, 255, 0]   # Cyan (Lục lam)
[255, 0, 255]   # Magenta (Hồng tím)
[0, 255, 255]   # Yellow (Vàng)
[255, 255, 255] # White (Trắng)
[128, 128, 128] # Gray (Xám)
```

---

## Best Practices Thiết Kế ROI

### 1. Vùng Bao Phủ
✅ Bao phủ toàn bộ khu vực cấm
✅ Bao gồm vùng đệm xung quanh khu vực quan trọng
❌ Đừng làm ROI quá nhỏ (bỏ sót phát hiện)
❌ Đừng làm ROI toàn bộ frame (false positives)

### 2. Hình Dạng
✅ Dùng polygon cho khu vực không đều
✅ Dùng rectangle cho vùng đơn giản
✅ Giữ từ 3-8 đỉnh (polygon)
❌ Tránh hình dạng quá phức tạp (xử lý chậm)

### 3. Vị Trí
✅ Tính đến méo phối cảnh (perspective distortion)
✅ Xem xét góc camera
✅ Test với footage mẫu
❌ Đừng dựa vào điểm mù

### 4. Nhiều ROI
✅ Dùng màu khác nhau để dễ nhìn
✅ Đặt tên mô tả rõ ràng (VD: "Cửa Trước", "Bãi Xe")
✅ Không chồng lấn là tốt nhất
⚠️ ROI chồng lấn sẽ kích hoạt riêng biệt

---

## Test Định Nghĩa ROI

### Script Test Nhanh

Lưu thành `tools/test_roi.py`:

```python
import cv2
import json
import sys

if len(sys.argv) < 3:
    print("Cách dùng: python tools/test_roi.py <video> <roi_json>")
    sys.exit(1)

video_path = sys.argv[1]
roi_path = sys.argv[2]

# Load video
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Lỗi: Không đọc được video")
    sys.exit(1)

# Load ROI
with open(roi_path, 'r') as f:
    roi_data = json.load(f)

# Vẽ ROIs
for roi in roi_data['restricted_areas']:
    if roi['type'] == 'polygon':
        import numpy as np
        points = np.array(roi['points'], dtype=np.int32)
        cv2.polylines(frame, [points], True, tuple(roi['color']), 2)

        # Vẽ label ở giữa
        center = points.mean(axis=0).astype(int)
        cv2.putText(frame, roi['name'], tuple(center),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, tuple(roi['color']), 2)

    elif roi['type'] == 'rectangle':
        x, y, w, h = roi['x'], roi['y'], roi['width'], roi['height']
        cv2.rectangle(frame, (x, y), (x+w, y+h), tuple(roi['color']), 2)
        cv2.putText(frame, roi['name'], (x+5, y+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, tuple(roi['color']), 2)

# Hiển thị
cv2.imshow("Test ROI - Nhấn phím bất kỳ để đóng", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"✅ Đã load {len(roi_data['restricted_areas'])} ROI")
```

**Cách dùng:**
```bash
python tools/test_roi.py data/input/test_video.mp4 data/roi/restricted_area.json
```

---

## Các Mẫu ROI Phổ Biến

### 1. Giám Sát Cửa Ra Vào

```json
{
  "name": "Cửa Chính",
  "type": "polygon",
  "points": [[280, 150], [420, 150], [430, 450], [270, 450]],
  "color": [255, 0, 0]
}
```

### 2. Hàng Rào Chu Vi

```json
{
  "name": "Chu Vi",
  "type": "polygon",
  "points": [[50, 400], [650, 400], [650, 480], [50, 480]],
  "color": [0, 0, 255]
}
```

### 3. Khu Vực Cấm (Giữa)

```json
{
  "name": "Khu Vực Cấm",
  "type": "rectangle",
  "x": 300,
  "y": 200,
  "width": 280,
  "height": 240,
  "color": [0, 255, 0]
}
```

### 4. Nhiều Điểm Vào

```json
{
  "restricted_areas": [
    {
      "name": "Cửa 1",
      "type": "polygon",
      "points": [[100, 200], [200, 200], [200, 350], [100, 350]],
      "color": [255, 0, 0]
    },
    {
      "name": "Cửa 2",
      "type": "polygon",
      "points": [[500, 200], [600, 200], [600, 350], [500, 350]],
      "color": [0, 255, 0]
    }
  ]
}
```

---

## Kiểm Tra Tính Hợp Lệ

### Validate Định Dạng JSON

```bash
# Dùng Python
python -c "import json; json.load(open('data/roi/restricted_area.json')); print('✅ JSON hợp lệ')"
```

### Validate Cấu Trúc ROI

```python
import json

def validate_roi(roi_path):
    with open(roi_path, 'r') as f:
        data = json.load(f)

    assert 'restricted_areas' in data, "Thiếu key 'restricted_areas'"

    for i, roi in enumerate(data['restricted_areas']):
        print(f"Đang validate ROI {i+1}: {roi.get('name', 'Không tên')}")

        # Kiểm tra các trường bắt buộc
        assert 'name' in roi, f"ROI {i+1}: Thiếu 'name'"
        assert 'type' in roi, f"ROI {i+1}: Thiếu 'type'"
        assert 'color' in roi, f"ROI {i+1}: Thiếu 'color'"

        # Validate theo loại
        if roi['type'] == 'polygon':
            assert 'points' in roi, f"ROI {i+1}: Thiếu 'points'"
            assert len(roi['points']) >= 3, f"ROI {i+1}: Cần ít nhất 3 điểm"
        elif roi['type'] == 'rectangle':
            assert all(k in roi for k in ['x', 'y', 'width', 'height']), \
                f"ROI {i+1}: Thiếu tham số rectangle"

        print(f"  ✅ Hợp lệ")

    print(f"\n✅ Tất cả {len(data['restricted_areas'])} ROI đều hợp lệ")

# Cách dùng
validate_roi('data/roi/restricted_area.json')
```

---

## Xử Lý Sự Cố

### Lỗi: ROI không hiển thị trong output
**Kiểm tra:**
- Màu không phải [0, 0, 0] (đen)
- Các điểm nằm trong khung hình
- Đường dẫn file ROI trong config.yaml đúng

### Lỗi: Điểm nằm ngoài frame
**Giải pháp:**
```python
# Giới hạn điểm trong ranh giới frame
import numpy as np

def clip_points(points, width, height):
    points = np.array(points)
    points[:, 0] = np.clip(points[:, 0], 0, width-1)
    points[:, 1] = np.clip(points[:, 1], 0, height-1)
    return points.tolist()
```

### Lỗi: Lỗi cú pháp JSON
**Giải pháp:**
- Dùng công cụ validate online: jsonlint.com
- Kiểm tra dấu phẩy bị thiếu
- Kiểm tra dấu phẩy thừa ở cuối (không được phép)

---

## Các Bước Tiếp Theo

✅ ROI đã được định nghĩa!

Tiếp theo: [4. Cấu Hình](4-configuration.md)

---

**Ngày tạo**: Tháng 1/2025
**Phiên bản**: 1.0
