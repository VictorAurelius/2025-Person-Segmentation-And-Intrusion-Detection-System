# Hướng Dẫn Chuẩn Bị Dữ Liệu

## 1. Yêu Cầu Video Input

### Định Dạng Được Hỗ Trợ
- **MP4** (khuyến nghị)
- **AVI**
- **MOV**
- **MKV**
- **Bất kỳ định dạng nào OpenCV hỗ trợ**

### Thông Số Khuyến Nghị
- **Độ phân giải**: Tối thiểu 640x480, khuyến nghị 1920x1080
- **Frame rate**: 25-30 FPS
- **Thời lượng**: 30 giây đến 5 phút (để test)
- **Nội dung**: Nhìn rõ khu vực cần giám sát
- **Ánh sáng**: Nhất quán (tránh thay đổi đột ngột)

### Tiêu Chí Chất Lượng
✅ Đặc điểm video tốt:
- Camera ổn định (dùng tripod hoặc giá đỡ cố định)
- Nhìn rõ người trong khung hình
- Ít rung lắc camera
- Độ tương phản tốt giữa người và background

❌ Nên tránh:
- Video nén quá mức (mất chất lượng)
- Độ phân giải quá thấp (<640x480)
- Motion blur quá nhiều
- Cảnh quá tối hoàn toàn

---

## 2. Lấy Video Test

### Phương Án A: Tải Video Mẫu

**Nguồn miễn phí:**
1. **Pexels.com** - Video stock miễn phí
   - Tìm kiếm: "surveillance", "people walking", "security camera"
   - https://www.pexels.com/videos/

2. **Pixabay.com** - Video stock miễn phí
   - https://pixabay.com/videos/

3. **YouTube** - Video Creative Commons
   - Lọc theo giấy phép Creative Commons
   - Dùng youtube-dl để tải

4. **VIRAT Video Dataset** - Bộ dữ liệu giám sát
   - http://viratdata.org/

**Ví dụ tải video:**
```bash
# Dùng wget
cd final-project/code/data/input
wget https://example.com/surveillance_video.mp4
```

### Phương Án B: Quay Video Riêng

**Dùng điện thoại:**
1. Đặt điện thoại ở góc độ cố định (dùng tripod/giá đỡ)
2. Quay khu vực bạn muốn giám sát
3. Nên bao gồm các tình huống:
   - Người đi vào khu vực cấm
   - Người đi gần (nhưng không vào)
   - Không có người
   - Nhiều người cùng lúc

**Chuyển video sang máy tính:**
```bash
# Copy từ điện thoại sang máy tính
cp /đường/dẫn/điện/thoại/video.mp4 final-project/code/data/input/
```

**Dùng webcam:**
```bash
# Test webcam trước
python -c "import cv2; cap = cv2.VideoCapture(0); print('Webcam OK' if cap.isOpened() else 'Lỗi')"
```

### Phương Án C: Dùng Camera Real-time

**USB Webcam:**
- Kết nối vào máy tính
- Dùng source index: `0`, `1`, `2`, v.v.

**IP Camera:**
- Lấy RTSP URL từ camera
- Định dạng: `rtsp://username:password@ip:port/stream`

---

## 3. Tiền Xử Lý Video (Tùy chọn)

### Thay Đổi Kích Thước Video

Nếu video quá lớn, giảm kích thước:

```python
import cv2

def resize_video(input_path, output_path, width=1280, height=720):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        resized = cv2.resize(frame, (width, height))
        out.write(resized)
        frame_count += 1

        if frame_count % 100 == 0:
            print(f"Đã xử lý {frame_count} frames...")

    cap.release()
    out.release()
    print(f"Hoàn tất! Tổng số frames: {frame_count}")

# Cách dùng
resize_video("data/input/video_lon.mp4", "data/input/video_resize.mp4")
```

### Cắt Video

Trích xuất đoạn cụ thể:

```python
import cv2

def trim_video(input_path, output_path, start_sec, end_sec):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for i in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    print(f"Đã trích xuất từ giây {start_sec} đến giây {end_sec}")

# Cách dùng: Lấy đoạn từ 30-60 giây
trim_video("data/input/video.mp4", "data/input/video_cat.mp4", 30, 60)
```

### Chuyển Đổi Định Dạng

```bash
# Dùng ffmpeg (nếu đã cài)
ffmpeg -i input.avi -c:v libx264 -preset fast -crf 22 output.mp4
```

---

## 4. Tổ Chức Dữ Liệu

### Cấu Trúc Thư Mục

```bash
cd final-project/code
mkdir -p data/input data/output data/roi
```

### Sắp Xếp Files

```
data/
├── input/                  # Video đầu vào
│   ├── canh_1_ban_ngay.mp4
│   ├── canh_2_thieusan.mp4
│   ├── canh_3_ban_dem.mp4
│   └── test_video.mp4
├── output/                 # Kết quả đầu ra
│   ├── result.mp4
│   ├── alerts.log
│   └── screenshots/
│       ├── alert_0001.jpg
│       └── alert_0002.jpg
└── roi/                    # Định nghĩa ROI
    └── restricted_area.json
```

### Quy Ước Đặt Tên

Dùng tên mô tả rõ ràng:
- ✅ `cua_chinh_van_phong_ban_ngay.mp4`
- ✅ `bai_do_xe_ban_dem.mp4`
- ✅ `hanh_lang_camera1.mp4`
- ❌ `video1.mp4`
- ❌ `test.mp4`

---

## 5. Kiểm Tra Thông Tin Video

### Lấy Thông Số Video

```python
import cv2

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Lỗi: Không mở được video")
        return

    # Lấy thông số
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0

    print(f"Thông tin video:")
    print(f"  Độ phân giải: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Tổng số frames: {frame_count}")
    print(f"  Thời lượng: {duration:.2f} giây")

    cap.release()

# Cách dùng
get_video_info("data/input/test_video.mp4")
```

### Script Kiểm Tra Nhanh

Lưu thành `tools/check_video.py`:

```python
import cv2
import sys

if len(sys.argv) < 2:
    print("Cách dùng: python tools/check_video.py <đường_dẫn_video>")
    sys.exit(1)

video_path = sys.argv[1]
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Không mở được video")
    sys.exit(1)

print("✅ Video mở thành công")

# Đọc frame đầu tiên
ret, frame = cap.read()
if ret:
    print("✅ Có thể đọc frames")
    cv2.imshow("Frame đầu tiên", frame)
    print("Nhấn phím bất kỳ để đóng...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("❌ Không đọc được frames")

cap.release()
```

**Cách dùng:**
```bash
python tools/check_video.py data/input/test_video.mp4
```

---

## 6. Các Tình Huống Test Mẫu

### Tình Huống 1: Ban Ngày (Ngoài trời)
- **Ánh sáng**: Sáng, ánh sáng tự nhiên
- **Thách thức**: Bóng đổ, độ tương phản cao
- **Tham số**: Cài đặt chuẩn
- **Kỳ vọng**: Độ chính xác cao

### Tình Huống 2: Thiếu Sáng (Trong nhà)
- **Ánh sáng**: Ánh sáng trong nhà mờ
- **Thách thức**: Độ tương phản thấp, nhiễu
- **Tham số**: Threshold thấp hơn, block_size lớn hơn
- **Kỳ vọng**: Độ chính xác trung bình

### Tình Huống 3: Ban Đêm (Ngoài trời)
- **Ánh sáng**: Rất thiếu sáng, có thể có IR
- **Thách thức**: Nhiễu cao, độ tương phản rất thấp
- **Tham số**: Khử nhiễu mạnh, CLAHE
- **Kỳ vọng**: Độ chính xác thấp hơn, false positives nhiều hơn

---

## 7. Checklist Trước Khi Xử Lý

### Danh Sách Kiểm Tra

- [ ] Video phát được trong media player (VLC, v.v.)
- [ ] Độ phân giải đủ rõ để nhìn thấy người
- [ ] Điều kiện ánh sáng phù hợp với use case
- [ ] Đường dẫn file đúng trong `config.yaml`
- [ ] Định dạng video được hỗ trợ (khuyến nghị MP4)
- [ ] Kích thước file hợp lý (không bị hỏng)

### Test Với Script Đơn Giản

```python
import cv2

video_path = "data/input/test_video.mp4"
cap = cv2.VideoCapture(video_path)

print("Đang test phát video...")
frame_count = 0

while cap.isOpened() and frame_count < 100:  # Test 100 frames đầu
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Test", frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"✅ Đã đọc thành công {frame_count} frames")
```

---

## 8. Backup và Version Control

### Backup Video Gốc

```bash
# Tạo backup
mkdir -p data/backup
cp data/input/*.mp4 data/backup/
```

### Dùng Git (Tùy chọn)

```bash
# Khởi tạo git (nếu chưa làm)
cd final-project
git init

# Thêm .gitignore cho file lớn
echo "*.mp4" >> .gitignore
echo "*.avi" >> .gitignore
echo "data/output/*" >> .gitignore

# Chỉ track code
git add code/src code/config code/tools
git commit -m "Commit ban đầu"
```

---

## 9. Xử Lý Sự Cố

### Lỗi: Video không phát được
**Kiểm tra:**
```bash
file data/input/video.mp4  # Kiểm tra loại file
ls -lh data/input/video.mp4  # Kiểm tra kích thước file
```

**Giải pháp:**
- Tải lại hoặc mã hóa lại video
- Thử định dạng khác

### Lỗi: Xử lý quá chậm
**Giải pháp:**
- Giảm độ phân giải video
- Giảm FPS (lấy mỗi frame thứ 2)

### Lỗi: Hết dung lượng đĩa
**Giải pháp:**
```bash
# Kiểm tra dung lượng đĩa
df -h

# Dọn thư mục output
rm -rf data/output/*
```

---

## 10. Các Bước Tiếp Theo

✅ Dữ liệu đã được chuẩn bị!

Tiếp theo: [3. Định Nghĩa ROI](3-roi-definition.md)

---

**Ngày tạo**: Tháng 1/2025
**Phiên bản**: 1.0
