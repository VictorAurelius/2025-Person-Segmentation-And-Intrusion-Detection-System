# Script: Tạo Chương 2 - Cơ Sở Thực Hành

## Mục Tiêu

Tạo file `documentation/report/02-chapter2-practice.md` chứa nội dung Chương 2 của báo cáo.

## Yêu Cầu Output

- **File output**: `documentation/report/02-chapter2-practice.md`
- **Độ dài**: Khoảng 15-20 trang (font Times New Roman 13, 1.5 line spacing)
- **Format**: Plain text, không dùng markdown syntax (#, *, -, >)
- **Ngôn ngữ**: Tiếng Việt học thuật
- **Hình ảnh**: Đánh dấu vị trí `[Hình 2.X: Mô tả]`
- **Dữ liệu thực tế**: Sử dụng kết quả từ `code/data/output/`

## Cấu Trúc Nội Dung

### CHƯƠNG 2: CƠ SỞ THỰC HÀNH

#### 2.1. Quy Trình Thu Thập và Chuẩn Bị Dữ Liệu (2-3 trang)

**Nội dung cần viết:**

1. **Tiêu chí dữ liệu**
   - Loại: Video surveillance footage
   - Format: MP4, AVI (codec H.264/H.265)
   - Độ phân giải: 720p-1080p
   - Frame rate: 25-30 FPS
   - Độ dài: 30s - 5 phút/clip
   - Tham khảo: `implementation-guide/2-data-preparation.md`

2. **Nguồn dữ liệu**
   - Dataset công khai: VIRAT, CAVIAR
   - Video tự quay
   - Vị trí lưu trữ: `code/data/input/`
   - Danh sách file hiện có (kiểm tra thực tế)

3. **Tiền xử lý dữ liệu**
   - Chuẩn hóa resolution
   - Chuyển đổi color space
   - Noise reduction
   - Tham khảo code: `code/src/utils.py`

4. **Định nghĩa ROI**
   - Tool: `code/tools/roi_selector.py`
   - Cách sử dụng
   - Format JSON: `code/data/roi/restricted_area.json`

5. **Hình ảnh cần đánh dấu:**
   - `[Hình 2.1: Giao diện ROI Selector Tool]`
   - `[Hình 2.2: Ví dụ ROI được định nghĩa trên video]`
   - `[Bảng 2.1: Thông số các video test]`

---

#### 2.2. Kiến Trúc Hệ Thống (2 trang)

**Nội dung cần viết:**

1. **Tổng quan kiến trúc**
   - Pipeline xử lý: Input → Processing → Output
   - Các module chính và vai trò:
     - `main.py`, `motion_detector.py`, `adaptive_threshold.py`
     - `edge_detector.py`, `region_grower.py`, `intrusion_detector.py`
     - `alert_system.py`
   - Tham khảo: `documentation/02-practical-implementation/2.1-system-architecture.md`

2. **Luồng xử lý từng frame**
   - 8 bước xử lý (mô tả chi tiết)
   - Tham chiếu code: `code/src/main.py`

3. **Hình ảnh cần đánh dấu:**
   - `[Hình 2.3: Sơ đồ kiến trúc hệ thống tổng thể]`
   - `[Hình 2.4: Flowchart xử lý một frame]`

---

#### 2.3. Phân Tích Chi Tiết Các Kỹ Thuật Áp Dụng (4-5 trang)

**Nội dung cần viết:**

##### 2.3.1. Motion Detection Module

1. **Class MotionDetector**
   - Tham chiếu: `code/src/motion_detector.py`
   - Constructor, methods, tham số quan trọng
   - Lựa chọn phương pháp: MOG2/KNN/Frame Diff

2. **Kết quả thực nghiệm**
   - `[Bảng 2.2: So sánh FPS của các phương pháp]`
   - Test trên video thực tế
   - Độ chính xác

3. **Hình ảnh**: `[Hình 2.5: So sánh foreground mask]`

##### 2.3.2. Adaptive Thresholding Module

1. **Class AdaptiveThreshold**
   - Tham chiếu: `code/src/adaptive_threshold.py`
   - Tham số: block_size, C

2. **Class CLAHEProcessor**
   - Clip limit, tile grid size

3. **Hình ảnh**: `[Hình 2.6: Kết quả adaptive thresholding]`

##### 2.3.3. Edge Detection Module

1. **Class EdgeDetector**
   - Tham chiếu: `code/src/edge_detector.py`
   - Phương pháp Canny
   - Tham số: low/high threshold

2. **Hình ảnh**: `[Hình 2.7: Edge detection kết hợp motion mask]`

##### 2.3.4. Intrusion Detection Module

1. **Class IntrusionDetector**
   - Tham chiếu: `code/src/intrusion_detector.py`
   - Method `detect_intrusions()`: Input, output, thuật toán
   - Tham số: overlap_threshold, time_threshold, min_object_area

2. **Tracking mechanism**
   - Dictionary lưu trữ
   - Reset tracking

3. **Hình ảnh**: `[Hình 2.8: Tính toán IoU]`

##### 2.3.5. Alert System Module

1. **Class AlertSystem**
   - Tham chiếu: `code/src/alert_system.py`
   - Visual alert, audio alert, logging, screenshot
   - Cooldown mechanism

2. **Alert log format**
   - Mô tả cấu trúc
   - `[Log 2.1: Mẫu alerts.log từ video]` - Trích từ `code/data/output/input-01/alerts.log`

---

#### 2.4. Cấu Hình và Tối Ưu Tham Số (2 trang)

**Nội dung cần viết:**

1. **File cấu hình YAML**
   - Tham chiếu: `code/config/config.yaml`
   - Cấu trúc: video, motion, threshold, edge, intrusion, alert, output

2. **Tuning theo điều kiện ánh sáng**
   - Tham khảo: `documentation/02-practical-implementation/2.4-parameter-tuning.md`
   - `[Bảng 2.3: Tham số tối ưu cho các điều kiện ánh sáng]`
     - Ban ngày, thiếu sáng, ban đêm

3. **Thực nghiệm tuning**
   - Phương pháp
   - Metrics
   - Kết quả

4. **Bảng biểu**: `[Bảng 2.4: Kết quả thực nghiệm tuning]`

---

#### 2.5. Quy Trình Thực Thi Hệ Thống (2 trang)

**Nội dung cần viết:**

1. **Setup môi trường**
   - Python 3.8+
   - Dependencies
   - Cài đặt: requirements.txt
   - Tham khảo: `implementation-guide/1-environment-setup.md`

2. **Chạy hệ thống**
   - Command cơ bản
   - Với custom config
   - Với video cụ thể
   - Tham khảo: `implementation-guide/5-running-system.md`

3. **Output**
   - Video đã xử lý
   - Alert log
   - Screenshots

4. **Hình ảnh cần đánh dấu:**
   - `[Hình 2.9: Screenshot terminal khi chạy]`
   - `[Hình 2.10: Cấu trúc thư mục output]`

---

#### 2.6. Đánh Giá Kết Quả Thực Nghiệm (3-4 trang)

**Nội dung cần viết:**

1. **Test scenarios**
   - Tham khảo: `documentation/03-evaluation/3.1-test-scenarios.md`
   - 8 kịch bản test (S1-S8)
   - Ghi chú: Kịch bản nào đã test, kịch bản nào chưa

2. **Performance metrics**
   - Tham khảo: `documentation/03-evaluation/3.2-performance-metrics.md`
   - Định nghĩa metrics:
     - Detection Rate, False Positive Rate, False Negative Rate
     - Precision, Recall, F1-Score
     - FPS

3. **Kết quả thực tế**
   - Phân tích từ `code/data/output/input-01/` và `input-02/`
   - Đọc alerts.log để lấy số liệu
   - Detection rate, false positive rate, average FPS
   - `[Bảng 2.5: Tổng hợp kết quả trên 2 video test]`

4. **Phân tích alerts**
   - Trích dẫn từ alerts.log
   - `[Log 2.2: Top 10 alerts từ video input-01]`
   - Phân tích: thời điểm, duration, vị trí, kích thước

5. **Hình ảnh cần đánh dấu:**
   - `[Hình 2.11: Frame có alert từ video input-01]`
   - `[Hình 2.12: Frame có alert từ video input-02]`
   - `[Hình 2.13: Biểu đồ phân bố alerts theo thời gian]`
   - `[Bảng 2.6: Confusion Matrix]`

---

#### 2.7. So Sánh Với Các Phương Pháp Khác (2 trang)

**Nội dung cần viết:**

1. **So sánh với phương pháp truyền thống**
   - Tham khảo: `documentation/03-evaluation/3.4-comparison.md`
   - Phương pháp so sánh:
     - Static threshold
     - Simple frame differencing
     - OpenCV HOG + SVM
   - `[Bảng 2.7: So sánh metrics với các phương pháp khác]`

2. **Ưu điểm của hệ thống**
   - Adaptive to lighting
   - Time-based validation
   - Flexible ROI
   - Real-time capable
   - Modular design

3. **Nhược điểm và hạn chế**
   - Tham khảo: `documentation/03-evaluation/3.5-limitations.md`
   - Camera phải cố định
   - Khó khăn với occlusion nặng
   - False positive khi có chuyển động nền
   - Chưa tối ưu cho ban đêm
   - Không phân biệt authorized/unauthorized person

---

## Hướng Dẫn Thực Hiện

### Bước 1: Thu thập dữ liệu thực tế

```bash
# Kiểm tra video input có sẵn
ls -lh code/data/input/

# Kiểm tra kết quả output
ls -lh code/data/output/input-01/
ls -lh code/data/output/input-02/

# Đọc alerts.log
cat code/data/output/input-01/alerts.log
cat code/data/output/input-02/alerts.log
```

### Bước 2: Đọc tài liệu tham khảo

Đọc các file sau:
- `code/src/*.py` (tất cả các module)
- `code/config/config.yaml`
- `documentation/02-practical-implementation/*.md`
- `documentation/03-evaluation/*.md`
- `implementation-guide/*.md`

### Bước 3: Tổng hợp số liệu

- Đếm số alerts từ log files
- Tính toán metrics (detection rate, false positive, FPS)
- Chuẩn bị bảng biểu

### Bước 4: Viết nội dung

- Viết theo cấu trúc 7 mục
- Sử dụng số liệu thực tế
- Trích dẫn code/file khi cần
- Giải thích rõ ràng

### Bước 5: Tạo file output

Tạo file `documentation/report/02-chapter2-practice.md` với toàn bộ nội dung.

---

## Yêu Cầu Format

1. **Không dùng markdown syntax** (#, *, -, >)
2. **Dùng plain text** với indent
3. **Đánh số thứ tự**: 2.1, 2.2, 2.3.1, ...
4. **Đánh dấu vị trí hình/bảng**: `[Hình 2.X: ...]`, `[Bảng 2.X: ...]`
5. **Trích dẫn code**: `code/src/main.py:123`
6. **Trích dẫn log**: Sử dụng format `[Log 2.X: Mô tả]`

---

## Checklist

- [ ] Đã chạy hệ thống với tất cả video test
- [ ] Đã có kết quả trong `code/data/output/`
- [ ] Đã đọc tất cả module code
- [ ] Đã đọc tất cả tài liệu implementation và evaluation
- [ ] Đã thu thập số liệu thực tế từ alerts.log
- [ ] Đã tính toán metrics
- [ ] Nội dung đủ 15-20 trang
- [ ] Format đúng yêu cầu
- [ ] Đã đánh dấu vị trí ~12 hình, 5-7 bảng, 2-3 log excerpts

---

## Lưu Ý Quan Trọng

1. **Sử dụng dữ liệu thực tế**: Tất cả số liệu phải lấy từ kết quả chạy hệ thống thực tế
2. **Trích dẫn chính xác**: Tham chiếu file/code với đường dẫn cụ thể
3. **Phân tích sâu**: Không chỉ liệt kê, mà phải giải thích ý nghĩa của kết quả
4. **So sánh khách quan**: Nhận diện cả ưu điểm và nhược điểm

---

**Phiên bản**: 1.0
**Ngày tạo**: 28 Tháng 11, 2025
