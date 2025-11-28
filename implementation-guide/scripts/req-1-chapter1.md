# Script: Tạo Chương 1 - Cơ Sở Lý Thuyết

## Mục Tiêu

Tạo file `documentation/report/01-chapter1-theory.md` chứa nội dung Chương 1 của báo cáo.

## Yêu Cầu Output

- **File output**: `documentation/report/01-chapter1-theory.md`
- **Độ dài**: Khoảng 12-15 trang (font Times New Roman 13, 1.5 line spacing)
- **Format**: Plain text, không dùng markdown syntax (#, *, -, >)
- **Ngôn ngữ**: Tiếng Việt học thuật
- **Hình ảnh**: Đánh dấu vị trí `[Hình 1.X: Mô tả]` (không chèn ảnh thực)

## Cấu Trúc Nội Dung

### CHƯƠNG 1: CƠ SỞ LÝ THUYẾT

#### 1.1. Tổng Quan Về Xử Lý Ảnh Số (2 trang)

**Nội dung cần viết:**

1. **Giới thiệu xử lý ảnh số**
   - Định nghĩa ảnh số (digital image)
   - Biểu diễn ảnh: ma trận pixel, độ phân giải
   - Các loại ảnh: grayscale, RGB, HSV
   - Tham khảo: `knowledge-base/01-fundamentals/image-processing-basics.md`

2. **Vai trò trong giám sát an ninh**
   - Tại sao cần xử lý ảnh trong hệ thống phát hiện xâm nhập?
   - Các thách thức: thay đổi ánh sáng, che khuất, nhiễu
   - Ứng dụng thực tế

3. **Hình ảnh cần đánh dấu:**
   - `[Hình 1.1: Biểu diễn ảnh số dưới dạng ma trận pixel]`
   - `[Hình 1.2: Các color space phổ biến (RGB, Grayscale, HSV)]`

---

#### 1.2. Motion Detection - Phát Hiện Chuyển Động (2-3 trang)

**Nội dung cần viết:**

1. **Frame Differencing**
   - Nguyên lý: So sánh pixel giữa các frame liên tiếp
   - Công thức toán học (mô tả bằng văn bản)
   - Ưu điểm và nhược điểm
   - Tham khảo: `documentation/01-theory-foundation/1.1-frame-differencing.md`

2. **Background Subtraction**
   - Giới thiệu khái niệm trừ nền
   - Phương pháp MOG2 (Mixture of Gaussians):
     - Mô hình hóa mỗi pixel bằng hỗn hợp Gaussian
     - Cập nhật mô hình nền theo thời gian
     - Xử lý thay đổi ánh sáng
   - Phương pháp KNN (K-Nearest Neighbors)
   - Tham khảo: `knowledge-base/02-motion-detection/background-subtraction.md`

3. **So sánh các phương pháp**
   - `[Bảng 1.1: So sánh Frame Differencing vs Background Subtraction]`
   - Tiêu chí: Tốc độ, độ chính xác, khả năng thích ứng ánh sáng
   - Lý do chọn MOG2 cho hệ thống

4. **Hình ảnh cần đánh dấu:**
   - `[Hình 1.3: Minh họa Frame Differencing]`
   - `[Hình 1.4: Quá trình Background Subtraction với MOG2]`
   - `[Hình 1.5: Ví dụ foreground mask từ MOG2]`

---

#### 1.3. Adaptive Thresholding - Ngưỡng Hóa Thích Ứng (1-2 trang)

**Nội dung cần viết:**

1. **Khái niệm ngưỡng hóa**
   - Global thresholding vs Adaptive thresholding
   - Tại sao cần adaptive trong môi trường ánh sáng không đồng đều?

2. **Phương pháp Gaussian Adaptive Threshold**
   - Nguyên lý: Tính ngưỡng cục bộ cho từng vùng
   - Tham số: block_size, constant C
   - Ứng dụng trong bài toán
   - Tham khảo: `documentation/01-theory-foundation/1.2-adaptive-thresholding.md`

3. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**
   - Mục đích: Cải thiện độ tương phản
   - Ứng dụng: Xử lý video thiếu sáng/ban đêm

4. **Hình ảnh cần đánh dấu:**
   - `[Hình 1.6: So sánh Global vs Adaptive Thresholding]`
   - `[Hình 1.7: Kết quả CLAHE trên ảnh thiếu sáng]`

---

#### 1.4. Edge Detection - Phát Hiện Biên (2 trang)

**Nội dung cần viết:**

1. **Khái niệm biên trong ảnh**
   - Biên là gì? Tại sao quan trọng?
   - Gradient và đạo hàm trong xử lý ảnh

2. **Thuật toán Canny Edge Detection**
   - 5 bước của Canny:
     1. Gaussian smoothing (làm mịn)
     2. Tính gradient (Sobel operator)
     3. Non-maximum suppression
     4. Double thresholding
     5. Edge tracking by hysteresis
   - Ưu điểm: Chính xác, ít nhiễu
   - Tham số: low_threshold, high_threshold
   - Tham khảo: `documentation/01-theory-foundation/1.3-edge-detection.md`

3. **Sobel Operator**
   - Kernel Sobel ngang và dọc
   - So sánh với Canny

4. **Vai trò trong hệ thống**
   - Phát hiện đường viền người
   - Hỗ trợ region growing

5. **Hình ảnh cần đánh dấu:**
   - `[Hình 1.8: 5 bước của thuật toán Canny Edge Detection]`
   - `[Hình 1.9: So sánh kết quả Canny vs Sobel]`

---

#### 1.5. Region Growing - Mở Rộng Vùng (1-2 trang)

**Nội dung cần viết:**

1. **Nguyên lý thuật toán**
   - Khởi tạo seed points (điểm giống)
   - Tiêu chí tương đồng (similarity criteria)
   - Mở rộng dần vùng theo điều kiện
   - Tham khảo: `documentation/01-theory-foundation/1.4-region-growing.md`

2. **Ứng dụng trong phân vùng người**
   - Kết hợp với motion mask và edge detection
   - Tách người khỏi nền phức tạp

3. **Hình ảnh cần đánh dấu:**
   - `[Hình 1.10: Quá trình Region Growing từ seed points]`

---

#### 1.6. Intrusion Detection - Phát Hiện Xâm Nhập (2 trang)

**Nội dung cần viết:**

1. **ROI (Region of Interest)**
   - Định nghĩa vùng cấm bằng polygon
   - Lưu trữ trong JSON format

2. **IoU (Intersection over Union)**
   - Công thức tính độ chồng lấp (mô tả văn bản)
   - Ngưỡng overlap_threshold để kích hoạt cảnh báo
   - Tham khảo: `documentation/01-theory-foundation/1.5-intrusion-detection.md`

3. **Time-based validation**
   - Ngưỡng thời gian để tránh false positive
   - Tracking đối tượng qua nhiều frame

4. **Hình ảnh cần đánh dấu:**
   - `[Hình 1.11: Ví dụ ROI polygon trên video giám sát]`
   - `[Hình 1.12: Minh họa tính toán IoU]`
   - `[Hình 1.13: Quy trình phát hiện xâm nhập hoàn chỉnh]`

---

#### 1.7. Các Yếu Tố Ảnh Hưởng Đến Chất Lượng (1-2 trang)

**Nội dung cần viết:**

1. **Độ phân giải**
   - Ảnh hưởng đến độ chi tiết và tốc độ xử lý
   - Trade-off: 1080p vs 720p vs 480p

2. **Điều kiện ánh sáng**
   - Ban ngày: Dễ phát hiện
   - Thiếu sáng: Cần CLAHE, tăng sensitivity
   - Ban đêm: Nhiễu cao, khó khăn nhất
   - Tham khảo: `documentation/02-practical-implementation/2.4-parameter-tuning.md`

3. **Nhiễu ảnh**
   - Nguồn gốc: Sensor camera, nén video
   - Giải pháp: Gaussian blur, median filter, morphological operations

4. **Chuyển động camera**
   - Giả định: Camera cố định
   - Hạn chế: Không xử lý camera di động

5. **Bảng tổng hợp:**
   - `[Bảng 1.2: Ảnh hưởng của các yếu tố môi trường đến hiệu năng]`

---

## Hướng Dẫn Thực Hiện

### Bước 1: Đọc tài liệu tham khảo

Đọc các file sau để thu thập thông tin:
- `knowledge-base/01-fundamentals/image-processing-basics.md`
- `knowledge-base/01-fundamentals/opencv-essentials.md`
- `knowledge-base/02-motion-detection/background-subtraction.md`
- `knowledge-base/02-motion-detection/frame-differencing.md`
- `knowledge-base/03-segmentation/thresholding-techniques.md`
- `documentation/01-theory-foundation/1.1-frame-differencing.md`
- `documentation/01-theory-foundation/1.2-adaptive-thresholding.md`
- `documentation/01-theory-foundation/1.3-edge-detection.md`
- `documentation/01-theory-foundation/1.4-region-growing.md`
- `documentation/01-theory-foundation/1.5-intrusion-detection.md`

### Bước 2: Tạo thư mục output

```bash
mkdir -p documentation/report
```

### Bước 3: Tổng hợp và viết nội dung

- Viết theo cấu trúc 7 mục như trên
- Mỗi mục 1-2 trang
- Ngôn ngữ học thuật, dễ hiểu
- Giải thích rõ ràng các khái niệm, công thức
- Không dùng code blocks

### Bước 4: Tạo file output

Tạo file `documentation/report/01-chapter1-theory.md` với toàn bộ nội dung.

---

## Yêu Cầu Format

1. **Không dùng markdown syntax** (#, *, -, >)
2. **Dùng plain text** với indent
3. **Đánh số thứ tự** cho các mục: 1.1, 1.2, ...
4. **Đánh dấu vị trí hình/bảng**: `[Hình 1.X: Mô tả chi tiết]`
5. **Giải thích thuật toán bằng văn bản**, không dùng code

---

## Checklist

- [ ] Đã đọc tất cả tài liệu tham khảo
- [ ] Đã hiểu rõ 7 kỹ thuật chính
- [ ] Đã tạo thư mục `documentation/report/`
- [ ] Nội dung đủ 12-15 trang
- [ ] Format đúng yêu cầu (plain text)
- [ ] Đã đánh dấu vị trí ~13 hình và 2 bảng

---

**Phiên bản**: 1.0
**Ngày tạo**: 28 Tháng 11, 2025
