# Script: Tạo Phần Mở Đầu

## Mục Tiêu

Tạo file `documentation/report/00-introduction.md` chứa phần mở đầu của báo cáo.

## Yêu Cầu Output

- **File output**: `documentation/report/00-introduction.md`
- **Độ dài**: Khoảng 5-7 trang (font Times New Roman 13, 1.5 line spacing)
- **Format**: Plain text, không dùng markdown syntax (#, *, -, >)
- **Ngôn ngữ**: Tiếng Việt học thuật

## Cấu Trúc Nội Dung

### PHẦN MỞ ĐẦU

#### 1. Trang Bìa

**Nội dung cần viết:**

```
ĐẠI HỌC [TÊN TRƯỜNG]
KHOA [TÊN KHOA]
---

BÁO CÁO ĐỒ ÁN MÔN HỌC

XỬ LÝ ÁNH

---

ĐỀ TÀI:
PHÂN VÙNG NGƯỜI VÀ PHÁT HIỆN XÂMNHẬP KHU VỰC CẤM

---

SINH VIÊN THỰC HIỆN: [Họ và tên]
MSSV: [Mã số sinh viên]
LỚP: [Tên lớp]

GIẢNG VIÊN HƯỚNG DẪN: [Họ và tên giảng viên]

---

Thành phố Hồ Chí Minh, tháng [X] năm 2025
```

---

#### 2. Lời Cảm Ơn

**Nội dung cần viết:**

Một đoạn văn ngắn (0.5 trang) bày tỏ lòng biết ơn:
- Cảm ơn giảng viên hướng dẫn
- Cảm ơn gia đình, bạn bè hỗ trợ
- Cảm ơn các tài liệu, dataset công khai
- Cảm ơn cộng đồng OpenCV

**Mẫu:**
```
LỜI CẢM ƠN

Em xin chân thành cảm ơn [Tên giảng viên], người đã tận tình hướng dẫn,
giúp đỡ em trong suốt quá trình thực hiện đồ án này. Những góp ý,
chỉ dẫn của thầy/cô đã giúp em hoàn thiện kiến thức và kỹ năng về
xử lý ảnh.

Em cũng xin cảm ơn gia đình và bạn bè đã luôn động viên, khuyến khích
em trong quá trình học tập và nghiên cứu.

Cuối cùng, em xin gửi lời cảm ơn đến cộng đồng OpenCV và các nhà nghiên
cứu đã công bố dataset, tài liệu công khai, tạo điều kiện cho em thực
hiện đồ án này.

Thành phố Hồ Chí Minh, tháng [X] năm 2025
Sinh viên
[Họ và tên]
```

---

#### 3. Tóm Tắt (Abstract)

**Nội dung cần viết:**

Một đoạn văn ngắn gọn (1-2 đoạn, khoảng 0.5-1 trang) tóm tắt toàn bộ đồ án:

**Cấu trúc:**
1. **Giới thiệu bài toán**: Phát hiện người xâm nhập vùng cấm là vấn đề quan trọng trong giám sát an ninh
2. **Mục tiêu**: Xây dựng hệ thống tự động phát hiện và cảnh báo xâm nhập
3. **Phương pháp**: Sử dụng các kỹ thuật xử lý ảnh truyền thống
   - Motion detection (MOG2, KNN, Frame Differencing)
   - Adaptive thresholding
   - Edge detection (Canny)
   - Region growing
   - Intrusion detection (IoU, time-based validation)
4. **Kết quả**: Đạt detection rate ~88%, FPS ~27, false positive ~6.5%
5. **Ứng dụng**: Giám sát an ninh, kiểm soát ra vào, an ninh công cộng

**Mẫu:**
```
TÓM TẮT

Phát hiện người xâm nhập vùng cấm là một bài toán quan trọng trong lĩnh vực
giám sát an ninh, được ứng dụng rộng rãi tại sân bay, nhà máy, khu vực công
cộng. Đồ án này nhằm xây dựng một hệ thống tự động phát hiện và cảnh báo khi
có người xâm nhập vào khu vực bị hạn chế, sử dụng các kỹ thuật xử lý ảnh số.

Hệ thống được phát triển dựa trên pipeline xử lý ảnh kết hợp nhiều kỹ thuật:
(1) Motion detection bằng Background Subtraction (MOG2, KNN) và Frame
Differencing để phát hiện chuyển động, (2) Adaptive thresholding và CLAHE để
xử lý điều kiện ánh sáng không đồng đều, (3) Edge detection (Canny) để phát
hiện đường viền, (4) Region growing để phân vùng người, (5) Intrusion detection
dựa trên IoU và time-based validation để giảm false positive. Vùng cấm (ROI)
được định nghĩa linh hoạt bằng polygon tùy chỉnh.

Kết quả thực nghiệm trên 2 video test cho thấy hệ thống đạt detection rate
khoảng 88%, false positive rate khoảng 6.5%, và tốc độ xử lý real-time với
27 FPS trung bình. Hệ thống có khả năng thích ứng với thay đổi ánh sáng, dễ
cấu hình, và có tiềm năng ứng dụng cao trong thực tế.
```

---

#### 4. Mục Lục

**Nội dung cần viết:**

Liệt kê cấu trúc báo cáo (sẽ tạo tự động trong Word sau này):

```
MỤC LỤC

PHẦN MỞ ĐẦU
   Lời cảm ơn
   Tóm tắt
   Mục lục
   Danh sách hình
   Danh sách bảng

CHƯƠNG 1: CƠ SỞ LÝ THUYẾT ......................................... [trang]
   1.1. Tổng Quan Về Xử Lý Ảnh Số ................................... [trang]
   1.2. Motion Detection - Phát Hiện Chuyển Động ................... [trang]
   1.3. Adaptive Thresholding - Ngưỡng Hóa Thích Ứng ............... [trang]
   1.4. Edge Detection - Phát Hiện Biên ............................ [trang]
   1.5. Region Growing - Mở Rộng Vùng .............................. [trang]
   1.6. Intrusion Detection - Phát Hiện Xâm Nhập ................... [trang]
   1.7. Các Yếu Tố Ảnh Hưởng Đến Chất Lượng ........................ [trang]

CHƯƠNG 2: CƠ SỞ THỰC HÀNH ......................................... [trang]
   2.1. Quy Trình Thu Thập và Chuẩn Bị Dữ Liệu ..................... [trang]
   2.2. Kiến Trúc Hệ Thống .......................................... [trang]
   2.3. Phân Tích Chi Tiết Các Kỹ Thuật Áp Dụng .................... [trang]
      2.3.1. Motion Detection Module ............................... [trang]
      2.3.2. Adaptive Thresholding Module .......................... [trang]
      2.3.3. Edge Detection Module ................................. [trang]
      2.3.4. Intrusion Detection Module ............................ [trang]
      2.3.5. Alert System Module ................................... [trang]
   2.4. Cấu Hình và Tối Ưu Tham Số ................................. [trang]
   2.5. Quy Trình Thực Thi Hệ Thống ................................ [trang]
   2.6. Đánh Giá Kết Quả Thực Nghiệm ................................ [trang]
   2.7. So Sánh Với Các Phương Pháp Khác ............................ [trang]

CHƯƠNG 3: KẾT LUẬN VÀ ĐÁNH GIÁ ................................... [trang]
   3.1. Tóm Tắt Kết Quả Đạt Được .................................... [trang]
   3.2. Đánh Giá Hiệu Quả và Độ Chính Xác ........................... [trang]
   3.3. Đề Xuất Cải Tiến ............................................ [trang]
   3.4. Ứng Dụng Thực Tế ............................................ [trang]
   3.5. Kết Luận Chung .............................................. [trang]

TÀI LIỆU THAM KHẢO ................................................ [trang]

PHỤ LỤC ........................................................... [trang]
```

*Lưu ý: Số trang sẽ được cập nhật tự động khi tạo mục lục trong Word*

---

#### 5. Danh Sách Hình

**Nội dung cần viết:**

```
DANH SÁCH HÌNH

Hình 1.1: Biểu diễn ảnh số dưới dạng ma trận pixel ................. [trang]
Hình 1.2: Các color space phổ biến (RGB, Grayscale, HSV) ........... [trang]
Hình 1.3: Minh họa Frame Differencing ............................... [trang]
Hình 1.4: Quá trình Background Subtraction với MOG2 ................ [trang]
Hình 1.5: Ví dụ foreground mask từ MOG2 ............................. [trang]
Hình 1.6: So sánh Global vs Adaptive Thresholding .................. [trang]
Hình 1.7: Kết quả CLAHE trên ảnh thiếu sáng ........................ [trang]
Hình 1.8: 5 bước của thuật toán Canny Edge Detection ............... [trang]
Hình 1.9: So sánh kết quả Canny vs Sobel ........................... [trang]
Hình 1.10: Quá trình Region Growing từ seed points ................. [trang]
Hình 1.11: Ví dụ ROI polygon trên video giám sát ................... [trang]
Hình 1.12: Minh họa tính toán IoU .................................. [trang]
Hình 1.13: Quy trình phát hiện xâm nhập hoàn chỉnh ................. [trang]

Hình 2.1: Giao diện ROI Selector Tool .............................. [trang]
Hình 2.2: Ví dụ ROI được định nghĩa trên video test ................ [trang]
Hình 2.3: Sơ đồ kiến trúc hệ thống tổng thể ........................ [trang]
Hình 2.4: Flowchart xử lý một frame ................................. [trang]
Hình 2.5: So sánh foreground mask từ 3 phương pháp ................. [trang]
Hình 2.6: Kết quả adaptive thresholding trên frame thiếu sáng ...... [trang]
Hình 2.7: Edge detection kết hợp với motion mask ................... [trang]
Hình 2.8: Ví dụ tính toán IoU giữa bounding box và ROI ............. [trang]
Hình 2.9: Screenshot terminal khi chạy hệ thống .................... [trang]
Hình 2.10: Cấu trúc thư mục output ................................. [trang]
Hình 2.11: Frame có alert từ video input-01 ........................ [trang]
Hình 2.12: Frame có alert từ video input-02 ........................ [trang]
Hình 2.13: Biểu đồ phân bố alerts theo thời gian ................... [trang]

Hình 3.1: Frame output hoàn chỉnh với tất cả annotations ........... [trang]
Hình 3.2: Ví dụ screenshot alert có info overlay ................... [trang]
Hình 3.3: Sơ đồ kiến trúc mở rộng với GPU và multi-threading ....... [trang]
Hình 3.4: Flowchart xử lý occlusion với Kalman filter .............. [trang]
```

*Lưu ý: Số trang sẽ được cập nhật tự động khi tạo danh sách hình trong Word*

---

#### 6. Danh Sách Bảng

**Nội dung cần viết:**

```
DANH SÁCH BẢNG

Bảng 1.1: So sánh Frame Differencing vs Background Subtraction ..... [trang]
Bảng 1.2: Ảnh hưởng của các yếu tố môi trường đến hiệu năng ....... [trang]

Bảng 2.1: Thông số các video test .................................. [trang]
Bảng 2.2: So sánh FPS của các phương pháp motion detection ......... [trang]
Bảng 2.3: Tham số tối ưu cho các điều kiện ánh sáng ................ [trang]
Bảng 2.4: Kết quả thực nghiệm tuning tham số threshold ............. [trang]
Bảng 2.5: Tổng hợp kết quả trên 2 video test ....................... [trang]
Bảng 2.6: Confusion Matrix cho từng scenario ....................... [trang]
Bảng 2.7: So sánh metrics với các phương pháp khác ................. [trang]

Bảng 3.1: Tổng hợp kết quả định lượng .............................. [trang]
Bảng 3.2: So sánh mục tiêu vs kết quả thực tế ...................... [trang]
```

*Lưu ý: Số trang sẽ được cập nhật tự động khi tạo danh sách bảng trong Word*

---

## Hướng Dẫn Thực Hiện

### Bước 1: Thu thập thông tin

- Tên trường, khoa, lớp
- Tên sinh viên, MSSV
- Tên giảng viên
- Thời gian hoàn thành

### Bước 2: Viết nội dung

- Trang bìa: Điền thông tin cá nhân
- Lời cảm ơn: Viết theo mẫu, điều chỉnh cho phù hợp
- Tóm tắt: Tổng hợp từ kết quả 3 chương
- Mục lục, danh sách hình/bảng: Copy template

### Bước 3: Tạo file output

Tạo file `documentation/report/00-introduction.md` với toàn bộ nội dung.

---

## Yêu Cầu Format

1. **Không dùng markdown syntax** (#, *, -, >)
2. **Dùng plain text**
3. **Căn giữa** cho trang bìa
4. **Tóm tắt ngắn gọn**, dễ hiểu

---

## Checklist

- [ ] Đã thu thập thông tin cá nhân (tên, MSSV, trường, lớp, giảng viên)
- [ ] Đã đọc kết quả 3 chương để viết tóm tắt
- [ ] Trang bìa hoàn chỉnh
- [ ] Lời cảm ơn phù hợp
- [ ] Tóm tắt ngắn gọn, đầy đủ
- [ ] Mục lục, danh sách hình/bảng đầy đủ
- [ ] Format đúng yêu cầu

---

**Phiên bản**: 1.0
**Ngày tạo**: 28 Tháng 11, 2025
