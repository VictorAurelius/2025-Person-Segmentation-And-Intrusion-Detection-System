# Hướng Dẫn Tạo Báo Cáo Đề Tài: Phân Vùng Người & Phát Hiện Xâm Nhập

## Mục Tiêu

Tạo báo cáo học thuật chuyên nghiệp cho đề tài "Phân Vùng Người & Phát Hiện Xâm Nhập Khu Vực Cấm" với 3 chương chính.

## Yêu Cầu Định Dạng

- **Ngôn ngữ**: Tiếng Việt, phong cách học thuật sinh viên đại học
- **Format**: Markdown thuần túy, dễ copy vào Microsoft Word
- **Hình ảnh**: Chỉ đánh dấu vị trí `[Hình X.Y: Mô tả]`, không chèn ảnh thực
- **Code**: Không hiển thị code blocks, chỉ mô tả thuật toán bằng văn bản
- **Độ dài**: Khoảng 40-50 trang A4 (font Times New Roman 13, 1.5 line spacing)

---

## ⚠️ QUAN TRỌNG: Tránh Lỗi Output Token

Để tránh lỗi vượt quá 32000 output tokens, báo cáo được chia thành **nhiều phần riêng biệt**.

### Cấu Trúc File Output

Báo cáo sẽ được tạo thành các file markdown riêng:

```
documentation/report/
├── 00-introduction.md          # Phần mở đầu (5-7 trang)
├── 01-chapter1-theory.md       # Chương 1: Cơ Sở Lý Thuyết (12-15 trang)
├── 02-chapter2-practice.md     # Chương 2: Cơ Sở Thực Hành (15-20 trang)
├── 03-chapter3-conclusion.md   # Chương 3: Kết Luận (8-10 trang)
├── 04-references.md            # Tài liệu tham khảo
└── 05-appendix.md              # Phụ lục
```

### Quy Trình Tạo Báo Cáo

**BƯỚC 1**: Sử dụng các script riêng để tạo từng phần:

1. **Tạo Chương 1**: Sử dụng `req-1-chapter1.md`
   - Tạo file: `documentation/report/01-chapter1-theory.md`
   - Nội dung: Cơ sở lý thuyết về xử lý ảnh
   - Độ dài: ~12-15 trang

2. **Tạo Chương 2**: Sử dụng `req-1-chapter2.md`
   - Tạo file: `documentation/report/02-chapter2-practice.md`
   - Nội dung: Thực hành, kết quả, đánh giá
   - Độ dài: ~15-20 trang

3. **Tạo Chương 3**: Sử dụng `req-1-chapter3.md`
   - Tạo file: `documentation/report/03-chapter3-conclusion.md`
   - Nội dung: Kết luận, đề xuất, ứng dụng
   - Độ dài: ~8-10 trang

**BƯỚC 2**: Tạo các phần bổ sung:

4. **Tạo Phần Mở Đầu**: Sử dụng `req-1-intro.md`
   - Tạo file: `documentation/report/00-introduction.md`
   - Nội dung: Lời mở đầu, mục lục, danh sách hình/bảng

5. **Tạo Tài Liệu Tham Khảo**: Sử dụng `req-1-references.md`
   - Tạo file: `documentation/report/04-references.md`

**BƯỚC 3**: Kết hợp các phần lại (thủ công hoặc tự động)

```bash
# Kết hợp các file markdown (tuỳ chọn)
cd documentation/report
cat 00-introduction.md 01-chapter1-theory.md 02-chapter2-practice.md 03-chapter3-conclusion.md 04-references.md > full-report.md
```

---

## Cấu Trúc Nội Dung Chi Tiết

### PHẦN MỞ ĐẦU (00-introduction.md)

1. **Trang bìa**
   - Tên đề tài
   - Thông tin sinh viên
   - Thông tin môn học

2. **Lời cảm ơn**

3. **Tóm tắt** (Abstract)
   - Mục tiêu đề tài
   - Phương pháp sử dụng
   - Kết quả đạt được
   - 1-2 đoạn văn

4. **Mục lục**
   - Danh sách chương/mục
   - (Sẽ tạo tự động trong Word)

5. **Danh sách hình**
   - (Sẽ tạo tự động trong Word)

6. **Danh sách bảng**
   - (Sẽ tạo tự động trong Word)

---

### CHƯƠNG 1: CƠ SỞ LÝ THUYẾT (01-chapter1-theory.md)

**Tổng quan**: 7 mục chính, 12-15 trang

1.1. Tổng Quan Về Xử Lý Ảnh Số (2 trang)
1.2. Motion Detection (2-3 trang)
1.3. Adaptive Thresholding (1-2 trang)
1.4. Edge Detection (2 trang)
1.5. Region Growing (1-2 trang)
1.6. Intrusion Detection (2 trang)
1.7. Các Yếu Tố Ảnh Hưởng (1-2 trang)

**Tham khảo**:
- `knowledge-base/01-fundamentals/`
- `knowledge-base/02-motion-detection/`
- `knowledge-base/03-segmentation/`
- `documentation/01-theory-foundation/`

**Hình ảnh cần chèn**: ~15 hình, 2 bảng

---

### CHƯƠNG 2: CƠ SỞ THỰC HÀNH (02-chapter2-practice.md)

**Tổng quan**: 7 mục chính, 15-20 trang

2.1. Quy Trình Thu Thập và Chuẩn Bị Dữ Liệu (2-3 trang)
2.2. Kiến Trúc Hệ Thống (2 trang)
2.3. Phân Tích Chi Tiết Các Kỹ Thuật Áp Dụng (4-5 trang)
2.4. Cấu Hình và Tối Ưu Tham Số (2 trang)
2.5. Quy Trình Thực Thi Hệ Thống (2 trang)
2.6. Đánh Giá Kết Quả Thực Nghiệm (3-4 trang)
2.7. So Sánh Với Các Phương Pháp Khác (2 trang)

**Tham khảo**:
- `code/src/` (tất cả các module)
- `code/config/`
- `code/data/output/` (kết quả thực tế)
- `documentation/02-practical-implementation/`
- `documentation/03-evaluation/`

**Hình ảnh cần chèn**: ~12 hình, 5-7 bảng
**Log excerpts**: 2-3 đoạn

---

### CHƯƠNG 3: KẾT LUẬN VÀ ĐÁNH GIÁ (03-chapter3-conclusion.md)

**Tổng quan**: 5 mục chính, 8-10 trang

3.1. Tóm Tắt Kết Quả Đạt Được (2 trang)
3.2. Đánh Giá Hiệu Quả và Độ Chính Xác (2 trang)
3.3. Đề Xuất Cải Tiến (2-3 trang)
3.4. Ứng Dụng Thực Tế (2 trang)
3.5. Kết Luận Chung (1 trang)

**Tham khảo**:
- Kết quả từ Chương 2
- `documentation/03-evaluation/3.5-limitations.md`
- `knowledge-base/04-advanced-topics/`

**Hình ảnh cần chèn**: ~4 hình, 2 bảng

---

### TÀI LIỆU THAM KHẢO (04-references.md)

Danh sách tài liệu tham khảo theo chuẩn IEEE hoặc APA:
- Sách giáo trình
- Papers
- Documentation (OpenCV, NumPy, etc.)
- Datasets
- Online resources

---

## Hướng Dẫn Sử Dụng Script

### 1. Tạo Chương 1

```bash
# Trong Claude Code, prompt:
"Thực hiện script req-1-chapter1.md để tạo Chương 1"
```

**Output**: `documentation/report/01-chapter1-theory.md`

### 2. Tạo Chương 2

```bash
"Thực hiện script req-1-chapter2.md để tạo Chương 2"
```

**Lưu ý**: Cần chạy hệ thống trước để có kết quả thực tế:
```bash
cd code
python src/main.py --source data/input/input-01.mp4
python src/main.py --source data/input/input-02.mp4
```

**Output**: `documentation/report/02-chapter2-practice.md`

### 3. Tạo Chương 3

```bash
"Thực hiện script req-1-chapter3.md để tạo Chương 3"
```

**Output**: `documentation/report/03-chapter3-conclusion.md`

### 4. Tạo Phần Mở Đầu và Tài Liệu Tham Khảo

```bash
"Thực hiện script req-1-intro.md để tạo phần mở đầu"
"Thực hiện script req-1-references.md để tạo tài liệu tham khảo"
```

---

## Quy Trình Hoàn Thiện Báo Cáo

### Bước 1: Chạy Hệ Thống và Thu Thập Dữ Liệu

```bash
# 1. Chạy với video test
cd code
python src/main.py --source data/input/input-01.mp4
python src/main.py --source data/input/input-02.mp4

# 2. Thu thập metrics từ alerts.log
cat data/output/input-01/alerts.log
cat data/output/input-02/alerts.log

# 3. Chụp screenshots cần thiết
# - ROI selector tool
# - Terminal output
# - Output frames với alerts
```

### Bước 2: Tạo Từng Phần Báo Cáo

Sử dụng các script riêng để tạo từng chương (như hướng dẫn ở trên).

### Bước 3: Kết Hợp và Format trong Word

1. **Mở Word**, tạo file mới
2. **Copy nội dung** từ các file markdown theo thứ tự:
   - 00-introduction.md
   - 01-chapter1-theory.md
   - 02-chapter2-practice.md
   - 03-chapter3-conclusion.md
   - 04-references.md
3. **Format**:
   - Font: Times New Roman 13
   - Line spacing: 1.5
   - Margin: 2.5cm (trái), 2cm (các cạnh khác)
4. **Chèn hình ảnh** vào các vị trí đã đánh dấu `[Hình X.Y: ...]`
5. **Tạo mục lục tự động**: References → Table of Contents
6. **Tạo danh sách hình/bảng**: References → Insert Table of Figures

### Bước 4: Review và Hoàn Thiện

1. **Đọc lại toàn bộ** báo cáo
2. **Kiểm tra**:
   - Logic mạch lạc
   - Chính tả, ngữ pháp
   - Hình ảnh rõ ràng, đúng vị trí
   - Tham chiếu chính xác
3. **Export PDF** để backup
4. **Nộp** theo yêu cầu

---

## Lưu Ý Quan Trọng

### Về Nội Dung

1. **Ngôn ngữ học thuật** nhưng dễ hiểu
2. **Trích dẫn code/file** khi cần (ví dụ: `code/src/main.py:123`)
3. **Dùng số liệu thực tế** từ kết quả chạy hệ thống
4. **So sánh với ground truth** để tính accuracy chính xác
5. **Giải thích rõ ràng** các khái niệm, thuật toán

### Về Format

1. **Không dùng markdown syntax** trong file output (#, *, -, >)
2. **Sử dụng plain text** với indent phù hợp
3. **Đánh dấu vị trí hình/bảng**: `[Hình X.Y: Mô tả chi tiết]`
4. **Mô tả thuật toán bằng văn bản**, không dùng code blocks

### Về Screenshots và Bằng Chứng

1. **Screenshots cần thiết**:
   - ROI selector interface
   - Terminal output khi chạy
   - Output frames có alerts
   - Folder structure
   - Ví dụ từ alerts.log
2. **Bảng biểu**:
   - So sánh phương pháp
   - Kết quả thực nghiệm
   - Confusion matrix
   - Tham số tối ưu

---

## Checklist Trước Khi Bắt Đầu

- [ ] Đã chạy hệ thống với tất cả video test
- [ ] Đã có kết quả trong `code/data/output/`
- [ ] Đã đọc tất cả tài liệu trong `documentation/` và `knowledge-base/`
- [ ] Đã chuẩn bị danh sách screenshots cần chụp
- [ ] Đã hiểu rõ cấu trúc 3 chương

---

## Danh Sách Script Con

- `req-1-intro.md`: Tạo phần mở đầu
- `req-1-chapter1.md`: Tạo Chương 1 - Cơ Sở Lý Thuyết
- `req-1-chapter2.md`: Tạo Chương 2 - Cơ Sở Thực Hành
- `req-1-chapter3.md`: Tạo Chương 3 - Kết Luận
- `req-1-references.md`: Tạo tài liệu tham khảo

---

**Ngày tạo**: 28 Tháng 11, 2025
**Phiên bản**: 2.0 (Chia nhỏ để tránh token limit)
**Người thực hiện**: Claude Code Assistant
