# Hướng Dẫn Cài Đặt Môi Trường

## 1. Yêu Cầu Hệ Thống

### Yêu Cầu Tối Thiểu
- **Hệ điều hành**: Windows 10/11, Ubuntu 20.04+, hoặc macOS 11+
- **Python**: 3.8 trở lên
- **RAM**: Tối thiểu 8GB, khuyến nghị 16GB
- **Ổ cứng**: 2GB cho code và dữ liệu
- **Bộ xử lý**: Intel Core i5 hoặc tương đương

### Yêu Cầu Khuyến Nghị
- **RAM**: 16GB trở lên
- **Bộ xử lý**: Intel Core i7 hoặc tương đương
- **GPU**: Tùy chọn (để tăng tốc với CUDA)
- **Camera**: USB webcam hoặc IP camera (để test real-time)

---

## 2. Cài Đặt Python

### Kiểm tra Python đã được cài chưa
```bash
python --version
# hoặc
python3 --version
```

### Cài đặt Python nếu chưa có

**Windows:**
1. Tải từ [python.org](https://www.python.org/downloads/)
2. Chạy file cài đặt
3. ✅ Tích chọn "Add Python to PATH"
4. Click "Install Now"

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

**macOS:**
```bash
# Dùng Homebrew
brew install python3
```

Xác nhận cài đặt thành công:
```bash
python3 --version
# Kết quả: Python 3.8.x hoặc cao hơn
```

---

## 3. Tạo Virtual Environment

### Tại sao cần Virtual Environment?
- Cô lập các dependencies của project
- Tránh xung đột với các project khác
- Dễ dàng tái tạo môi trường

### Tạo venv

```bash
# Di chuyển đến thư mục project
cd final-project/code

# Tạo virtual environment
python3 -m venv venv
```

### Kích hoạt Virtual Environment

**Linux/Mac:**
```bash
source venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```

**Xác nhận đã kích hoạt:**
Bạn sẽ thấy `(venv)` ở đầu dòng terminal:
```
(venv) user@computer:~/final-project/code$
```

---

## 4. Cài Đặt Dependencies

### Nâng cấp pip
```bash
pip install --upgrade pip
```

### Cài đặt các package cần thiết
```bash
pip install -r requirements.txt
```

Các package sẽ được cài:
- opencv-python (≥4.8.0)
- numpy (≥1.24.0)
- scikit-image (≥0.21.0)
- matplotlib (≥3.7.0)
- pyyaml (≥6.0)
- pytest (≥7.4.0)
- scipy (≥1.10.0)

### Xác nhận cài đặt thành công

```bash
# Test OpenCV
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"

# Test NumPy
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"

# Test tất cả imports
python -c "import cv2, numpy, yaml, scipy; print('Tất cả imports thành công!')"
```

---

## 5. Cài Đặt IDE (Tùy chọn)

### Visual Studio Code

1. Cài VS Code: https://code.visualstudio.com/
2. Cài extension Python:
   - Mở VS Code
   - Vào Extensions (Ctrl+Shift+X)
   - Tìm "Python"
   - Cài "Python" by Microsoft

3. Chọn Python interpreter:
   - Ctrl+Shift+P
   - Gõ "Python: Select Interpreter"
   - Chọn `./venv/bin/python`

4. Extensions khuyến nghị:
   - Python
   - Pylance
   - Python Docstring Generator

### PyCharm

1. Cài PyCharm: https://www.jetbrains.com/pycharm/
2. Mở thư mục project
3. Cấu hình interpreter:
   - File → Settings → Project → Python Interpreter
   - Add Interpreter → Existing Environment
   - Chọn `venv/bin/python`

### Jupyter Notebook (để thử nghiệm)

```bash
pip install jupyter
jupyter notebook
```

---

## 6. Hỗ Trợ GPU (Tùy chọn)

Để xử lý nhanh hơn với CUDA:

### Cài CUDA Toolkit
1. Tải từ: https://developer.nvidia.com/cuda-downloads
2. Làm theo hướng dẫn cài đặt cho hệ điều hành của bạn

### Cài OpenCV với CUDA
```bash
pip uninstall opencv-python
pip install opencv-contrib-python
```

### Xác nhận GPU có sẵn
```python
import cv2
print(cv2.cuda.getCudaEnabledDeviceCount())
# Nếu > 0 thì GPU khả dụng
```

---

## 7. Xử Lý Sự Cố

### Lỗi: Không tìm thấy lệnh pip
**Giải pháp:**
```bash
# Dùng python -m pip thay thế
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### Lỗi: Permission denied
**Giải pháp (Linux/Mac):**
```bash
# Dùng flag --user
pip install --user -r requirements.txt
```

### Lỗi: Virtual environment không kích hoạt được (Windows PowerShell)
**Giải pháp:**
```powershell
# Cho phép chạy script
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Lỗi: ImportError cho cv2
**Giải pháp:**
```bash
pip uninstall opencv-python
pip install opencv-python-headless
```

### Lỗi: Thiếu build tools (Windows)
**Giải pháp:**
Cài Visual C++ Build Tools:
https://visualstudio.microsoft.com/visual-cpp-build-tools/

---

## 8. Checklist Xác Nhận

- [ ] Python 3.8+ đã được cài đặt
- [ ] Virtual environment đã được tạo và kích hoạt
- [ ] Tất cả dependencies đã cài thành công
- [ ] OpenCV import không có lỗi
- [ ] NumPy import không có lỗi
- [ ] Có thể chạy `python src/main.py --help`

---

## 9. Kiểm Tra Cấu Trúc Thư Mục

Xác nhận cấu trúc thư mục của bạn:
```bash
cd final-project
tree -L 2
```

Kết quả mong đợi:
```
final-project/
├── code/
│   ├── src/
│   ├── config/
│   ├── data/
│   ├── tools/
│   ├── venv/          ← Virtual environment
│   └── requirements.txt
├── documentation/
├── implementation-guide/
└── knowledge-base/
```

---

## 10. Các Bước Tiếp Theo

✅ Môi trường đã sẵn sàng!

Tiếp theo: [2. Chuẩn Bị Dữ Liệu](2-data-preparation.md)

---

## Tham Khảo Nhanh Các Lệnh

```bash
# Kích hoạt venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Tắt venv
deactivate

# Cài dependencies
pip install -r requirements.txt

# Cập nhật dependencies
pip install --upgrade -r requirements.txt

# Liệt kê các package đã cài
pip list

# Kiểm tra đường dẫn Python
which python  # Linux/Mac
where python  # Windows
```

---

**Ngày tạo**: Tháng 1/2025
**Phiên bản**: 1.0
