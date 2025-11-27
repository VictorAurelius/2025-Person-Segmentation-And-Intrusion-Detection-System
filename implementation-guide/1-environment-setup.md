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

## 3. Lựa Chọn Môi Trường: Virtual Environment vs WSL

Có hai cách chính để setup môi trường phát triển:

### So Sánh

| Tiêu chí | Virtual Environment | WSL |
|----------|-------------------|-----|
| **Isolation** | ✅ Hoàn toàn cô lập | ⚠️ Chia sẻ với system |
| **Setup** | Phức tạp hơn (activate/deactivate) | Đơn giản hơn |
| **Performance** | Tốt | Rất tốt (native Linux) |
| **Disk Space** | Tốn nhiều (mỗi venv ~500MB) | Tiết kiệm |
| **Multi-project** | ✅ Lý tưởng | ⚠️ Có thể conflict |
| **Windows** | ✅ Native | ✅ Qua WSL |

### Khuyến Nghị

- **Dùng Virtual Environment nếu:**
  - Bạn làm việc với nhiều Python projects
  - Cần isolation hoàn toàn giữa các projects
  - Muốn dễ dàng share môi trường với người khác

- **Dùng WSL nếu:**
  - Chỉ làm việc với project này
  - Muốn setup nhanh, đơn giản
  - Thích Linux commands
  - Cần performance tốt hơn cho OpenCV

---

## 4. Setup với Virtual Environment

### Tạo Virtual Environment

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

## 5. Setup với WSL (Windows Subsystem for Linux)

### Cài Đặt WSL (nếu chưa có)

#### Bước 1: Enable WSL
```powershell
# Chạy PowerShell as Administrator
wsl --install
```

Hoặc thủ công:
```powershell
# Enable WSL feature
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart

# Enable Virtual Machine Platform
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# Restart máy
```

#### Bước 2: Cài Ubuntu từ Microsoft Store
1. Mở Microsoft Store
2. Tìm "Ubuntu" (khuyến nghị Ubuntu 22.04 LTS)
3. Click "Get" để cài đặt
4. Launch Ubuntu và tạo username/password

#### Bước 3: Kiểm tra WSL version
```bash
wsl --list --verbose
```

### Setup Python trong WSL

```bash
# Cập nhật package manager
sudo apt update && sudo apt upgrade -y

# Cài Python và pip
sudo apt install -y python3 python3-pip

# Verify
python3 --version
pip3 --version
```

### Navigate đến Project Directory

Windows drives được mount tại `/mnt/`:

```bash
# Ví dụ: F:\nam4\project\code -> /mnt/f/nam4/project/code
cd /mnt/f/nam4/2025-Image-Processing-Assignment/2025-Person-Segmentation-And-Intrusion-Detection-System/2025-Person-Segmentation-And-Intrusion-Detection-System/code

# Hoặc tạo symlink cho tiện
ln -s /mnt/f/nam4/2025-Image-Processing-Assignment/2025-Person-Segmentation-And-Intrusion-Detection-System/2025-Person-Segmentation-And-Intrusion-Detection-System ~/project
cd ~/project/code
```

### Cài Dependencies trong WSL

```bash
# Cài với --user flag (recommended)
pip3 install --user -r requirements.txt
```

**⚠️ Nếu gặp lỗi `externally-managed-environment`:**

Python 3.11+ trên Ubuntu/Debian block pip install để bảo vệ system. Giải pháp:

```bash
# Remove restriction file (cần sudo 1 lần)
sudo rm /usr/lib/python3.*/EXTERNALLY-MANAGED

# Sau đó cài bình thường
pip3 install --user -r requirements.txt
```

**Lưu ý:** Đây là cách đúng để dùng WSL. Nếu phải dùng venv trong WSL thì mất hết ý nghĩa của WSL rồi!

### Verify Installation

```bash
# Test imports
python3 -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
python3 -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python3 -c "import cv2, numpy, yaml, scipy; print('All imports successful!')"
```

### Chạy Application từ WSL

```bash
# Chạy main
python3 src/main.py

# Với options
python3 src/main.py --source data/input/video.mp4
python3 src/main.py --debug
```

### Lưu Ý khi dùng WSL

**✅ Ưu điểm:**
- Không cần activate/deactivate venv
- Performance tốt hơn (native Linux)
- Dễ cài đặt system packages
- Linux commands quen thuộc

**⚠️ Hạn chế:**
- GUI display có thể cần setup thêm (X server)
- File permissions khác Windows
- Path separators khác (/ thay vì \)

### Setup X Server cho GUI (Optional)

Để hiển thị OpenCV windows trong WSL:

#### Option 1: VcXsrv (Recommended)
```bash
# 1. Download và cài VcXsrv: https://sourceforge.net/projects/vcxsrv/
# 2. Chạy XLaunch với settings:
#    - Display number: 0
#    - Disable access control: ✅

# 3. Trong WSL, set DISPLAY
echo "export DISPLAY=:0" >> ~/.bashrc
source ~/.bashrc

# 4. Test
python3 -c "import cv2; cv2.imshow('test', cv2.imread('image.jpg')); cv2.waitKey(0)"
```

#### Option 2: WSLg (Windows 11)
WSL 2 trên Windows 11 đã có built-in GUI support:
```bash
# Không cần setup gì, chỉ cần update WSL
wsl --update
```

### Troubleshooting WSL

#### Lỗi: Cannot open display
```bash
# Check DISPLAY variable
echo $DISPLAY

# Set lại
export DISPLAY=:0

# Kiểm tra X server đang chạy (Windows)
```

#### Lỗi: Permission denied khi cài pip
```bash
# Dùng --user flag
pip3 install --user -r requirements.txt

# Hoặc cài pip cho user
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py --user
```

#### Lỗi: ModuleNotFoundError sau khi cài
```bash
# Check xem package được cài ở đâu
pip3 show opencv-python

# Add vào PATH nếu cần
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

---

## 6. Cài Đặt Dependencies (cho Virtual Environment)

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

## 7. Cài Đặt IDE (Tùy chọn)

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

## 8. Hỗ Trợ GPU (Tùy chọn)

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

## 9. Xử Lý Sự Cố

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

## 10. Checklist Xác Nhận

### Cho Virtual Environment:
- [ ] Python 3.8+ đã được cài đặt
- [ ] Virtual environment đã được tạo và kích hoạt
- [ ] Tất cả dependencies đã cài thành công
- [ ] OpenCV import không có lỗi
- [ ] NumPy import không có lỗi
- [ ] Có thể chạy `python src/main.py --help`

### Cho WSL:
- [ ] WSL đã được cài đặt và cấu hình
- [ ] Ubuntu/Linux distribution đã được setup
- [ ] Python 3.8+ và pip đã được cài trong WSL
- [ ] Tất cả dependencies đã cài thành công
- [ ] OpenCV import không có lỗi (test với `python3 -c "import cv2"`)
- [ ] Có thể navigate đến project directory từ WSL
- [ ] (Optional) X server được cấu hình cho GUI display

---

## 11. Kiểm Tra Cấu Trúc Thư Mục

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

## 12. Các Bước Tiếp Theo

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
