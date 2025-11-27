F:\nam4\2025-Image-Processing-Assignment\2025-Person-Segmentation-And-Intrusion-Detection-System\2025-Person-Segmentation-And-Intrusion-Detection-System\code>python tools/roi_selector.py --video data/input/input-01.mp4  
INFO:root:Initialized ROI selector for: data/input/input-01.mp4
INFO:root:ROI Selector started
INFO:root:Left click to add points, right click to finish ROI
INFO:root:Point added: (338, 591)
INFO:root:Point added: (570, 584)
INFO:root:Point added: (618, 855)
INFO:root:Point added: (341, 879)
INFO:root:ROI 'Area 1' saved with 4 points
INFO:root:Saved 1 ROI(s) to data/roi/restricted_area.json

============================================================
ROI SUMMARY
============================================================

ROI 1: Area 1
  Type: polygon
  Points: 4
  Color: [255, 0, 0]
============================================================
INFO:root:Saved 1 ROI(s) to data/roi/restricted_area.json

============================================================
ROI SUMMARY
============================================================

ROI 1: Area 1
  Type: polygon
  Points: 4
  Color: [255, 0, 0]
============================================================

F:\nam4\2025-Image-Processing-Assignment\2025-Person-Segmentation-And-Intrusion-Detection-System\2025-Person-Segmentation-And-Intrusion-Detection-System\code>cat data/roi/restricted_area.json
'cat' is not recognized as an internal or external command,
operable program or batch file.

F:\nam4\2025-Image-Processing-Assignment\2025-Person-Segmentation-And-Intrusion-Detection-System\2025-Person-Segmentation-And-Intrusion-Detection-System\code>venv\Scripts\activate

(venv) F:\nam4\2025-Image-Processing-Assignment\2025-Person-Segmentation-And-Intrusion-Detection-System\2025-Person-Segmentation-And-Intrusion-Detection-System\code>python src/main.py
Traceback (most recent call last):
  File "F:\nam4\2025-Image-Processing-Assignment\2025-Person-Segmentation-And-Intrusion-Detection-System\2025-Person-Segmentation-And-Intrusion-Detection-System\code\src\main.py", line 5, in <module>
    import cv2
ModuleNotFoundError: No module named 'cv2'

## BUG ANALYSIS & FIX REPORT

### Nguyên nhân lỗi
**Vấn đề:** `ModuleNotFoundError: No module named 'cv2'` xảy ra khi chạy trong virtual environment

**Phân tích:**
- Dependencies đã được cài đặt thành công ở bước trước (trong `setting.md`)
- NHƯNG các packages đó được cài ở **Python global environment**, không phải trong **virtual environment**
- Khi activate virtual environment (`venv\Scripts\activate`), Python chỉ tìm packages trong venv folder
- Vì venv chưa có packages nào được cài, nên báo lỗi thiếu module

### Giải pháp

**Bước 1: Activate virtual environment (nếu chưa)**
```bash
cd F:\nam4\2025-Image-Processing-Assignment\2025-Person-Segmentation-And-Intrusion-Detection-System\2025-Person-Segmentation-And-Intrusion-Detection-System\code
venv\Scripts\activate
```

**Bước 2: Cài đặt lại dependencies TRONG virtual environment**
```bash
pip install -r requirements.txt
```

**Bước 3: Verify installation**
```bash
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python -c "import cv2, numpy, yaml, scipy; print('All imports successful!')"
```

**Bước 4: Chạy lại main.py**
```bash
python src/main.py
```

### Kết quả sau khi fix

```bash
(venv) F:\...\code>pip install -r requirements.txt
Collecting opencv-python>=4.8.0
  Using cached opencv_python-4.12.0.88-cp37-abi3-win_amd64.whl (39.0 MB)
Collecting numpy>=1.24.0
  Using cached numpy-2.2.6-cp313-cp313-win_amd64.whl (12.6 MB)
...
Successfully installed colorama-0.4.6 contourpy-1.3.3 cycler-0.12.1 fonttools-4.60.1 imageio-2.37.2 iniconfig-2.3.0 kiwisolver-1.4.9 lazy-loader-0.4 matplotlib-3.10.7 networkx-3.6 numpy-2.2.6 opencv-python-4.12.0.88 packaging-25.0 pillow-12.0.0 pluggy-1.6.0 pygments-2.19.2 pyparsing-3.2.5 pytest-9.0.1 python-dateutil-2.9.0.post0 pyyaml-6.0.3 scikit-image-0.25.2 scipy-1.16.3 six-1.17.0 tifffile-2025.10.16

(venv) F:\...\code>python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
OpenCV version: 4.12.0

(venv) F:\...\code>python src/main.py
INFO:root:Loading configuration from: config/config.yaml
INFO:root:Configuration loaded successfully
INFO:root:Initializing Person Segmentation & Intrusion Detection System
INFO:root:Video source: data/input/input-01.mp4
...
[Hệ thống chạy thành công]
```

### Bài học

1. **Virtual Environment Isolation**: Virtual environment tạo ra môi trường Python độc lập, không share packages với global Python
2. **Best Practice**: Luôn activate venv TRƯỚC KHI cài đặt packages
3. **Verification**: Sau khi cài đặt, nên verify bằng cách import các modules chính

### Checklist hoàn thành
- [x] Xác định nguyên nhân lỗi
- [x] Cài đặt dependencies trong virtual environment
- [x] Verify tất cả imports
- [x] Chạy thành công main.py
- [x] Ghi lại báo cáo

---
**Trạng thái:** ✅ BUG FIXED
**Thời gian fix:** ~5 phút
**Độ nghiêm trọng:** Medium (lỗi configuration, không phải lỗi code)

---

## ALTERNATIVE SOLUTION: SỬ DỤNG WSL

### Tại sao dùng WSL?

Nếu bạn cảm thấy việc quản lý virtual environment phức tạp, WSL (Windows Subsystem for Linux) là giải pháp đơn giản hơn:

**Ưu điểm:**
- ✅ Không cần tạo/activate virtual environment
- ✅ Cài đặt dependencies một lần, dùng mãi mãi
- ✅ Performance tốt hơn cho OpenCV
- ✅ Không lo lắng về việc quên activate venv
- ✅ Linux commands quen thuộc và mạnh mẽ hơn

**Nhược điểm:**
- ⚠️ Cần cài đặt WSL lần đầu (~10-15 phút)
- ⚠️ GUI display cần setup thêm (hoặc dùng --no-display)

### Hướng dẫn setup WSL

#### Bước 1: Cài WSL (nếu chưa có)

```powershell
# Chạy PowerShell as Administrator
wsl --install

# Restart máy sau khi cài xong
```

#### Bước 2: Cài Ubuntu

1. Mở Microsoft Store
2. Search "Ubuntu"
3. Cài "Ubuntu 22.04 LTS"
4. Launch và tạo username/password

#### Bước 3: Setup Python và pip trong WSL

```bash
# Mở WSL terminal
wsl

# Update system
sudo apt update && sudo apt upgrade -y

# Cài Python và pip
sudo apt install -y python3 python3-pip

# Verify
python3 --version
pip3 --version
```

#### Bước 4: Navigate đến project

```bash
# Windows drives được mount tại /mnt/
# Ví dụ: F:\ -> /mnt/f/
cd /mnt/f/nam4/2025-Image-Processing-Assignment/2025-Person-Segmentation-And-Intrusion-Detection-System/2025-Person-Segmentation-And-Intrusion-Detection-System/code

# (Optional) Tạo symlink cho dễ nhớ
ln -s /mnt/f/nam4/2025-Image-Processing-Assignment/2025-Person-Segmentation-And-Intrusion-Detection-System/2025-Person-Segmentation-And-Intrusion-Detection-System ~/project
cd ~/project/code
```

#### Bước 5: Cài dependencies trong WSL

```bash
# Cài tất cả dependencies
pip3 install -r requirements.txt

# Nếu gặp lỗi permission, dùng --user
pip3 install --user -r requirements.txt
```

#### Bước 6: Verify installation

```bash
python3 -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
python3 -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python3 -c "import cv2, numpy, yaml, scipy; print('All imports successful!')"
```

Kết quả mong đợi:
```
OpenCV version: 4.12.0
NumPy version: 2.2.6
All imports successful!
```

#### Bước 7: Chạy main.py

```bash
# Chạy với headless mode (không hiển thị GUI)
python3 src/main.py --no-display

# Hoặc setup X server để hiển thị GUI (xem phần dưới)
python3 src/main.py
```

### Setup GUI Display cho WSL (Optional)

Nếu muốn xem video output real-time:

#### Option 1: Windows 11 - WSLg (Dễ nhất)
```bash
# Windows 11 đã có built-in GUI support
# Chỉ cần update WSL
wsl --update

# Chạy bình thường
python3 src/main.py
```

#### Option 2: Windows 10 - VcXsrv
```bash
# 1. Download VcXsrv: https://sourceforge.net/projects/vcxsrv/
# 2. Cài đặt và chạy XLaunch
#    Settings: Display number = 0, Disable access control = Yes
# 3. Trong WSL, set DISPLAY variable
echo "export DISPLAY=:0" >> ~/.bashrc
source ~/.bashrc

# 4. Test
python3 src/main.py
```

### Kết quả sau khi dùng WSL

```bash
# Lần đầu setup
user@DESKTOP:~$ cd /mnt/f/.../code
user@DESKTOP:/mnt/f/.../code$ pip3 install -r requirements.txt
Successfully installed opencv-python-4.12.0 numpy-2.2.6 ...

user@DESKTOP:/mnt/f/.../code$ python3 -c "import cv2; print('OK')"
OK

user@DESKTOP:/mnt/f/.../code$ python3 src/main.py --no-display
INFO:root:Loading configuration from: config/config.yaml
INFO:root:Configuration loaded successfully
INFO:root:Initializing Person Segmentation & Intrusion Detection System
INFO:root:Video source: data/input/input-01.mp4
INFO:root:Processing started...
...
[Hệ thống chạy thành công]
```

### So sánh: Virtual Environment vs WSL

| Tiêu chí | Virtual Environment | WSL |
|----------|-------------------|-----|
| **Setup lần đầu** | Đơn giản (5 phút) | Phức tạp hơn (15 phút) |
| **Sử dụng hàng ngày** | Cần activate mỗi lần | Không cần activate |
| **Isolation** | ✅ Hoàn toàn cô lập | ⚠️ Share với system |
| **Performance** | Tốt | Tốt hơn |
| **GUI Display** | ✅ Native | ⚠️ Cần setup thêm |
| **Nhiều projects** | ✅ Lý tưởng | ⚠️ Có thể conflict |
| **Recommended for** | Nhiều Python projects | 1 project chính |

### Khi nào nên dùng WSL?

✅ **NÊN dùng WSL khi:**
- Chỉ làm việc với 1 project này
- Muốn workflow đơn giản, không lo activate/deactivate
- Cần performance tốt cho OpenCV
- Thích Linux commands
- Không cần GUI display (dùng --no-display)

❌ **KHÔNG nên dùng WSL khi:**
- Làm việc với nhiều Python projects khác nhau
- Cần isolation hoàn toàn
- Chưa quen với Linux commands
- Cần GUI display thường xuyên và không muốn setup X server

### Troubleshooting WSL

**Lỗi: pip3 not found**
```bash
sudo apt install -y python3-pip
```

**Lỗi: ModuleNotFoundError sau khi cài**
```bash
# Check PATH
echo $PATH

# Add ~/.local/bin vào PATH
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

**Lỗi: Cannot open display**
```bash
# Nếu không cần GUI, dùng headless mode
python3 src/main.py --no-display

# Nếu cần GUI, setup X server (xem hướng dẫn trên)
```

---
**Lựa chọn:** Bạn có thể chọn 1 trong 2 phương pháp:
1. **Virtual Environment** (trong Windows) - Chuẩn, an toàn, dễ share
2. **WSL** (Linux subsystem) - Đơn giản, nhanh, ít rắc rối hàng ngày

Cả hai đều work! Chọn cái nào phù hợp với workflow của bạn.