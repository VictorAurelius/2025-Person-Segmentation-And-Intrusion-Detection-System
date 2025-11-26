F:\nam4\2025-Image-Processing-Assignment\2025-Person-Segmentation-And-Intrusion-Detection-System\2025-Person-Segmentation-And-Intrusion-Detection-System\code>pip install -r requirements.txt
Collecting opencv-python>=4.8.0 (from -r requirements.txt (line 1))
  Downloading opencv_python-4.12.0.88-cp37-abi3-win_amd64.whl.metadata (19 kB)
Collecting numpy>=1.24.0 (from -r requirements.txt (line 2))
  Downloading numpy-2.3.5-cp313-cp313-win_amd64.whl.metadata (60 kB)
Collecting scikit-image>=0.21.0 (from -r requirements.txt (line 3))
  Downloading scikit_image-0.25.2-cp313-cp313-win_amd64.whl.metadata (14 kB)
Collecting matplotlib>=3.7.0 (from -r requirements.txt (line 4))
  Downloading matplotlib-3.10.7-cp313-cp313-win_amd64.whl.metadata (11 kB)
Collecting pyyaml>=6.0 (from -r requirements.txt (line 5))
  Using cached pyyaml-6.0.3-cp313-cp313-win_amd64.whl.metadata (2.4 kB)
Collecting pytest>=7.4.0 (from -r requirements.txt (line 6))
  Downloading pytest-9.0.1-py3-none-any.whl.metadata (7.6 kB)
Collecting scipy>=1.10.0 (from -r requirements.txt (line 7))
  Downloading scipy-1.16.3-cp313-cp313-win_amd64.whl.metadata (60 kB)
Collecting numpy>=1.24.0 (from -r requirements.txt (line 2))
  Downloading numpy-2.2.6-cp313-cp313-win_amd64.whl.metadata (60 kB)
Collecting networkx>=3.0 (from scikit-image>=0.21.0->-r requirements.txt (line 3))
  Downloading networkx-3.6-py3-none-any.whl.metadata (6.8 kB)
Collecting pillow>=10.1 (from scikit-image>=0.21.0->-r requirements.txt (line 3))
  Downloading pillow-12.0.0-cp313-cp313-win_amd64.whl.metadata (9.0 kB)
Collecting imageio!=2.35.0,>=2.33 (from scikit-image>=0.21.0->-r requirements.txt (line 3))
  Downloading imageio-2.37.2-py3-none-any.whl.metadata (9.7 kB)
Collecting tifffile>=2022.8.12 (from scikit-image>=0.21.0->-r requirements.txt (line 3))
  Downloading tifffile-2025.10.16-py3-none-any.whl.metadata (31 kB)
Collecting packaging>=21 (from scikit-image>=0.21.0->-r requirements.txt (line 3))
  Using cached packaging-25.0-py3-none-any.whl.metadata (3.3 kB)
Collecting lazy-loader>=0.4 (from scikit-image>=0.21.0->-r requirements.txt (line 3))
  Using cached lazy_loader-0.4-py3-none-any.whl.metadata (7.6 kB)
Collecting contourpy>=1.0.1 (from matplotlib>=3.7.0->-r requirements.txt (line 4))
  Downloading contourpy-1.3.3-cp313-cp313-win_amd64.whl.metadata (5.5 kB)
Collecting cycler>=0.10 (from matplotlib>=3.7.0->-r requirements.txt (line 4))
  Downloading cycler-0.12.1-py3-none-any.whl.metadata (3.8 kB)
Collecting fonttools>=4.22.0 (from matplotlib>=3.7.0->-r requirements.txt (line 4))
  Downloading fonttools-4.60.1-cp313-cp313-win_amd64.whl.metadata (114 kB)
Collecting kiwisolver>=1.3.1 (from matplotlib>=3.7.0->-r requirements.txt (line 4))
  Downloading kiwisolver-1.4.9-cp313-cp313-win_amd64.whl.metadata (6.4 kB)
Collecting pyparsing>=3 (from matplotlib>=3.7.0->-r requirements.txt (line 4))
  Downloading pyparsing-3.2.5-py3-none-any.whl.metadata (5.0 kB)
Collecting python-dateutil>=2.7 (from matplotlib>=3.7.0->-r requirements.txt (line 4))
  Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl.metadata (8.4 kB)
Collecting colorama>=0.4 (from pytest>=7.4.0->-r requirements.txt (line 6))
  Using cached colorama-0.4.6-py2.py3-none-any.whl.metadata (17 kB)
Collecting iniconfig>=1.0.1 (from pytest>=7.4.0->-r requirements.txt (line 6))
  Downloading iniconfig-2.3.0-py3-none-any.whl.metadata (2.5 kB)
Collecting pluggy<2,>=1.5 (from pytest>=7.4.0->-r requirements.txt (line 6))
  Using cached pluggy-1.6.0-py3-none-any.whl.metadata (4.8 kB)
Collecting pygments>=2.7.2 (from pytest>=7.4.0->-r requirements.txt (line 6))
  Using cached pygments-2.19.2-py3-none-any.whl.metadata (2.5 kB)
Collecting six>=1.5 (from python-dateutil>=2.7->matplotlib>=3.7.0->-r requirements.txt (line 4))
  Using cached six-1.17.0-py2.py3-none-any.whl.metadata (1.7 kB)
Downloading opencv_python-4.12.0.88-cp37-abi3-win_amd64.whl (39.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 39.0/39.0 MB 2.4 MB/s  0:00:16
Downloading numpy-2.2.6-cp313-cp313-win_amd64.whl (12.6 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 12.6/12.6 MB 2.7 MB/s  0:00:04
Downloading scikit_image-0.25.2-cp313-cp313-win_amd64.whl (12.9 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 12.9/12.9 MB 3.0 MB/s  0:00:04
Downloading matplotlib-3.10.7-cp313-cp313-win_amd64.whl (8.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.1/8.1 MB 2.5 MB/s  0:00:03
Using cached pyyaml-6.0.3-cp313-cp313-win_amd64.whl (154 kB)
Downloading pytest-9.0.1-py3-none-any.whl (373 kB)
Using cached pluggy-1.6.0-py3-none-any.whl (20 kB)
Downloading scipy-1.16.3-cp313-cp313-win_amd64.whl (38.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 38.5/38.5 MB 2.8 MB/s  0:00:13
Using cached colorama-0.4.6-py2.py3-none-any.whl (25 kB)
Downloading contourpy-1.3.3-cp313-cp313-win_amd64.whl (226 kB)
Downloading cycler-0.12.1-py3-none-any.whl (8.3 kB)
Downloading fonttools-4.60.1-cp313-cp313-win_amd64.whl (2.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.3/2.3 MB 2.2 MB/s  0:00:01
Downloading imageio-2.37.2-py3-none-any.whl (317 kB)
Downloading iniconfig-2.3.0-py3-none-any.whl (7.5 kB)
Downloading kiwisolver-1.4.9-cp313-cp313-win_amd64.whl (73 kB)
Using cached lazy_loader-0.4-py3-none-any.whl (12 kB)
Downloading networkx-3.6-py3-none-any.whl (2.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/2.1 MB 2.8 MB/s  0:00:00
Using cached packaging-25.0-py3-none-any.whl (66 kB)
Downloading pillow-12.0.0-cp313-cp313-win_amd64.whl (7.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7.0/7.0 MB 2.1 MB/s  0:00:03
Using cached pygments-2.19.2-py3-none-any.whl (1.2 MB)
Downloading pyparsing-3.2.5-py3-none-any.whl (113 kB)
Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)
Using cached six-1.17.0-py2.py3-none-any.whl (11 kB)
Downloading tifffile-2025.10.16-py3-none-any.whl (231 kB)
Installing collected packages: six, pyyaml, pyparsing, pygments, pluggy, pillow, packaging, numpy, networkx, kiwisolver, iniconfig, fonttools, cycler, colorama, tifffile, scipy, python-dateutil, pytest, opencv-python, lazy-loader, imageio, contourpy, scikit-image, matplotlib
Successfully installed colorama-0.4.6 contourpy-1.3.3 cycler-0.12.1 fonttools-4.60.1 imageio-2.37.2 iniconfig-2.3.0 kiwisolver-1.4.9 lazy-loader-0.4 matplotlib-3.10.7 networkx-3.6 numpy-2.2.6 opencv-python-4.12.0.88 packaging-25.0 pillow-12.0.0 pluggy-1.6.0 pygments-2.19.2 pyparsing-3.2.5 pytest-9.0.1 python-dateutil-2.9.0.post0 pyyaml-6.0.3 scikit-image-0.25.2 scipy-1.16.3 six-1.17.0 tifffile-2025.10.16

F:\nam4\2025-Image-Processing-Assignment\2025-Person-Segmentation-And-Intrusion-Detection-System\2025-Person-Segmentation-And-Intrusion-Detection-System\code>python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
OpenCV version: 4.12.0

F:\nam4\2025-Image-Processing-Assignment\2025-Person-Segmentation-And-Intrusion-Detection-System\2025-Person-Segmentation-And-Intrusion-Detection-System\code>python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
NumPy version: 2.2.6

F:\nam4\2025-Image-Processing-Assignment\2025-Person-Segmentation-And-Intrusion-Detection-System\2025-Person-Segmentation-And-Intrusion-Detection-System\code>python -c "import cv2, numpy, yaml, scipy; print('Tất cả imports thành công!')"
Tất cả imports thành công!