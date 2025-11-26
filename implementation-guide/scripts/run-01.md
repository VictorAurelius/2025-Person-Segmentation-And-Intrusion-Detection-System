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