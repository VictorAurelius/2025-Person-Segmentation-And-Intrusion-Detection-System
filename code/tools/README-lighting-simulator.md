# Lighting Simulator Tool

Tool để tạo các video với điều kiện chiếu sáng khác nhau (thiếu sáng, ban đêm) từ video gốc. Video tạo ra sẽ có đặc trưng của từng điều kiện sáng nhưng vẫn đủ rõ để hệ thống intrusion detection có thể hoạt động.

## Tính năng

- Tạo video với 3 chế độ chiếu sáng:
  - **Normal**: Điều kiện sáng bình thường (giữ nguyên video gốc)
  - **Low-light**: Điều kiện thiếu sáng (hoàng hôn, trời u ám, trong nhà)
  - **Night**: Điều kiện ban đêm với hiệu ứng night-vision

- Tự động điều chỉnh các thông số:
  - Brightness (độ sáng)
  - Noise level (mức độ nhiễu)
  - Night-vision effect (hiệu ứng quan sát ban đêm)

- Tạo preview so sánh original vs processed

## Cài đặt

Tool sử dụng các thư viện có sẵn trong môi trường:
```bash
pip install opencv-python numpy
```

## Sử dụng

### 1. Tạo video với preset modes

**Low-light mode:**
```bash
cd code
python tools/lighting_simulator.py --input data/input/input-01.mp4 --mode low-light
```

**Night mode:**
```bash
python tools/lighting_simulator.py --input data/input/input-01.mp4 --mode night
```

**Normal mode (copy with effects disabled):**
```bash
python tools/lighting_simulator.py --input data/input/input-01.mp4 --mode normal
```

### 2. Tạo video với custom parameters

```bash
python tools/lighting_simulator.py \
  --input data/input/input-01.mp4 \
  --mode low-light \
  --brightness 0.3 \
  --noise 12 \
  --output data/input/custom-lighting.mp4
```

### 3. Tạo comparison preview

```bash
python tools/lighting_simulator.py \
  --input data/input/input-01.mp4 \
  --mode low-light \
  --comparison
```

## Preset Parameters

| Mode      | Brightness | Noise Level | Night Vision | Description                    |
|-----------|------------|-------------|--------------|--------------------------------|
| normal    | 1.0        | 0           | No           | Điều kiện sáng bình thường     |
| low-light | 0.45       | 8           | No           | Thiếu sáng (dusk/indoor)       |
| night     | 0.20       | 15          | Yes          | Ban đêm với night-vision       |

## Workflow hoàn chỉnh cho Input-01

### Bước 1: Tạo các video variants

```bash
cd code

# Low-light version
python tools/lighting_simulator.py \
  --input data/input/input-01.mp4 \
  --mode low-light \
  --comparison

# Night version
python tools/lighting_simulator.py \
  --input data/input/input-01.mp4 \
  --mode night \
  --comparison
```

Kết quả:
- `data/input/input-01-low-light.mp4`
- `data/input/input-01-low-light_comparison.jpg`
- `data/input/input-01-night.mp4`
- `data/input/input-01-night_comparison.jpg`

### Bước 2: Chạy hệ thống với config tương ứng

**Normal lighting:**
```bash
python src/main.py --config config/config-normal.yaml
```

**Low-light:**
```bash
python src/main.py --config config/config-lowlight.yaml
```

**Night:**
```bash
python src/main.py --config config/config-night.yaml
```

## Config Files

Đã tạo sẵn 3 config files tối ưu cho từng điều kiện sáng:

### config-normal.yaml
- Video: `input-01.mp4` (original)
- Tham số chuẩn cho điều kiện sáng tốt
- Output: `data/output/input-01/result-normal.mp4`

### config-lowlight.yaml
- Video: `input-01-low-light.mp4`
- Tham số điều chỉnh:
  - Motion threshold: 16 (giảm từ 20)
  - Block size: 15 (tăng từ 11)
  - Edge thresholds: 30/100 (giảm từ 50/150)
  - Morphology iterations: 3 (tăng từ 2)
  - Min object area: 1200 (giảm từ 1500)
- Output: `data/output/input-01-lowlight/result-lowlight.mp4`

### config-night.yaml
- Video: `input-01-night.mp4`
- Tham số điều chỉnh mạnh:
  - Motion threshold: 12 (rất nhạy)
  - Block size: 19 (rất lớn)
  - Edge thresholds: 20/70 (rất thấp)
  - Morphology: kernel=9, iterations=4
  - Min object area: 1000
  - Shadow detection: disabled
- Output: `data/output/input-01-night/result-night.mp4`

## Giải thích kỹ thuật

### Low-light Simulation
1. **Brightness reduction (0.45x)**: Giảm độ sáng xuống 45% để mô phỏng thiếu ánh sáng
2. **Gaussian noise (σ=8)**: Thêm nhiễu để mô phỏng noise của sensor trong điều kiện thiếu sáng
3. Vẫn giữ đủ chi tiết để nhận diện người

### Night Simulation
1. **Brightness reduction (0.20x)**: Giảm độ sáng xuống 20%
2. **Higher noise (σ=15)**: Nhiễu cao hơn do điều kiện tối
3. **Night-vision effect**:
   - Convert to grayscale
   - CLAHE enhancement (Contrast Limited Adaptive Histogram Equalization)
   - Green tint để tạo hiệu ứng night-vision thực tế
4. Vẫn đủ rõ để hệ thống hoạt động nhưng có challenge

## Tuning Parameters

Nếu cần điều chỉnh thêm, có thể chỉnh trong config YAML:

### Motion Detection
- `threshold`: Giảm để nhạy hơn với chuyển động nhỏ
- `history`: Tăng để xây dựng background model tốt hơn trong điều kiện nhiễu

### Adaptive Thresholding
- `block_size`: Tăng cho low-contrast (phải là số lẻ)
- `C`: Tăng để compensate cho noise

### Edge Detection
- Lower thresholds cho low-light conditions

### Morphology
- `kernel_size`, `iterations`: Tăng để xử lý noise

### Intrusion Detection
- `min_object_area`: Giảm cho low-light vì object có thể bị detect nhỏ hơn
- `overlap_threshold`: Giảm để lenient hơn

## Troubleshooting

**Video quá tối, không detect được:**
- Tăng brightness factor trong lighting_simulator
- Giảm motion threshold trong config
- Giảm min_object_area

**Quá nhiều false positives:**
- Tăng min_object_area
- Tăng morphology iterations
- Tăng time_threshold

**Noise quá nhiều:**
- Giảm noise level trong lighting_simulator
- Tăng morphology operations trong config
- Tăng block_size trong adaptive thresholding

## Examples

### Tạo tất cả variants cho input-01
```bash
#!/bin/bash
cd code

# Generate low-light
python tools/lighting_simulator.py \
  --input data/input/input-01.mp4 \
  --mode low-light \
  --comparison

# Generate night
python tools/lighting_simulator.py \
  --input data/input/input-01.mp4 \
  --mode night \
  --comparison

# Test all configurations
echo "Testing normal lighting..."
python src/main.py --config config/config-normal.yaml

echo "Testing low-light..."
python src/main.py --config config/config-lowlight.yaml

echo "Testing night..."
python src/main.py --config config/config-night.yaml
```

### Tạo custom lighting condition
```bash
# Very challenging night condition
python tools/lighting_simulator.py \
  --input data/input/input-01.mp4 \
  --mode night \
  --brightness 0.15 \
  --noise 20 \
  --output data/input/input-01-extreme-night.mp4
```

## Output Structure

```
code/
├── data/
│   ├── input/
│   │   ├── input-01.mp4                    # Original
│   │   ├── input-01-low-light.mp4          # Generated
│   │   ├── input-01-low-light_comparison.jpg
│   │   ├── input-01-night.mp4              # Generated
│   │   └── input-01-night_comparison.jpg
│   └── output/
│       ├── input-01/
│       │   ├── alerts.log
│       │   └── result-normal.mp4
│       ├── input-01-lowlight/
│       │   ├── alerts.log
│       │   └── result-lowlight.mp4
│       └── input-01-night/
│           ├── alerts.log
│           └── result-night.mp4
└── config/
    ├── config-normal.yaml
    ├── config-lowlight.yaml
    └── config-night.yaml
```

## Notes

- Video variants giữ nguyên fps, resolution của video gốc
- Comparison images được lưu tự động khi dùng flag `--comparison`
- Tool có thể xử lý bất kỳ video nào, không chỉ input-01
- Parameters được tune để cân bằng giữa realistic và detectable
