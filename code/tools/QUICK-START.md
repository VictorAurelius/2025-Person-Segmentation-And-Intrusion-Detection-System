# Quick Start - Lighting Variants

Hướng dẫn nhanh để tạo và test các điều kiện chiếu sáng khác nhau.

## TL;DR - Chạy ngay

```bash
cd code

# Bước 1: Tạo video variants (low-light + night)
./tools/generate_lighting_variants.sh

# Bước 2: Test tất cả configs
./tools/test_all_lighting.sh
```

## Chi tiết từng bước

### 1. Tạo video với điều kiện sáng khác nhau

**Cách 1: Dùng script tự động (khuyến nghị)**
```bash
cd code
./tools/generate_lighting_variants.sh input-01
```

**Cách 2: Chạy manual**
```bash
cd code

# Low-light
python tools/lighting_simulator.py \
  --input data/input/input-01.mp4 \
  --mode low-light \
  --comparison

# Night
python tools/lighting_simulator.py \
  --input data/input/input-01.mp4 \
  --mode night \
  --comparison
```

### 2. Kiểm tra kết quả

Xem comparison images:
- `data/input/input-01-low-light_comparison.jpg`
- `data/input/input-01-night_comparison.jpg`

### 3. Chạy hệ thống

**Test từng config riêng lẻ:**

```bash
cd code

# Normal (video gốc)
python src/main.py --config config/config-normal.yaml

# Low-light
python src/main.py --config config/config-lowlight.yaml

# Night
python src/main.py --config config/config-night.yaml
```

**Hoặc test tất cả:**
```bash
cd code
./tools/test_all_lighting.sh
```

## Kết quả mong đợi

### Video outputs
```
data/output/
├── input-01/
│   └── result-normal.mp4
├── input-01-lowlight/
│   └── result-lowlight.mp4
└── input-01-night/
    └── result-night.mp4
```

### Alert logs
```
data/output/
├── input-01/alerts.log
├── input-01-lowlight/alerts.log
└── input-01-night/alerts.log
```

## Config Files

| Điều kiện | Video Input | Config File | Output |
|-----------|-------------|-------------|--------|
| Normal | input-01.mp4 | config-normal.yaml | result-normal.mp4 |
| Low-light | input-01-low-light.mp4 | config-lowlight.yaml | result-lowlight.mp4 |
| Night | input-01-night.mp4 | config-night.yaml | result-night.mp4 |

## Tham số quan trọng

### Normal (Baseline)
- Brightness: 1.0 (100%)
- Noise: 0
- Motion threshold: 20
- Block size: 11

### Low-light
- Brightness: 0.45 (45%)
- Noise: 8
- Motion threshold: 16 ↓
- Block size: 15 ↑
- Min object area: 1200 ↓

### Night
- Brightness: 0.20 (20%)
- Noise: 15
- Night-vision: Enabled
- Motion threshold: 12 ↓
- Block size: 19 ↑
- Min object area: 1000 ↓

## Troubleshooting

**Video quá tối:**
```bash
# Tăng brightness
python tools/lighting_simulator.py \
  --input data/input/input-01.mp4 \
  --mode low-light \
  --brightness 0.55
```

**Quá nhiều false alarms:**
- Tăng `min_object_area` trong config
- Tăng `time_threshold`
- Tăng `morphology.iterations`

**Không detect được người:**
- Giảm `motion.threshold`
- Giảm `intrusion.min_object_area`
- Giảm `intrusion.overlap_threshold`

## Custom Parameters

Tạo điều kiện sáng custom:
```bash
python tools/lighting_simulator.py \
  --input data/input/input-01.mp4 \
  --mode low-light \
  --brightness 0.35 \
  --noise 10 \
  --output data/input/custom.mp4
```

Sau đó copy và chỉnh config:
```bash
cp config/config-lowlight.yaml config/config-custom.yaml
# Edit config-custom.yaml để point đến custom.mp4
python src/main.py --config config/config-custom.yaml
```

## So sánh kết quả

Sau khi chạy xong, so sánh:
1. **Video quality**: Xem output videos
2. **Detection accuracy**: So sánh alerts.log
3. **Performance**: Kiểm tra processing time

## Đọc thêm

- Chi tiết về tool: `tools/README-lighting-simulator.md`
- Chi tiết về config: `config/*.yaml`
- System documentation: `../documentation/`
