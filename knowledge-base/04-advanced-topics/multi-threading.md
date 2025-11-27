# Multi-threading (Đa Luồng) và Parallel Processing (Xử Lý Song Song)

## 1. Tại Sao Cần Xử Lý Song Song?

### A. Lợi Ích

- **Xử Lý Nhanh Hơn**: Sử dụng nhiều lõi CPU
- **Phản Hồi Tốt Hơn**: Tách biệt I/O và xử lý
- **Throughput (Thông Lượng) Cao Hơn**: Xử lý nhiều frames đồng thời

### B. Python GIL (Global Interpreter Lock)

**Thách Thức:** Python GIL ngăn chặn việc thực thi song song thực sự của Python bytecode.

**Giải Pháp:**
- **Threading (Đa luồng)**: Tốt cho I/O-bound tasks (video capture, file I/O)
- **Multiprocessing (Đa tiến trình)**: Tốt cho CPU-bound tasks (image processing)
- **NumPy/OpenCV**: Giải phóng GIL trong các thao tác C

---

## 2. Threading (Đa Luồng) Cho Video Capture

### A. Threaded Capture Cơ Bản

```python
import threading
import cv2
import queue

class VideoCapture:
    """Threaded video capture"""

    def __init__(self, src):
        self.src = src
        self.cap = cv2.VideoCapture(src)
        self.q = queue.Queue(maxsize=128)
        self.stopped = False

    def start(self):
        """Khởi động capture thread"""
        t = threading.Thread(target=self._capture, args=())
        t.daemon = True
        t.start()
        return self

    def _capture(self):
        """Vòng lặp capture"""
        while not self.stopped:
            if not self.q.full():
                ret, frame = self.cap.read()

                if not ret:
                    self.stopped = True
                    break

                self.q.put(frame)
            else:
                # Queue đầy, đợi
                threading.Event().wait(0.01)

    def read(self):
        """Đọc frame từ queue"""
        return self.q.get()

    def stop(self):
        """Dừng capture"""
        self.stopped = True
        self.cap.release()


# Sử dụng
cap = VideoCapture('video.mp4').start()

while True:
    frame = cap.read()

    # Xử lý frame (không cần đợi I/O!)
    processed = process_frame(frame)

    cv2.imshow('Frame', processed)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.stop()
cv2.destroyAllWindows()
```

### B. Cải Thiện FPS

```python
import time

# Capture thông thường
cap = cv2.VideoCapture('video.mp4')
start = time.time()
frames = 0

while frames < 300:
    ret, frame = cap.read()
    if not ret:
        break

    processed = process_frame(frame)
    frames += 1

regular_fps = frames / (time.time() - start)
cap.release()

# Threaded capture
cap = VideoCapture('video.mp4').start()
start = time.time()
frames = 0

while frames < 300:
    frame = cap.read()
    processed = process_frame(frame)
    frames += 1

threaded_fps = frames / (time.time() - start)
cap.stop()

print(f"Thông thường: {regular_fps:.1f} FPS")
print(f"Threaded: {threaded_fps:.1f} FPS")
print(f"Cải thiện: {(threaded_fps / regular_fps - 1) * 100:.1f}%")

# Kết quả điển hình: cải thiện 20-40%
```

---

## 3. Producer-Consumer Pattern (Mẫu Sản Xuất-Tiêu Thụ)

### A. Pipeline Hoàn Chỉnh

```python
import threading
import queue
import time

class VideoPipeline:
    """Multi-threaded video processing pipeline"""

    def __init__(self, src, buffer_size=128):
        self.src = src
        self.stopped = False

        # Queues
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.processed_queue = queue.Queue(maxsize=buffer_size)

    def start(self):
        """Khởi động tất cả threads"""
        # Capture thread
        t1 = threading.Thread(target=self._capture)
        t1.daemon = True
        t1.start()

        # Processing threads (nhiều workers)
        for _ in range(2):
            t = threading.Thread(target=self._process)
            t.daemon = True
            t.start()

        return self

    def _capture(self):
        """Capture frames"""
        cap = cv2.VideoCapture(self.src)

        while not self.stopped:
            ret, frame = cap.read()

            if not ret:
                self.stopped = True
                break

            # Đưa frame vào queue (chặn nếu đầy)
            self.frame_queue.put(frame)

        cap.release()

    def _process(self):
        """Xử lý frames"""
        while not self.stopped:
            try:
                # Lấy frame (với timeout)
                frame = self.frame_queue.get(timeout=1)

                # Xử lý
                processed = process_frame(frame)

                # Đưa kết quả vào queue
                self.processed_queue.put(processed)

            except queue.Empty:
                continue

    def read(self):
        """Đọc processed frame"""
        try:
            return self.processed_queue.get(timeout=1)
        except queue.Empty:
            return None

    def stop(self):
        """Dừng pipeline"""
        self.stopped = True


# Sử dụng
pipeline = VideoPipeline('video.mp4', buffer_size=128).start()

while True:
    frame = pipeline.read()

    if frame is None:
        break

    cv2.imshow('Processed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()
```

---

## 4. Multiprocessing (Đa Tiến Trình) Cho CPU-Bound Tasks

### A. Process Pool

```python
from multiprocessing import Pool, cpu_count
import cv2
import glob

def process_image(image_path):
    """Xử lý một ảnh (CPU-intensive)"""
    # Đọc
    image = cv2.imread(image_path)

    # Xử lý nặng
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Tìm contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Lưu kết quả
    output_path = image_path.replace('input', 'output')
    cv2.imwrite(output_path, binary)

    return len(contours)


# Lấy tất cả ảnh
image_paths = glob.glob('input/*.jpg')

# Xử lý tuần tự
import time
start = time.time()
results = [process_image(path) for path in image_paths]
sequential_time = time.time() - start

# Xử lý song song
start = time.time()
with Pool(processes=cpu_count()) as pool:
    results = pool.map(process_image, image_paths)
parallel_time = time.time() - start

print(f"Tuần tự: {sequential_time:.2f}s")
print(f"Song song: {parallel_time:.2f}s")
print(f"Tăng tốc: {sequential_time / parallel_time:.2f}x")
```

### B. Xử Lý Video Với Multiprocessing

```python
from multiprocessing import Process, Queue
import cv2

def capture_process(output_queue, src):
    """Capture frames trong process riêng"""
    cap = cv2.VideoCapture(src)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        output_queue.put(frame)

    output_queue.put(None)  # Sentinel
    cap.release()


def processing_process(input_queue, output_queue):
    """Xử lý frames trong process riêng"""
    while True:
        frame = input_queue.get()

        if frame is None:  # Sentinel
            output_queue.put(None)
            break

        # Xử lý nặng
        processed = heavy_processing(frame)

        output_queue.put(processed)


def display_process(input_queue):
    """Hiển thị frames trong process riêng"""
    while True:
        frame = input_queue.get()

        if frame is None:  # Sentinel
            break

        cv2.imshow('Result', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


# Tạo queues
capture_queue = Queue(maxsize=128)
processed_queue = Queue(maxsize=128)

# Tạo processes
p1 = Process(target=capture_process, args=(capture_queue, 'video.mp4'))
p2 = Process(target=processing_process, args=(capture_queue, processed_queue))
p3 = Process(target=display_process, args=(processed_queue,))

# Khởi động
p1.start()
p2.start()
p3.start()

# Đợi
p1.join()
p2.join()
p3.join()
```

---

## 5. ThreadPoolExecutor

### A. Threading Thanh Lịch

```python
from concurrent.futures import ThreadPoolExecutor
import cv2

def process_batch(frames):
    """Xử lý batch frames"""
    results = []

    for frame in frames:
        processed = process_frame(frame)
        results.append(processed)

    return results


# Capture frames
cap = cv2.VideoCapture('video.mp4')
frames = []

for _ in range(100):
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

cap.release()

# Chia thành batches
batch_size = 10
batches = [frames[i:i + batch_size] for i in range(0, len(frames), batch_size)]

# Xử lý song song
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_batch, batches))

# Làm phẳng kết quả
all_results = [r for batch in results for r in batch]

print(f"Đã xử lý {len(all_results)} frames")
```

---

## 6. Xử Lý Async Với asyncio

### A. Xử Lý Video Async

```python
import asyncio
import cv2
from concurrent.futures import ProcessPoolExecutor

async def async_process_frame(frame, executor):
    """Xử lý frame bất đồng bộ"""
    loop = asyncio.get_event_loop()

    # Chạy trong process pool
    result = await loop.run_in_executor(executor, process_frame, frame)

    return result


async def async_video_pipeline(video_path):
    """Pipeline xử lý video async"""
    cap = cv2.VideoCapture(video_path)
    executor = ProcessPoolExecutor(max_workers=4)

    tasks = []

    # Đọc frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Tạo async task
        task = asyncio.create_task(async_process_frame(frame, executor))
        tasks.append(task)

        # Giới hạn concurrent tasks
        if len(tasks) >= 10:
            # Đợi một số hoàn thành
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

            # Xử lý các task đã hoàn thành
            for task in done:
                result = task.result()
                # Hiển thị hoặc lưu kết quả

            tasks = list(pending)

    # Đợi các tasks còn lại
    results = await asyncio.gather(*tasks)

    cap.release()
    executor.shutdown()

    return results


# Chạy
results = asyncio.run(async_video_pipeline('video.mp4'))
```

---

## 7. Lock và Synchronization (Đồng Bộ Hóa)

### A. Thread-Safe Queue

```python
import threading

class ThreadSafeQueue:
    """Queue an toàn luồng với lock"""

    def __init__(self):
        self.queue = []
        self.lock = threading.Lock()

    def put(self, item):
        """Thêm item vào queue"""
        with self.lock:
            self.queue.append(item)

    def get(self):
        """Lấy item từ queue"""
        with self.lock:
            if len(self.queue) > 0:
                return self.queue.pop(0)
            return None

    def size(self):
        """Lấy kích thước queue"""
        with self.lock:
            return len(self.queue)
```

### B. Shared Counter

```python
import threading

class SharedCounter:
    """Bộ đếm an toàn luồng"""

    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()

    def increment(self):
        """Tăng bộ đếm"""
        with self.lock:
            self.value += 1

    def get(self):
        """Lấy giá trị hiện tại"""
        with self.lock:
            return self.value


# Sử dụng
counter = SharedCounter()

def worker():
    for _ in range(1000):
        counter.increment()

# Khởi động threads
threads = []
for _ in range(10):
    t = threading.Thread(target=worker)
    t.start()
    threads.append(t)

# Đợi
for t in threads:
    t.join()

print(f"Kết quả cuối: {counter.get()}")  # Phải là 10000
```

---

## 8. Ví Dụ Thực Tế

### A. Hệ Thống Multi-Threaded Hoàn Chỉnh

```python
import threading
import queue
import cv2
import time

class MultiThreadedVideoProcessor:
    """Hệ thống xử lý video multi-threaded hoàn chỉnh"""

    def __init__(self, video_source, num_workers=2):
        self.video_source = video_source
        self.num_workers = num_workers

        # Queues
        self.frame_queue = queue.Queue(maxsize=128)
        self.result_queue = queue.Queue(maxsize=128)

        # Điều khiển
        self.stopped = False

        # Thống kê
        self.frames_captured = 0
        self.frames_processed = 0
        self.start_time = None

        # Lock
        self.lock = threading.Lock()

    def start(self):
        """Khởi động tất cả threads"""
        self.start_time = time.time()

        # Capture thread
        t_capture = threading.Thread(target=self._capture_thread)
        t_capture.daemon = True
        t_capture.start()

        # Worker threads
        for i in range(self.num_workers):
            t_worker = threading.Thread(target=self._worker_thread, args=(i,))
            t_worker.daemon = True
            t_worker.start()

        return self

    def _capture_thread(self):
        """Capture frames"""
        print("[INFO] Đang khởi động capture thread")
        cap = cv2.VideoCapture(self.video_source)

        while not self.stopped and cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                self.stopped = True
                break

            # Đưa frame vào queue
            self.frame_queue.put(frame)

            with self.lock:
                self.frames_captured += 1

        cap.release()
        print("[INFO] Capture thread đã dừng")

    def _worker_thread(self, worker_id):
        """Xử lý frames"""
        print(f"[INFO] Đang khởi động worker {worker_id}")

        while not self.stopped:
            try:
                # Lấy frame
                frame = self.frame_queue.get(timeout=1)

                # Xử lý
                result = self._process_frame(frame)

                # Đưa kết quả vào queue
                self.result_queue.put(result)

                with self.lock:
                    self.frames_processed += 1

            except queue.Empty:
                continue

        print(f"[INFO] Worker {worker_id} đã dừng")

    def _process_frame(self, frame):
        """Xử lý một frame"""
        # Xử lý nặng
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        return edges

    def read(self):
        """Đọc processed frame"""
        try:
            return self.result_queue.get(timeout=1)
        except queue.Empty:
            return None

    def get_stats(self):
        """Lấy thống kê xử lý"""
        with self.lock:
            elapsed = time.time() - self.start_time
            capture_fps = self.frames_captured / elapsed if elapsed > 0 else 0
            process_fps = self.frames_processed / elapsed if elapsed > 0 else 0

            return {
                'captured': self.frames_captured,
                'processed': self.frames_processed,
                'capture_fps': capture_fps,
                'process_fps': process_fps,
                'frame_queue': self.frame_queue.qsize(),
                'result_queue': self.result_queue.qsize()
            }

    def stop(self):
        """Dừng xử lý"""
        self.stopped = True


# Sử dụng
processor = MultiThreadedVideoProcessor('video.mp4', num_workers=4).start()

frame_count = 0

while True:
    frame = processor.read()

    if frame is None:
        if processor.stopped:
            break
        continue

    frame_count += 1

    # Hiển thị
    cv2.imshow('Result', frame)

    # In thống kê mỗi 30 frames
    if frame_count % 30 == 0:
        stats = processor.get_stats()
        print(f"[STATS] Captured: {stats['captured']}, "
              f"Processed: {stats['processed']}, "
              f"Capture FPS: {stats['capture_fps']:.1f}, "
              f"Process FPS: {stats['process_fps']:.1f}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

processor.stop()
cv2.destroyAllWindows()

# Thống kê cuối
final_stats = processor.get_stats()
print("\n[THỐNG KÊ CUỐI]")
print(f"Tổng số Captured: {final_stats['captured']}")
print(f"Tổng số Processed: {final_stats['processed']}")
print(f"Capture FPS trung bình: {final_stats['capture_fps']:.1f}")
print(f"Process FPS trung bình: {final_stats['process_fps']:.1f}")
```

---

## 9. Best Practices (Thực Hành Tốt)

### A. Số Lượng Threads

```python
import os

# Quy tắc chung
num_io_threads = 2  # Cho I/O (capture, display)
num_cpu_threads = os.cpu_count()  # Cho CPU-bound (processing)

# Không vượt quá
max_threads = os.cpu_count() * 2
```

### B. Kích Thước Queue

```python
# Quá nhỏ: Threads đợi thường xuyên
queue_size = 10  # Quá nhỏ

# Quá lớn: Sử dụng bộ nhớ
queue_size = 10000  # Quá lớn

# Cân bằng tốt
queue_size = 128  # Tốt cho hầu hết trường hợp
```

### C. Xử Lý Lỗi

```python
def safe_worker():
    """Worker với xử lý lỗi"""
    while not stopped:
        try:
            frame = frame_queue.get(timeout=1)
            result = process_frame(frame)
            result_queue.put(result)

        except queue.Empty:
            continue

        except Exception as e:
            print(f"[LỖI] Ngoại lệ Worker: {e}")
            # Ghi log lỗi, tiếp tục hoặc dừng
```

---

## 10. Giám Sát Hiệu Năng

### A. Profiling

```python
import cProfile
import pstats

def profile_pipeline():
    """Profile multi-threaded pipeline"""
    profiler = cProfile.Profile()
    profiler.enable()

    # Chạy pipeline
    processor = MultiThreadedVideoProcessor('video.mp4').start()

    for _ in range(300):
        frame = processor.read()
        if frame is None:
            break

    processor.stop()

    profiler.disable()

    # In thống kê
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
```

---

**Ngày tạo**: Tháng 1/2025
