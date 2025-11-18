# Multi-threading and Parallel Processing

## 1. Why Parallel Processing?

### A. Benefits

- **Faster Processing**: Utilize multiple CPU cores
- **Better Responsiveness**: Separate I/O and processing
- **Higher Throughput**: Process multiple frames simultaneously

### B. Python GIL (Global Interpreter Lock)

**Challenge:** Python GIL prevents true parallel execution của Python bytecode.

**Solutions:**
- **Threading**: Good for I/O-bound tasks (video capture, file I/O)
- **Multiprocessing**: Good for CPU-bound tasks (image processing)
- **NumPy/OpenCV**: Release GIL during C operations

---

## 2. Threading for Video Capture

### A. Basic Threaded Capture

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
        """Start capture thread"""
        t = threading.Thread(target=self._capture, args=())
        t.daemon = True
        t.start()
        return self

    def _capture(self):
        """Capture loop"""
        while not self.stopped:
            if not self.q.full():
                ret, frame = self.cap.read()

                if not ret:
                    self.stopped = True
                    break

                self.q.put(frame)
            else:
                # Queue full, wait
                threading.Event().wait(0.01)

    def read(self):
        """Read frame from queue"""
        return self.q.get()

    def stop(self):
        """Stop capture"""
        self.stopped = True
        self.cap.release()


# Usage
cap = VideoCapture('video.mp4').start()

while True:
    frame = cap.read()

    # Process frame (no waiting for I/O!)
    processed = process_frame(frame)

    cv2.imshow('Frame', processed)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.stop()
cv2.destroyAllWindows()
```

### B. FPS Improvement

```python
import time

# Regular capture
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

print(f"Regular: {regular_fps:.1f} FPS")
print(f"Threaded: {threaded_fps:.1f} FPS")
print(f"Improvement: {(threaded_fps / regular_fps - 1) * 100:.1f}%")

# Typical result: 20-40% improvement
```

---

## 3. Producer-Consumer Pattern

### A. Complete Pipeline

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
        """Start all threads"""
        # Capture thread
        t1 = threading.Thread(target=self._capture)
        t1.daemon = True
        t1.start()

        # Processing threads (multiple workers)
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

            # Put frame in queue (blocks if full)
            self.frame_queue.put(frame)

        cap.release()

    def _process(self):
        """Process frames"""
        while not self.stopped:
            try:
                # Get frame (with timeout)
                frame = self.frame_queue.get(timeout=1)

                # Process
                processed = process_frame(frame)

                # Put result
                self.processed_queue.put(processed)

            except queue.Empty:
                continue

    def read(self):
        """Read processed frame"""
        try:
            return self.processed_queue.get(timeout=1)
        except queue.Empty:
            return None

    def stop(self):
        """Stop pipeline"""
        self.stopped = True


# Usage
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

## 4. Multiprocessing for CPU-Bound Tasks

### A. Process Pool

```python
from multiprocessing import Pool, cpu_count
import cv2
import glob

def process_image(image_path):
    """Process single image (CPU-intensive)"""
    # Read
    image = cv2.imread(image_path)

    # Heavy processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Save result
    output_path = image_path.replace('input', 'output')
    cv2.imwrite(output_path, binary)

    return len(contours)


# Get all images
image_paths = glob.glob('input/*.jpg')

# Sequential processing
import time
start = time.time()
results = [process_image(path) for path in image_paths]
sequential_time = time.time() - start

# Parallel processing
start = time.time()
with Pool(processes=cpu_count()) as pool:
    results = pool.map(process_image, image_paths)
parallel_time = time.time() - start

print(f"Sequential: {sequential_time:.2f}s")
print(f"Parallel: {parallel_time:.2f}s")
print(f"Speedup: {sequential_time / parallel_time:.2f}x")
```

### B. Video Processing with Multiprocessing

```python
from multiprocessing import Process, Queue
import cv2

def capture_process(output_queue, src):
    """Capture frames in separate process"""
    cap = cv2.VideoCapture(src)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        output_queue.put(frame)

    output_queue.put(None)  # Sentinel
    cap.release()


def processing_process(input_queue, output_queue):
    """Process frames in separate process"""
    while True:
        frame = input_queue.get()

        if frame is None:  # Sentinel
            output_queue.put(None)
            break

        # Heavy processing
        processed = heavy_processing(frame)

        output_queue.put(processed)


def display_process(input_queue):
    """Display frames in separate process"""
    while True:
        frame = input_queue.get()

        if frame is None:  # Sentinel
            break

        cv2.imshow('Result', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


# Create queues
capture_queue = Queue(maxsize=128)
processed_queue = Queue(maxsize=128)

# Create processes
p1 = Process(target=capture_process, args=(capture_queue, 'video.mp4'))
p2 = Process(target=processing_process, args=(capture_queue, processed_queue))
p3 = Process(target=display_process, args=(processed_queue,))

# Start
p1.start()
p2.start()
p3.start()

# Wait
p1.join()
p2.join()
p3.join()
```

---

## 5. ThreadPoolExecutor

### A. Elegant Threading

```python
from concurrent.futures import ThreadPoolExecutor
import cv2

def process_batch(frames):
    """Process batch of frames"""
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

# Split into batches
batch_size = 10
batches = [frames[i:i + batch_size] for i in range(0, len(frames), batch_size)]

# Process in parallel
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_batch, batches))

# Flatten results
all_results = [r for batch in results for r in batch]

print(f"Processed {len(all_results)} frames")
```

---

## 6. Async Processing with asyncio

### A. Async Video Processing

```python
import asyncio
import cv2
from concurrent.futures import ProcessPoolExecutor

async def async_process_frame(frame, executor):
    """Process frame asynchronously"""
    loop = asyncio.get_event_loop()

    # Run in process pool
    result = await loop.run_in_executor(executor, process_frame, frame)

    return result


async def async_video_pipeline(video_path):
    """Async video processing pipeline"""
    cap = cv2.VideoCapture(video_path)
    executor = ProcessPoolExecutor(max_workers=4)

    tasks = []

    # Read frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Create async task
        task = asyncio.create_task(async_process_frame(frame, executor))
        tasks.append(task)

        # Limit concurrent tasks
        if len(tasks) >= 10:
            # Wait for some to complete
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

            # Process completed
            for task in done:
                result = task.result()
                # Display or save result

            tasks = list(pending)

    # Wait for remaining
    results = await asyncio.gather(*tasks)

    cap.release()
    executor.shutdown()

    return results


# Run
results = asyncio.run(async_video_pipeline('video.mp4'))
```

---

## 7. Lock and Synchronization

### A. Thread-Safe Queue

```python
import threading

class ThreadSafeQueue:
    """Thread-safe queue with lock"""

    def __init__(self):
        self.queue = []
        self.lock = threading.Lock()

    def put(self, item):
        """Add item to queue"""
        with self.lock:
            self.queue.append(item)

    def get(self):
        """Get item from queue"""
        with self.lock:
            if len(self.queue) > 0:
                return self.queue.pop(0)
            return None

    def size(self):
        """Get queue size"""
        with self.lock:
            return len(self.queue)
```

### B. Shared Counter

```python
import threading

class SharedCounter:
    """Thread-safe counter"""

    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()

    def increment(self):
        """Increment counter"""
        with self.lock:
            self.value += 1

    def get(self):
        """Get current value"""
        with self.lock:
            return self.value


# Usage
counter = SharedCounter()

def worker():
    for _ in range(1000):
        counter.increment()

# Start threads
threads = []
for _ in range(10):
    t = threading.Thread(target=worker)
    t.start()
    threads.append(t)

# Wait
for t in threads:
    t.join()

print(f"Final count: {counter.get()}")  # Should be 10000
```

---

## 8. Real-World Example

### A. Complete Multi-Threaded System

```python
import threading
import queue
import cv2
import time

class MultiThreadedVideoProcessor:
    """Complete multi-threaded video processing system"""

    def __init__(self, video_source, num_workers=2):
        self.video_source = video_source
        self.num_workers = num_workers

        # Queues
        self.frame_queue = queue.Queue(maxsize=128)
        self.result_queue = queue.Queue(maxsize=128)

        # Control
        self.stopped = False

        # Statistics
        self.frames_captured = 0
        self.frames_processed = 0
        self.start_time = None

        # Lock
        self.lock = threading.Lock()

    def start(self):
        """Start all threads"""
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
        print("[INFO] Starting capture thread")
        cap = cv2.VideoCapture(self.video_source)

        while not self.stopped and cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                self.stopped = True
                break

            # Put frame
            self.frame_queue.put(frame)

            with self.lock:
                self.frames_captured += 1

        cap.release()
        print("[INFO] Capture thread stopped")

    def _worker_thread(self, worker_id):
        """Process frames"""
        print(f"[INFO] Starting worker {worker_id}")

        while not self.stopped:
            try:
                # Get frame
                frame = self.frame_queue.get(timeout=1)

                # Process
                result = self._process_frame(frame)

                # Put result
                self.result_queue.put(result)

                with self.lock:
                    self.frames_processed += 1

            except queue.Empty:
                continue

        print(f"[INFO] Worker {worker_id} stopped")

    def _process_frame(self, frame):
        """Process single frame"""
        # Heavy processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        return edges

    def read(self):
        """Read processed frame"""
        try:
            return self.result_queue.get(timeout=1)
        except queue.Empty:
            return None

    def get_stats(self):
        """Get processing statistics"""
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
        """Stop processing"""
        self.stopped = True


# Usage
processor = MultiThreadedVideoProcessor('video.mp4', num_workers=4).start()

frame_count = 0

while True:
    frame = processor.read()

    if frame is None:
        if processor.stopped:
            break
        continue

    frame_count += 1

    # Display
    cv2.imshow('Result', frame)

    # Print stats every 30 frames
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

# Final stats
final_stats = processor.get_stats()
print("\n[FINAL STATS]")
print(f"Total Captured: {final_stats['captured']}")
print(f"Total Processed: {final_stats['processed']}")
print(f"Average Capture FPS: {final_stats['capture_fps']:.1f}")
print(f"Average Process FPS: {final_stats['process_fps']:.1f}")
```

---

## 9. Best Practices

### A. Thread Count

```python
import os

# Rule of thumb
num_io_threads = 2  # For I/O (capture, display)
num_cpu_threads = os.cpu_count()  # For CPU-bound (processing)

# Don't exceed
max_threads = os.cpu_count() * 2
```

### B. Queue Size

```python
# Too small: Threads wait frequently
queue_size = 10  # ❌ Too small

# Too large: Memory usage
queue_size = 10000  # ❌ Too large

# Good balance
queue_size = 128  # ✅ Good for most cases
```

### C. Error Handling

```python
def safe_worker():
    """Worker with error handling"""
    while not stopped:
        try:
            frame = frame_queue.get(timeout=1)
            result = process_frame(frame)
            result_queue.put(result)

        except queue.Empty:
            continue

        except Exception as e:
            print(f"[ERROR] Worker exception: {e}")
            # Log error, continue or stop
```

---

## 10. Performance Monitoring

### A. Profiling

```python
import cProfile
import pstats

def profile_pipeline():
    """Profile multi-threaded pipeline"""
    profiler = cProfile.Profile()
    profiler.enable()

    # Run pipeline
    processor = MultiThreadedVideoProcessor('video.mp4').start()

    for _ in range(300):
        frame = processor.read()
        if frame is None:
            break

    processor.stop()

    profiler.disable()

    # Print stats
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
```

---

**Ngày tạo**: Tháng 1/2025
