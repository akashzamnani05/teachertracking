import cv2
import threading
from queue import Queue
import time

class CameraManager:
    def __init__(self, config, process_frame_func):
        self.config = config
        self.process_frame_func = process_frame_func
        self.frame_queues = [Queue(maxsize=5) for _ in range(len(config["cameras"]))]
        self.running = True
        self.threads = []
        self.frame_counters = [0] * len(config["cameras"])

    def camera_thread(self, camera_config, camera_idx, queue):
        rtsp_url = f"rtsp://{camera_config['hikvision_user']}:{camera_config['hikvision_password']}@{camera_config['hikvision_ip']}:{camera_config['hikvision_port']}/{camera_config['hikvision_stream']}"
        cap = cv2.VideoCapture(rtsp_url) 
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_idx} stream")
            return
        
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        
        while self.running:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue
            
            self.frame_counters[camera_idx] += 1
            if self.frame_counters[camera_idx] % self.config["frame_skip"] != 0:
                continue
            
            processed_frame = self.process_frame_func(frame, camera_idx)
            if processed_frame is not None and not queue.full():
                queue.put((processed_frame, camera_idx))
        
        cap.release()

    def start_cameras(self):
        for idx, camera_config in enumerate(self.config["cameras"]):
            t = threading.Thread(target=self.camera_thread, args=(camera_config, idx, self.frame_queues[idx]))
            t.daemon = True
            t.start()
            self.threads.append(t)

    def stop(self):
        self.running = False
        for t in self.threads:
            t.join()