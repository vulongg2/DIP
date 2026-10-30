"""
Multi-Camera Tracking & ReID Demo (Hỗ trợ tối thiểu 3 camera theo chuẩn Extended Level 3).
Đặc biệt:
- Nếu chỉ có 1 Webcam: Bật chế độ --virtual_mode (mặc định), hệ thống tự động tách không gian
  thành 3 camera khu vực không liên tục (Cam 1: Lối vào, Cam 2: Hành lang, Cam 3: Văn phòng) có vùng khuất (blind zone).
- Nếu có 3 camera thật / video: Chạy lệnh: python multi_cam_demo.py --sources 0 1 2 hoặc video1.mp4 video2.mp4 video3.mp4
- Hiển thị giao diện tổng hợp 4 khung hình:
  [ Camera 1 ]  [ Camera 2 ]
  [ Camera 3 ]  [ Sơ đồ mặt bằng 2D Floor Plan (Vị trí & Đường đi) ]
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
import torch
from ultralytics import YOLO

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from core.global_manager import GlobalIdentityManager
from core.topology import CameraTopology


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Camera Tracking & Floorplan Demo")
    parser.add_argument("--sources", nargs="+", default=["0"], 
                        help="Danh sách nguồn camera: 3 số (1 2 3), 3 link IP camera (http://...:8080/video), hoặc 3 file video")
    parser.add_argument("--virtual_mode", action="store_true", default=False, 
                        help="Ép buộc dùng chế độ giả lập 3 camera từ 1 webcam")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model (yolov8n.pt để chạy mượt 3 luồng)")
    parser.add_argument("--sim_thresh", type=float, default=0.65, help="Ngưỡng tương đồng ReID")
    parser.add_argument("--reid_backend", type=str, default="custom", choices=["custom", "onnx"],
                        help="Backend ReID: 'custom' hoặc 'onnx'")
    return parser.parse_args()


import threading


class ThreadedCameraStream:
    """
    Đọc frame trong một background thread riêng biệt để tránh tình trạng buffer lag
    khi nhận luồng video qua Wi-Fi / RTSP / HTTP từ điện thoại.
    """
    def __init__(self, src):
        self.src = src
        self.src_val = int(src) if str(src).isdigit() else str(src)
        self.cap = cv2.VideoCapture(self.src_val)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()
        self.thread = None

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()
        return self

    def update(self):
        while self.started:
            if not self.cap.isOpened():
                time.sleep(0.1)
                continue
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                if grabbed:
                    self.frame = frame
            time.sleep(0.005)

    def read(self):
        with self.read_lock:
            if self.frame is not None:
                return self.grabbed, self.frame.copy()
            return False, None

    def release(self):
        self.started = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.5)
        if self.cap.isOpened():
            self.cap.release()


class VirtualMultiCamSimulator:
    """
    Giả lập 3 góc camera indoor từ 1 luồng webcam duy nhất:
    Cam 1: Khu vực bên trái (Cửa vào - Zone 1)
    Cam 2: Khu vực giữa (Hành lang - Zone 2)
    Cam 3: Khu vực bên phải (Phòng làm việc - Zone 3)
    Ở giữa các khu vực có dải ngăn cách (blind zone) mô phỏng vùng không có camera.
    """
    def __init__(self, cap):
        self.cap = cap

    def read_frames(self):
        ret, frame = self.cap.read()
        if not ret:
            return False, {}

        # Lật như gương để cử động di chuyển trái/phải tự nhiên
        frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        
        # Chia khung hình thành 3 vùng với khoảng cách che khuất ở giữa
        # Cam 1: 0 -> 36% width
        # Blind Zone 1: 36% -> 42%
        # Cam 2: 42% -> 68% width
        # Blind Zone 2: 68% -> 74%
        # Cam 3: 74% -> 100% width
        w1_end = int(w * 0.38)
        w2_start = int(w * 0.36)
        w2_end = int(w * 0.64)
        w3_start = int(w * 0.62)

        cam1_frame = frame[:, :w1_end].copy()
        cam2_frame = frame[:, w2_start:w2_end].copy()
        cam3_frame = frame[:, w3_start:].copy()

        # Resize về cùng kích thước chuẩn để ghép
        target_size = (480, 360)
        c1 = cv2.resize(cam1_frame, target_size)
        c2 = cv2.resize(cam2_frame, target_size)
        c3 = cv2.resize(cam3_frame, target_size)

        return True, {
            "Cam_1": (c1, "Cam 1: Loi vao (Entrance)"),
            "Cam_2": (c2, "Cam 2: Hanh lang (Hallway)"),
            "Cam_3": (c3, "Cam 3: Phong lam viec (Office)")
        }


def draw_floorplan(width, height, active_positions, topology):
    """
    Vẽ sơ đồ mặt bằng khu vực 2D (Floor Plan) hiển thị vị trí và đường đi của các đối tượng.
    """
    plan = np.ones((height, width, 3), dtype=np.uint8) * 245  # Nền xám trắng

    # Tọa độ các phòng trên bản đồ 2D
    zones = {
        "Cam_1": {"rect": (25, 40, 140, 240), "color": (230, 240, 255), "label": "Loi vao (Zone 1)"},
        "Cam_2": {"rect": (170, 80, 310, 200), "color": (255, 245, 230), "label": "Hanh lang (Zone 2)"},
        "Cam_3": {"rect": (340, 40, 455, 240), "color": (235, 255, 235), "label": "Phong hoc (Zone 3)"},
    }

    # Vẽ nền các phòng
    for cam_id, zinfo in zones.items():
        x1, y1, x2, y2 = zinfo["rect"]
        cv2.rectangle(plan, (x1, y1), (x2, y2), zinfo["color"], -1)
        cv2.rectangle(plan, (x1, y1), (x2, y2), (180, 180, 180), 2)
        cv2.putText(plan, zinfo["label"], (x1 + 8, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 60), 1)

    # Vẽ đường nối hành lang (Topology Edges)
    cv2.line(plan, (140, 140), (170, 140), (100, 100, 255), 3)  # Cửa Cam 1 <-> Cam 2
    cv2.line(plan, (310, 140), (340, 140), (100, 100, 255), 3)  # Cửa Cam 2 <-> Cam 3
    cv2.putText(plan, "Cua", (142, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 180), 1)
    cv2.putText(plan, "Cua", (312, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 180), 1)

    # Tiêu đề sơ đồ
    cv2.putText(plan, "SO DO MAT BANG KHU VUC (INDOOR 2D MAP)", (25, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 120), 2)

    # Hiển thị các đối tượng đang hiện diện trên sơ đồ
    # active_positions = {global_id: {"cam": cam_id, "color": (b,g,r)}}
    for gid, info in active_positions.items():
        cam_id = info["cam"]
        color = info["color"]
        if cam_id in zones:
            zx1, zy1, zx2, zy2 = zones[cam_id]["rect"]
            cx = (zx1 + zx2) // 2 + ((gid * 17) % 30 - 15)
            cy = (zy1 + zy2) // 2 + ((gid * 19) % 40 - 20)
            
            # Chấm tròn vị trí
            cv2.circle(plan, (cx, cy), 14, color, -1)
            cv2.circle(plan, (cx, cy), 16, (30, 30, 30), 2)
            cv2.putText(plan, f"ID:{gid}", (cx - 12, cy + 4), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

    # Chú thích phía dưới
    cv2.putText(plan, f"Dang theo doi: {len(active_positions)} nguoi tren he thong", 
                (25, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 120, 0), 1)

    return plan


def main():
    args = parse_args()
    device = 0 if torch.cuda.is_available() else "cpu"

    # Xác định chế độ chạy: nếu truyền 1 camera thì tự động giả lập 3 vùng, nếu truyền >=3 camera thì dùng 3 camera thật
    is_virtual = args.virtual_mode or len(args.sources) == 1

    print("=" * 65)
    print("[*] HỆ THỐNG GIÁM SÁT LIÊN CAMERA (3-CAMERA DEMO)")
    print(f"[*] Chế độ: {'Virtual 3-Camera từ 1 Webcam' if is_virtual else f'{len(args.sources)} Camera Thật / Luồng Mạng'}")
    print(f"[*] Nguồn: {args.sources}")
    print(f"[*] Model: {args.model} | Device: {device}")
    print("=" * 65)

    yolo_model = YOLO(args.model)
    global_mgr = GlobalIdentityManager(
        reid_backend=args.reid_backend,
        reid_model_path="yolo26n-reid.onnx",
        similarity_thresh=args.sim_thresh,
        use_topology=True
    )
    topology = CameraTopology()

    # Mở nguồn camera
    if is_virtual:
        src = int(args.sources[0]) if args.sources[0].isdigit() else args.sources[0]
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"[!] Không mở được camera: {src}")
            sys.exit(1)
        sim = VirtualMultiCamSimulator(cap)
        streams = []
    else:
        # Nếu truyền 3 điện thoại (IP hoặc index webcam)
        print(f"[*] Đang kết nối tới {len(args.sources)} luồng camera điện thoại/mạng...")
        streams = [ThreadedCameraStream(s).start() for s in args.sources[:3]]
        sim = None

    tracker_cfg = "botsort_reid.yaml" if os.path.exists("botsort_reid.yaml") else "botsort.yaml"
    window_name = "Multi-Camera Indoor Person Tracking & 2D Floor Plan (Level 3 Demo)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            current_time = time.time()
            active_positions = {}  # {gid: {"cam": cam_id, "color": color}}

            if is_virtual:
                ret, cam_frames = sim.read_frames()
                if not ret:
                    break
            else:
                cam_frames = {}
                for i, strm in enumerate(streams):
                    ret, fr = strm.read()
                    cam_id = f"Cam_{i+1}"
                    if ret and fr is not None:
                        cam_frames[cam_id] = (cv2.resize(fr, (480, 360)), f"Camera {i+1} (Dien thoai {i+1})")

            annotated_frames = []

            for cam_id in ["Cam_1", "Cam_2", "Cam_3"]:
                if cam_id not in cam_frames:
                    blank = np.zeros((360, 480, 3), dtype=np.uint8)
                    cv2.putText(blank, f"{cam_id} Offline", (150, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
                    annotated_frames.append(blank)
                    continue

                frame, title = cam_frames[cam_id]
                h, w = frame.shape[:2]

                results = yolo_model.track(
                    frame,
                    persist=True,
                    tracker=tracker_cfg,
                    classes=[0],
                    conf=0.40,
                    imgsz=480,
                    device=device,
                    verbose=False
                )

                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().numpy()

                    for box, track_id in zip(boxes, track_ids):
                        x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
                        x2, y2 = min(w, int(box[2])), min(h, int(box[3]))
                        crop = frame[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else None

                        gid, color, is_new, sim_score = global_mgr.match_or_create(
                            cam_id=cam_id,
                            local_track_id=int(track_id),
                            crop_bgr=crop,
                            box=(x1, y1, x2, y2),
                            current_time=current_time
                        )

                        active_positions[gid] = {"cam": cam_id, "color": color}

                        # Vẽ box và tag
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"Global ID: {gid}"
                        cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + 110, y1), color, -1)
                        cv2.putText(frame, label, (x1 + 5, max(12, y1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)

                # Vẽ viền và tiêu đề camera
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (50, 50, 50), 2)
                cv2.rectangle(frame, (0, 0), (w, 28), (30, 30, 30), -1)
                cv2.putText(frame, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

                annotated_frames.append(frame)

            # Vẽ ô thứ 4: Sơ đồ mặt bằng khu vực 2D Floor Plan
            floorplan_img = draw_floorplan(480, 360, active_positions, topology)
            annotated_frames.append(floorplan_img)

            # Ghép thành màn hình 2x2:
            # [Cam 1] [Cam 2]
            # [Cam 3] [Floorplan]
            top_row = np.hstack((annotated_frames[0], annotated_frames[1]))
            bot_row = np.hstack((annotated_frames[2], annotated_frames[3]))
            full_display = np.vstack((top_row, bot_row))

            cv2.imshow(window_name, full_display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                global_mgr.reset()
                print("[*] Đã reset Global Manager!")

    finally:
        if is_virtual:
            cap.release()
        else:
            for s in streams:
                s.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
