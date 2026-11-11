"""
Multi-Camera Tracking & ReID Demo (Hỗ trợ 3 Camera IP / Điện thoại / Webcam theo chuẩn Extended Level 3).
Đặc biệt:
- Khi khởi chạy, chương trình cho phép nhập trực tiếp 3 địa chỉ IP Camera (VD: 192.168.1.10:8080)
  hoặc đường dẫn RTSP/HTTP/Video cho Cam 1, Cam 2, Cam 3.
- Nếu nhấn Enter bỏ trống ở Cam 1, hệ thống tự động chuyển sang chế độ Virtual 3-Camera từ 1 Webcam.
- Hiển thị giao diện tổng hợp 4 khung hình (2x2 Grid):
  [ Camera 1 (Zone 1) ]  [ Camera 2 (Zone 2) ]
  [ Camera 3 (Zone 3) ]  [ Sơ đồ mặt bằng 2D Floor Plan (Vị trí & Đường đi) ]
"""

import os
import sys
import time
import argparse
import threading
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Đảm bảo đường dẫn root có trong sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.association.global_manager import GlobalIdentityManager
from src.topology.camera_graph import CameraTopology
from src.detection.yolo_detector import resolve_yolo_weights
from src.tracking import resolve_tracker_config


def normalize_camera_source(raw_src):
    """
    Chuẩn hóa chuỗi địa chỉ IP / URL camera do người dùng nhập vào:
    - Nếu nhập số ('0', '1', '2') -> giữ nguyên để mở cổng USB Webcam.
    - Nếu nhập file video có thật ('video1.mp4') -> giữ nguyên.
    - Nếu chỉ nhập IP:Port (VD: '192.168.1.10:8080' hoặc '192.168.1.10:4747')
      -> Tự động thêm 'http://' và '/video' thành 'http://192.168.1.10:8080/video'.
    - Nếu chỉ nhập mỗi IP (VD: '192.168.1.10')
      -> Tự động thêm 'http://192.168.1.10:8080/video'.
    - Nếu đã nhập đầy đủ 'http://...' hoặc 'rtsp://...' -> nếu thiếu '/video' ở cổng 8080/4747 thì tự bổ sung.
    """
    s = str(raw_src).strip()
    if not s:
        return ""
    if s.isdigit() or os.path.exists(s):
        return s
    if s.startswith(("rtsp://", "rtmp://")):
        return s
    if s.startswith(("http://", "https://")):
        # Kiểm tra nếu chỉ mới gõ http://192.168.1.5:8080 mà chưa có /video
        after_proto = s.split("://", 1)[1].rstrip("/")
        if "/" not in after_proto:
            return s.rstrip("/") + "/video"
        return s

    # Trường hợp người dùng chỉ gõ IP hoặc IP:Port (ví dụ: 192.168.1.10:8080 hoặc 192.168.1.10)
    if "/" not in s:
        if ":" not in s:
            return f"http://{s}:8080/video"
        else:
            return f"http://{s}/video"
    return f"http://{s}"


def prompt_3_camera_sources():
    """
    Hiển thị giao diện nhập 3 địa chỉ IP Camera trên màn hình Console trước khi chạy.
    """
    print("=" * 70)
    print(" NHAP DIA CHI 3 CAMERA IP / WEBCAM (MULTI-CAMERA MTMCT SETUP)")
    print("=" * 70)
    print(" Huong dan nhap:")
    print("  - Cach 1 (Nhap IP dien thoai dùng app IP Webcam / DroidCam):")
    print("      Chi can go IP:Port, vi du: 192.168.1.10:8080")
    print("      (He thong tu dong chuyen thanh http://192.168.1.10:8080/video)")
    print("  - Cach 2 (Nhap link RTSP/HTTP day du hoac duong dan file video):")
    print("      Vi du: rtsp://admin:123456@192.168.1.50:554/stream1")
    print("  - Cach 3 (Chay thu bang 1 Webcam laptop chia 3 vung):")
    print("      Nhan [ENTER] bo trong o Camera 1 de chay che do Virtual 3-Cam.")
    print("-" * 70)

    cam1 = input(" >> Nhap IP / Nguon cho CAMERA 1 (Loi vao - Zone 1)   : ").strip()
    if not cam1:
        print("[*] Ban da bo trong -> Chuyen sang che do Gia lap 3 Camera tu Webcam 0.")
        return ["0"], True

    cam2 = input(" >> Nhap IP / Nguon cho CAMERA 2 (Hanh lang - Zone 2) : ").strip()
    cam3 = input(" >> Nhap IP / Nguon cho CAMERA 3 (Phong hoc - Zone 3) : ").strip()

    sources = [normalize_camera_source(cam1)]
    if cam2:
        sources.append(normalize_camera_source(cam2))
    if cam3:
        sources.append(normalize_camera_source(cam3))

    # Nếu người dùng chỉ nhập 1 cam duy nhất thì hỏi xem muốn chạy thật 1 cam hay chia 3 vùng
    if len(sources) == 1:
        return sources, False

    return sources, False


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Camera Tracking & Floorplan Demo")
    parser.add_argument("--sources", nargs="+", default=None, 
                        help="Danh sách nguồn camera: 3 IP camera (192.168.1.10:8080 ...), 3 số (0 1 2), hoặc 3 file video")
    parser.add_argument("--virtual_mode", "--virtual-split", "--virtual_split", dest="virtual_mode",
                        action="store_true", default=False, 
                        help="Ép buộc dùng chế độ giả lập 3 camera từ 1 webcam (bỏ qua bước hỏi IP)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model (yolov8n.pt để chạy mượt 3 luồng)")
    parser.add_argument("--sim_thresh", type=float, default=0.28, help="Ngưỡng tương đồng ReID (mặc định 0.28 kết hợp CLAHE triệt tiêu lệch sáng liên camera)")
    parser.add_argument("--reid_backend", type=str, default="custom", choices=["custom", "onnx"],
                        help="Backend ReID: 'custom' hoặc 'onnx'")
    return parser.parse_args()


class ThreadedCameraStream:
    """
    Đọc frame trong một background thread riêng biệt để tránh tình trạng buffer lag
    khi nhận luồng video qua Wi-Fi / RTSP / HTTP từ 3 điện thoại IP Camera.
    """
    def __init__(self, src):
        self.src = src
        self.src_val = int(src) if str(src).isdigit() else str(src)
        print(f"[*] Dang ket noi toi luong camera: {self.src_val} ...")
        self.cap = cv2.VideoCapture(self.src_val)
        # Giảm buffer size xuống 1 để luôn lấy frame mới nhất từ IP Camera (chống trễ hình)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.grabbed, self.frame = self.cap.read()
        if self.grabbed:
            print(f"    [+] Ket noi THANH CONG: {self.src_val}")
        else:
            print(f"    [!] CANH BAO: Chua nhan duoc hinh tu {self.src_val} (kiem tra lai IP/Wi-Fi)")
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
                time.sleep(0.5)
                # Tự động thử kết nối lại nếu IP cam bị rớt mạng tạm thời
                self.cap.open(self.src_val)
                continue
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                if grabbed:
                    self.frame = frame
            time.sleep(0.005)

    def read(self):
        with self.read_lock:
            if self.frame is not None and self.grabbed:
                return True, self.frame.copy()
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
    """
    def __init__(self, cap):
        self.cap = cap

    def read_frames(self):
        ret, frame = self.cap.read()
        if not ret:
            return False, {}

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        w1_end = int(w * 0.38)
        w2_start = int(w * 0.36)
        w2_end = int(w * 0.64)
        w3_start = int(w * 0.62)

        cam1_frame = frame[:, :w1_end].copy()
        cam2_frame = frame[:, w2_start:w2_end].copy()
        cam3_frame = frame[:, w3_start:].copy()

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
    plan = np.ones((height, width, 3), dtype=np.uint8) * 245

    zones = {
        "Cam_1": {"rect": (25, 40, 140, 240), "color": (230, 240, 255), "label": "Loi vao (Zone 1)"},
        "Cam_2": {"rect": (170, 80, 310, 200), "color": (255, 245, 230), "label": "Hanh lang (Zone 2)"},
        "Cam_3": {"rect": (340, 40, 455, 240), "color": (235, 255, 235), "label": "Phong hoc (Zone 3)"},
    }

    for cam_id, zinfo in zones.items():
        x1, y1, x2, y2 = zinfo["rect"]
        cv2.rectangle(plan, (x1, y1), (x2, y2), zinfo["color"], -1)
        cv2.rectangle(plan, (x1, y1), (x2, y2), (180, 180, 180), 2)
        cv2.putText(plan, zinfo["label"], (x1 + 8, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 60), 1)

    cv2.line(plan, (140, 140), (170, 140), (100, 100, 255), 3)
    cv2.line(plan, (310, 140), (340, 140), (100, 100, 255), 3)
    cv2.putText(plan, "Cua", (142, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 180), 1)
    cv2.putText(plan, "Cua", (312, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 180), 1)

    cv2.putText(plan, "SO DO MAT BANG KHU VUC (INDOOR 2D MAP)", (25, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 120), 2)

    for gid, info in active_positions.items():
        cam_id = info["cam"]
        color = info["color"]
        if cam_id in zones:
            zx1, zy1, zx2, zy2 = zones[cam_id]["rect"]
            cx = (zx1 + zx2) // 2 + ((gid * 17) % 30 - 15)
            cy = (zy1 + zy2) // 2 + ((gid * 19) % 40 - 20)
            
            cv2.circle(plan, (cx, cy), 14, color, -1)
            cv2.circle(plan, (cx, cy), 16, (30, 30, 30), 2)
            cv2.putText(plan, f"ID:{gid}", (cx - 12, cy + 4), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

    cv2.putText(plan, f"Dang theo doi: {len(active_positions)} nguoi tren he thong", 
                (25, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 120, 0), 1)

    return plan


def main():
    args = parse_args()
    device = 0 if torch.cuda.is_available() else "cpu"

    # Nếu người dùng truyền sẵn --virtual_mode trên dòng lệnh
    if args.virtual_mode:
        sources = args.sources if args.sources else ["0"]
        is_virtual = True
    # Nếu người dùng truyền sẵn --sources trên dòng lệnh (VD: --sources 192.168.1.10:8080 ...)
    elif args.sources is not None:
        sources = [normalize_camera_source(s) for s in args.sources]
        is_virtual = False
    # Ngược lại (chạy bình thường hoặc click đúp file .bat) -> Hỏi nhập 3 địa chỉ IP Camera!
    else:
        sources, is_virtual = prompt_3_camera_sources()

    print("=" * 70)
    print("[*] HE THONG GIAM SAT LIEN CAMERA (3-CAMERA MTMCT SYSTEM)")
    print(f"[*] Che do: {'Virtual 3-Camera tu 1 Webcam' if is_virtual else f'{len(sources)} Camera IP / Luong Thuc Te'}")
    for idx, s in enumerate(sources):
        print(f"    - Cam_{idx+1}: {s}")
    print(f"[*] Model: {args.model} | Device: {device}")
    print("=" * 70)

    weights_path = resolve_yolo_weights(args.model)
    # Khởi tạo 3 bộ YOLO+Tracker độc lập cho 3 Camera để Kalman Filter của Cam 1 không bị lẫn tọa độ với Cam 2, Cam 3
    yolo_models = {
        "Cam_1": YOLO(weights_path),
        "Cam_2": YOLO(weights_path),
        "Cam_3": YOLO(weights_path),
    }

    global_mgr = GlobalIdentityManager(
        reid_backend=args.reid_backend,
        reid_model_path="custom_reid_net.pth" if args.reid_backend == "custom" else "yolo26n-reid.onnx",
        similarity_thresh=args.sim_thresh,
        use_topology=True
    )
    topology = CameraTopology()

    if is_virtual:
        src = int(sources[0]) if str(sources[0]).isdigit() else sources[0]
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"[!] Khong mo duoc camera: {src}")
            sys.exit(1)
        sim = VirtualMultiCamSimulator(cap)
        streams = []
    else:
        streams = [ThreadedCameraStream(s).start() for s in sources[:3]]
        sim = None

    tracker_cfg = resolve_tracker_config("tracker.yaml")
    window_name = "Multi-Camera Indoor Person Tracking & 2D Floor Plan (Level 3 Demo)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            current_time = time.time()
            active_positions = {}

            if is_virtual:
                ret, cam_frames = sim.read_frames()
                if not ret:
                    break
            else:
                cam_frames = {}
                zone_names = [
                    "Cam 1: Loi vao (Zone 1)",
                    "Cam 2: Hanh lang (Zone 2)",
                    "Cam 3: Phong hoc (Zone 3)"
                ]
                for i, strm in enumerate(streams):
                    ret, fr = strm.read()
                    cam_id = f"Cam_{i+1}"
                    if ret and fr is not None:
                        cam_frames[cam_id] = (cv2.resize(fr, (480, 360)), zone_names[i])

            annotated_frames = []

            for cam_id in ["Cam_1", "Cam_2", "Cam_3"]:
                if cam_id not in cam_frames:
                    blank = np.zeros((360, 480, 3), dtype=np.uint8)
                    cv2.putText(blank, f"{cam_id} Offline / Waiting...", (110, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 100, 100), 2)
                    annotated_frames.append(blank)
                    continue

                frame, title = cam_frames[cam_id]
                h, w = frame.shape[:2]

                results = yolo_models[cam_id].track(
                    frame,
                    persist=True,
                    tracker=tracker_cfg,
                    classes=[0],
                    conf=0.25,      # Hạ ngưỡng conf xuống 0.25 để bắt nhạy cả khi đèn hành lang mờ/bật trễ
                    imgsz=480,
                    device=device,
                    verbose=False
                )

                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().numpy() if results[0].boxes.id is not None else None

                    for i, box in enumerate(boxes):
                        x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
                        x2, y2 = min(w, int(box[2])), min(h, int(box[3]))
                        if (x2 - x1) < 15 or (y2 - y1) < 30:
                            continue

                        track_id = int(track_ids[i]) if track_ids is not None else (i + 1)
                        crop = frame[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else None

                        gid, color, is_new, sim_score = global_mgr.match_or_create(
                            cam_id=cam_id,
                            local_track_id=track_id,
                            crop_bgr=crop,
                            box=(x1, y1, x2, y2),
                            current_time=current_time
                        )

                        active_positions[gid] = {"cam": cam_id, "color": color}

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"Global ID: {gid} ({sim_score:.2f})"
                        cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + 145, y1), color, -1)
                        cv2.putText(frame, label, (x1 + 5, max(12, y1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2)

                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (50, 50, 50), 2)
                cv2.rectangle(frame, (0, 0), (w, 28), (30, 30, 30), -1)
                cv2.putText(frame, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1)

                annotated_frames.append(frame)

            floorplan_img = draw_floorplan(480, 360, active_positions, topology)
            annotated_frames.append(floorplan_img)

            top_row = np.hstack((annotated_frames[0], annotated_frames[1]))
            bot_row = np.hstack((annotated_frames[2], annotated_frames[3]))
            full_display = np.vstack((top_row, bot_row))

            cv2.imshow(window_name, full_display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                global_mgr.reset()
                print("[*] Da reset Global Manager!")

    finally:
        if is_virtual:
            cap.release()
        else:
            for s in streams:
                s.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
