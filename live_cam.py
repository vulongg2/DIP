"""
Live Camera Multi-Target Tracking & ReID.
Bật camera lên là nhận diện ngay lập tức với YOLOv8 + BoT-SORT + ReID Online Association.
- Tự động nhận diện lại đúng Global ID khi người đi ra khỏi khung hình rồi bước vào lại.
- Hiển thị HUD thời gian thực: FPS, số người, độ tương đồng ReID, thiết bị (CUDA/CPU).
- Phím tắt:
    'q' : Thoát chương trình
    'r' : Reset lại bộ nhớ ReID Gallery (test lại từ đầu)
    's' : Chụp ảnh snapshot lưu vào outputs/
    'h' : Bật / tắt bảng HUD thông tin
"""

import os
import sys
import time
import argparse
import csv
import cv2
import torch
from ultralytics import YOLO

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from core.global_manager import GlobalIdentityManager


def parse_args():
    parser = argparse.ArgumentParser(description="Live Camera Tracking & ReID")
    parser.add_argument("--source", type=str, default="0", help="Camera index (0, 1...) hoặc file video")
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.45, help="Detection confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--sim_thresh", type=float, default=0.65, help="ReID cosine similarity threshold")
    parser.add_argument("--reid_backend", type=str, default="custom", choices=["custom", "onnx"],
                        help="Backend ReID: 'custom' (Customized Metric Learning Net) hoặc 'onnx' (File có sẵn)")
    parser.add_argument("--no_flip", dest="flip", action="store_false", help="Tắt chế độ lật như gương")
    parser.set_defaults(flip=True)
    parser.add_argument("--save_crops", action="store_true", default=True, 
                        help="Tự động thu thập ảnh crop vào outputs/crops/<id>/ để train")
    parser.add_argument("--crop_interval", type=int, default=12, 
                        help="Khoảng cách số frame giữa 2 lần lưu ảnh crop")
    return parser.parse_args()


def draw_fancy_box(img, pt1, pt2, color, thickness=2, corner_len=15):
    """Vẽ bounding box kèm 4 góc nổi bật cho giao diện chuyên nghiệp."""
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    
    # 4 góc
    # Top-Left
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, thickness)
    # Top-Right
    cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, thickness)
    # Bottom-Left
    cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, thickness)
    # Bottom-Right
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, thickness)


def main():
    args = parse_args()

    # Kiểm tra CUDA
    device = 0 if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("=" * 65)
    print(f"[*] HỆ THỐNG THEO DÕI & TÁI ĐỊNH DANH (REID) THỜI GIAN THỰC")
    print(f"[*] Thiết bị: {device_name} (device={device})")
    print(f"[*] Nguồn camera: {args.source}")
    print(f"[*] Model: {args.model}")
    print(f"[*] Phím tắt: 'q'=Thoát | 'r'=Reset Gallery | 's'=Chụp ảnh | 'h'=Ẩn/Hiện HUD")
    print("=" * 65)

    # Khởi tạo YOLO và Global Manager
    print("[*] Đang tải model YOLOv8 và ReID Extractor...")
    yolo_model = YOLO(args.model)
    global_mgr = GlobalIdentityManager(
        reid_backend=args.reid_backend,
        reid_model_path="yolo26n-reid.onnx",
        similarity_thresh=args.sim_thresh,
        use_topology=False  # Đơn camera không cần ép topology
    )

    # Mở Camera / Nguồn video
    source_val = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source_val)
    if not cap.isOpened():
        print(f"[!] LỖI: Không thể mở camera hoặc nguồn video '{args.source}'.")
        sys.exit(1)

    # Thiết lập độ phân giải camera (720p nếu webcam hỗ trợ)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Chuẩn bị file log
    os.makedirs("outputs", exist_ok=True)
    log_file_path = "outputs/live_tracks.csv"
    log_file = open(log_file_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["timestamp", "global_id", "local_track_id", "x1", "y1", "x2", "y2", "conf", "sim"])

    # Biến trạng thái hiển thị & thu thập data
    show_hud = True
    is_mirror = args.flip
    auto_save_crops = args.save_crops
    crops_dir = "outputs/crops"
    os.makedirs(crops_dir, exist_ok=True)
    last_saved_frame = {}
    total_crops_saved = 0

    prev_time = time.time()
    fps_smooth = 0.0
    frame_idx = 0
    tracker_cfg = "botsort_reid.yaml" if os.path.exists("botsort_reid.yaml") else "botsort.yaml"

    window_name = "Real-time Person Tracking & ReID (Webcam)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[*] Mất tín hiệu camera hoặc hết video.")
                break

            # Lật khung hình như soi gương (Mirror Mode)
            if is_mirror:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            current_time = time.time()

            # Tracking từng frame bằng YOLO
            results = yolo_model.track(
                frame,
                persist=True,
                tracker=tracker_cfg,
                classes=[0],        # class 0 = Person
                conf=args.conf,
                imgsz=args.imgsz,
                device=device,
                verbose=False
            )

            current_persons_count = 0

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                current_persons_count = len(track_ids)

                for box, track_id, conf in zip(boxes, track_ids, confs):
                    x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
                    x2, y2 = min(w, int(box[2])), min(h, int(box[3]))

                    # Lấy ảnh crop người để ReID
                    crop = frame[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else None

                    # Gán hoặc khôi phục Global ID thông qua ReID
                    gid, color, is_new, sim = global_mgr.match_or_create(
                        cam_id="Cam_Live",
                        local_track_id=int(track_id),
                        crop_bgr=crop,
                        box=(x1, y1, x2, y2),
                        current_time=current_time
                    )

                    # Ghi log CSV
                    csv_writer.writerow([
                        f"{current_time:.2f}", gid, int(track_id),
                        x1, y1, x2, y2, f"{conf:.2f}", f"{sim:.2f}"
                    ])

                    # Tự động thu thập ảnh crop người vào outputs/crops/<gid>/ để train model
                    if auto_save_crops and crop is not None:
                        last_saved = last_saved_frame.get(gid, -args.crop_interval)
                        if frame_idx - last_saved >= args.crop_interval and (x2 - x1) >= 40 and (y2 - y1) >= 80:
                            gid_dir = os.path.join(crops_dir, str(gid))
                            os.makedirs(gid_dir, exist_ok=True)
                            crop_fname = f"crop_id{gid}_{int(current_time)}_{frame_idx:06d}.jpg"
                            cv2.imwrite(os.path.join(gid_dir, crop_fname), crop)
                            last_saved_frame[gid] = frame_idx
                            total_crops_saved += 1

                    # Vẽ Bounding Box
                    draw_fancy_box(frame, (x1, y1), (x2, y2), color, thickness=2)

                    # Nhãn thông tin Global ID
                    tag_text = f"Global ID: {gid}"
                    sub_text = f"ReID Sim: {sim:.2f}" if sim > 0 else "New Track"
                    
                    # Nền nhãn
                    (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                    cv2.rectangle(frame, (x1, max(0, y1 - 32)), (x1 + max(tw, 90) + 10, y1), color, -1)
                    cv2.putText(
                        frame, tag_text, (x1 + 5, max(12, y1 - 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
                    )
                    cv2.putText(
                        frame, sub_text, (x1 + 5, max(24, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40, 40, 40), 1
                    )

            # Tính FPS
            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / dt)

            # Vẽ HUD thông tin
            if show_hud:
                # Panel nền mờ góc trái trên
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (330, 165), (20, 20, 20), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

                infer_ms = results[0].speed.get("inference", 0.0)
                cv2.putText(frame, "REALTIME MULTI-TARGET ReID", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.putText(frame, f"FPS: {fps_smooth:.1f}  |  Infer: {infer_ms:.1f}ms", (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                cv2.putText(frame, f"Device: {device_name}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
                cv2.putText(frame, f"Dang co: {current_persons_count} nguoi trong khung hinh", (20, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                cv2.putText(frame, f"Tong Global IDs da luu: {len(global_mgr.gallery)}", (20, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
                
                # Trạng thái thu thập Data
                crop_status = f"Thu data: {'BAT (' + str(total_crops_saved) + ' anh da luu)' if auto_save_crops else 'TAT'}"
                cv2.putText(frame, crop_status, (20, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.42, 
                            (0, 220, 255) if auto_save_crops else (120, 120, 120), 1)

                # Dòng trợ giúp phía dưới
                cv2.putText(frame, "[q]: Thoat | [r]: Reset | [c]: Bat/Tat Thu Data | [s]: Chup anh | [m]: Lat guong | [h]: An HUD", 
                            (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                global_mgr.reset()
                print("[*] ĐÃ RESET REID GALLERY! Danh sách ID bắt đầu lại từ 1.")
            elif key == ord("c"):
                auto_save_crops = not auto_save_crops
                print(f"[*] Chế độ tự động thu thập data (Crops): {'BẬT' if auto_save_crops else 'TẮT'}")
            elif key == ord("m"):
                is_mirror = not is_mirror
                print(f"[*] Chế độ lật gương (Mirror mode): {'BẬT' if is_mirror else 'TẮT'}")
            elif key == ord("s"):
                snap_path = f"outputs/snapshot_{int(time.time())}.jpg"
                cv2.imwrite(snap_path, frame)
                print(f"[*] Đã chụp ảnh màn hình lưu tại: {snap_path}")
            elif key == ord("h"):
                show_hud = not show_hud

            frame_idx += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        log_file.close()
        print("=" * 65)
        print(f"[*] Đã đóng camera. Log đã lưu tại: {log_file_path}")
        print(f"[*] Tổng số Global ID đã ghi nhận trong phiên: {len(global_mgr.gallery)}")
        print("=" * 65)


if __name__ == "__main__":
    main()
