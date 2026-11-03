# Danh Mục Mô Hình & Trọng Số (Model Checkpoints)

Thư mục này quản lý các trọng số mạng nơ-ron học sâu (Deep Neural Networks) phục vụ bài toán **Theo dõi đối tượng liên camera trong nhà (MTMCT)**.

---

## 1. Danh sách Checkpoint phân phối

| Tên File | Vai trò | Kiến trúc / Backbone | Đầu vào | Đầu ra | Kích thước |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **`yolov8s.pt`** | Phát hiện đối tượng (Người) độ chính xác cao | YOLOv8s CSPDarknet | 640 × 640 BGR | Bounding boxes [x1, y1, x2, y2], Class 0, Conf | ~22.6 MB |
| **`yolov8n.pt`** | Phát hiện đối tượng siêu nhẹ (Fallback cho CPU yếu) | YOLOv8n Nano | 640 × 640 BGR | Bounding boxes [x1, y1, x2, y2], Class 0, Conf | ~6.5 MB |
| **`custom_reid_net.pth`** | Mạng ReID tùy biến (Customized Model) | MobileNetV3 + BNNeck + Triplet Loss | 256 × 128 RGB | Vector đặc trưng $L_2$-norm 512-dim | ~5.0 MB |
| **`yolo26n-reid.onnx`** | Trích xuất đặc trưng ReID qua ONNX Runtime | ONNX Graph Optimized | 256 × 128 RGB | Vector đặc trưng $L_2$-norm 512-dim | ~9.8 MB |

---

## 2. Thông tin kỹ thuật mạng ReID tùy biến (`custom_reid_net.pth`)

* **Kiến trúc:** MobileNetV3-Small kết hợp cổ chai Batch Normalization (`BNNeck`).
* **Hàm mất mát (Metric Learning):** 
  $$\mathcal{L} = \mathcal{L}_{\text{CrossEntropy}} + \mathcal{L}_{\text{Triplet}}$$
  * Khoảng cách biên Triplet Margin: $\alpha = 0.35$.
  * Hard-Mining: Tìm cặp Hard-Positive và Hard-Negative trong từng batch để cực đại hóa khoảng cách liên danh tính.
* **Định dạng:** PyTorch state_dict (FP32).
* **Thiết bị hỗ trợ:** Tương thích cả NVIDIA CUDA (tối ưu tính toán song song) và CPU.

---

## 3. Cách cập nhật hoặc huấn luyện lại

Sau khi thu thập thêm ảnh crop từ camera thực tế vào thư mục `outputs/crops/`, chạy lệnh sau để huấn luyện lại:
```powershell
python scripts/train_reid.py
```
Trọng số mới được tối ưu sẽ tự động lưu đè vào `models/custom_reid_net.pth`.
