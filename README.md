# MTMCT-Vision

**Hệ thống theo dõi liên camera và tái định danh đối tượng trong môi trường trong nhà nhiều khu vực.**

MTMCT-Vision (Multi-Target Multi-Camera Tracking Vision) là giải pháp thị giác máy tính toàn diện phục vụ bài toán giám sát hành trình và duy trì danh tính toàn cục (Global ID) của đối tượng khi di chuyển xuyên qua nhiều vùng quan sát độc lập. Hệ thống kết hợp mạng nơ-ron phát hiện đối tượng YOLOv8, bộ lọc thích nghi BoT-SORT / ByteTrack, mô hình ReID tùy biến (CustomReIDNet với cấu trúc BNNeck & Triplet Loss) cùng ràng buộc đồ thị không gian - thời gian NetworkX Topology. Dự án phục vụ học tập, nghiên cứu và báo cáo đồ án môn học Xử Lý Ảnh Số (DIP).

---

## Tính năng chính

- **Nguồn dữ liệu linh hoạt:** Hỗ trợ luồng webcam máy tính, tệp video ngoại tuyến, luồng RTSP / IP Camera từ điện thoại (DroidCam, IP Webcam) và cơ chế **Virtual Multi-Camera Split** (tách 1 webcam thành 3 góc camera độc lập kèm vùng mù).
- **Phát hiện & Bám vết cục bộ (Local Tracking):** Nhận diện người chính xác cao bằng YOLOv8; duy trì vết chuyển động cục bộ qua bộ lọc Kalman và giải thuật liên kết ByteTrack / BoT-SORT với bộ đệm bộ nhớ `track_buffer = 30`.
- **Mạng ReID tùy biến với Metric Learning (CustomReIDNet):** Kiến trúc sâu tích hợp cổ chai **BNNeck (Batch Normalization Neck)** và hàm mất mát **Hard-Mining Triplet Loss**, chiếu vector đặc trưng người 512 chiều lên mặt cầu đơn vị ($L_2$ sphere).
- **Ràng buộc Đồ thị Không - Thời Gian (NetworkX Topology):** Mô hình hóa bản đồ giao thông liên phòng bằng đồ thị có trọng số; kiểm tra tính khả thi vật lý của bước chuyển $[T_{\min}, T_{\max}]$, triệt tiêu hiện tượng dịch chuyển tức thời phi thực tế.
- **Ràng buộc loại trừ không gian (Spatial Mutual Exclusion):** Chặn đứng hiện tượng trùng danh tính (hai người cùng đứng trong một khung hình không bao giờ bị gán chung một Global ID).
- **Cập nhật đặc trưng động (EMA Feature Fusion):** Thích nghi mượt mà theo thời gian thực ($f_{t} = 0.85 f_{t-1} + 0.15 f_{\text{new}}$) khi đối tượng thay đổi dáng đi, xoay người hoặc đổi hướng di chuyển.
- **Bản đồ mặt bằng 2D tương tác (2D Floor Plan):** Chiếu tọa độ và biểu diễn trực quan quỹ đạo di chuyển (Trajectory) xuyên suốt các phòng theo thời gian thực.
- **Bảng điều khiển Web Dashboard:** Giám sát trực tiếp, tra cứu lịch sử sự kiện và phân tích biểu đồ FPS độ trễ qua Streamlit.

---

## Công nghệ sử dụng

| Công nghệ | Vai trò trong hệ thống |
|---|---|
| **Python 3.10 / 3.11** | Ngôn ngữ phát triển cốt lõi và điều phối pipeline |
| **PyTorch / TorchVision** | Xây dựng, huấn luyện và suy luận mạng ReID (CustomReIDNet) trên CUDA/CPU |
| **Ultralytics YOLOv8** | Mô hình phát hiện người (Person Detection) thời gian thực |
| **OpenCV (HighGUI)** | Thu nhận luồng video, tiền xử lý khung hình và trực quan hóa giao diện HUD |
| **ByteTrack / BoT-SORT** | Duy trì quỹ đạo bám vết cục bộ (Local Tracklets) và bộ lọc Kalman |
| **NetworkX** | Xây dựng đồ thị không - thời gian (Spatio-Temporal Graph) giám sát luật chuyển tiếp |
| **Streamlit** | Xây dựng Web Dashboard giám sát trực quan 3 luồng camera và mặt bằng 2D |
| **ONNX Runtime** | Tùy chọn backend suy luận tối ưu cho mô hình ReID nhúng |

---

## Yêu cầu hệ thống

- **Hệ điều hành:** Windows 10 / Windows 11 x64 (có sẵn PowerShell).
- **Python:** Phiên bản Python 3.10.x hoặc 3.11.x x64 bản tiêu chuẩn (tích chọn `Add python.exe to PATH` khi cài đặt).
- **Card đồ họa (GPU):** Khuyên dùng GPU NVIDIA (GTX 1650, RTX 3050 trở lên) có driver hỗ trợ CUDA để đạt tốc độ > 35–50 FPS; hệ thống vẫn hỗ trợ chạy mượt trên CPU với chế độ tối ưu.
- **Camera:** 01 Webcam tích hợp (hoặc 3 điện thoại kết nối qua Wi-Fi làm IP Camera).
- **Dung lượng trống:** Tối thiểu 2 GB cho mã nguồn, trọng số mô hình và môi trường ảo Python.

---

## Hướng dẫn cài đặt

### Cách 1: Cài đặt tự động bằng 1-Click (Khuyên dùng trên Windows)

Trong thư mục dự án, chỉ cần nhấp đúp chuột vào tệp:

```powershell
.\Setup MTMCT Vision.bat
```

Bộ cài đặt sẽ tự động:
1. Tạo môi trường ảo độc lập `.venv`.
2. Kiểm tra phần cứng: Tự động tải bản PyTorch CUDA 12.1 nếu phát hiện GPU NVIDIA, hoặc bản CPU nếu không có GPU rời.
3. Cài đặt toàn bộ các thư viện phụ thuộc từ `requirements-runtime.txt`.
4. Hoàn tất và thông báo hệ thống sẵn sàng.

---

### Cách 2: Cài đặt thủ công qua PowerShell

Nếu muốn chủ động quản lý môi trường bằng lệnh:

```powershell
# 1. Tạo và kích hoạt môi trường ảo
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

# 2. Cài đặt PyTorch (Bản GPU CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# (Hoặc bản CPU nếu máy không có card NVIDIA):
# pip install torch torchvision

# 3. Cài đặt các thư viện phụ thuộc
pip install -r requirements-runtime.txt

# 4. Chạy kiểm thử hệ thống
python -m unittest discover tests
```

---

## Cách sử dụng cơ bản

Dự án cung cấp sẵn các tệp thực thi nhanh `.bat` ở thư mục gốc hoặc qua dòng lệnh Python:

### 1. Giám sát 1 Camera & Tự Thu Thập Dữ Liệu ReID
Nhấp đúp **`Run Live Cam.bat`** hoặc chạy lệnh:
```powershell
python scripts/run_live_cam.py --source 0
```
- **Phím tắt điều khiển:**
  - `c` : Bật/Tắt chế độ tự động crop ảnh người lưu vào `outputs/crops/` phục vụ huấn luyện.
  - `m` : Lật gương (Mirror mode) để chuyển động tự nhiên hơn trước webcam.
  - `r` : Xóa trắng bộ nhớ Gallery (Reset danh sách ID để test lại từ đầu).
  - `s` : Lưu ảnh chụp nhanh (Snapshot) kèm nhãn nhận diện vào thư mục `outputs/`.
  - `q` : Thoát chương trình.

### 2. Giám sát Đa Camera & Sơ đồ Mặt Bằng 2D Floor Plan
Nhấp đúp **`Run Multi Cam Demo.bat`** hoặc chạy lệnh:
```powershell
# Chế độ Giả lập (Tự tách 1 webcam làm 3 phòng A, B, C có vùng mù ở giữa):
python scripts/run_multi_cam.py --virtual-split

# Chế độ 3 Camera vật lý (Sử dụng 3 IP Camera qua điện thoại):
python scripts/run_multi_cam.py --sources http://192.168.1.10:8080/video http://192.168.1.11:8080/video http://192.168.1.12:8080/video
```

### 3. Khởi chạy Web Dashboard Trực Quan
Nhấp đúp **`Run Web Dashboard.bat`** hoặc chạy lệnh:
```powershell
streamlit run scripts/run_dashboard.py
```
Trình duyệt sẽ tự động mở giao diện tại `http://localhost:8501` hiển thị 3 phân hệ:
- **Live Stream:** Theo dõi luồng video trực tiếp từ các phòng.
- **2D Floor Plan:** Quỹ đạo di chuyển và vị trí tức thời của từng đối tượng.
- **Analytics & Benchmark:** Biểu đồ FPS, số lượng ID phân bổ và độ trễ các module.

### 4. Huấn luyện lại Mạng ReID trên Dữ Liệu Mới
Sau khi thu thập ảnh crop của các thành viên qua phím `c`:
```powershell
python scripts/train_reid.py
```
Trọng số mới được tối ưu sẽ tự động lưu vào `models/custom_reid_net.pth`.

### 5. Đánh giá Định lượng & Đo Đạc Hiệu Năng (Benchmark)
```powershell
python scripts/evaluate_benchmark.py
```

---

## Cấu trúc dự án

```text
MTMCT-Vision/
├── Setup MTMCT Vision.bat      # 1-Click cài đặt tự động môi trường và thư viện
├── Run Live Cam.bat            # 1-Click chạy giám sát 1 camera webcam
├── Run Multi Cam Demo.bat      # 1-Click chạy giả lập 3 camera + bản đồ 2D
├── Run Web Dashboard.bat       # 1-Click mở Web Dashboard qua Streamlit
├── requirements-runtime.txt    # Danh sách thư viện môi trường chuẩn
│
├── src/                        # Toàn bộ mã nguồn kiến trúc mô-đun chuẩn
│   ├── detection/              # Module phát hiện đối tượng (YOLOv8 Wrapper)
│   │   └── yolo_detector.py
│   ├── tracking/               # Module bám vết cục bộ (ByteTrack / BoT-SORT)
│   │   └── __init__.py
│   ├── reid/                   # Module ReID, BNNeck, Triplet Loss & Feature Extractor
│   │   ├── custom_reid_net.py
│   │   └── reid_extractor.py
│   ├── topology/               # Module đồ thị không - thời gian NetworkX
│   │   └── camera_graph.py
│   ├── association/            # Module quản lý Global ID Gallery, EMA & Ràng buộc loại trừ
│   │   └── global_manager.py
│   ├── visualization/          # Module vẽ bản đồ 2D Floor Plan & HUD
│   │   └── floor_plan_renderer.py
│   └── pipeline.py             # Điều phối toàn diện quy trình MTMCT
│
├── configs/                    # Cấu hình siêu tham số hệ thống
│   ├── tracker.yaml            # Cấu hình ByteTrack / BoT-SORT (track_buffer, thresholds)
│   └── topology_indoor.json    # Cấu hình đồ thị camera và thời gian di chuyển giữa các phòng
│
├── models/                     # Trọng số và thông tin định danh mô hình
│   ├── README.md               # Bảng tra cứu thông số và kích thước checkpoint
│   ├── custom_reid_net.pth     # Trọng số mạng PyTorch ReID đã huấn luyện (5.0 MB)
│   ├── yolo26n-reid.onnx       # Trọng số ReID định dạng ONNX Runtime (9.8 MB)
│   ├── yolov8n.pt              # Trọng số YOLOv8 Nano (6.5 MB)
│   └── yolov8s.pt              # Trọng số YOLOv8 Small độ chính xác cao (22.6 MB)
│
├── scripts/                    # Các kịch bản thực thi chính
│   ├── run_live_cam.py         # Kịch bản chạy 1 camera
│   ├── run_multi_cam.py        # Kịch bản chạy đa camera
│   ├── run_dashboard.py        # Kịch bản chạy Web Dashboard
│   ├── evaluate_benchmark.py   # Kịch bản đo đạc FPS & Độ trễ
│   ├── train_reid.py           # Kịch bản huấn luyện Metric Learning
│   └── windows/                # Bộ kịch bản tự động hóa PowerShell
│
├── tools/                      # Công cụ mô phỏng trực quan phục vụ thuyết trình
│   ├── simulate_cnn_reid.py
│   ├── simulate_gradient_descent.py
│   └── simulate_matrix_training.py
│
├── tests/                      # Bộ kiểm thử đơn vị (Unit Tests) tự động
│   ├── test_reid.py
│   ├── test_topology.py
│   └── test_pipeline.py
│
└── docs/                       # Tài liệu báo cáo học thuật & Sơ đồ kiến trúc
    ├── Bao_Cao_BTL_DIP_He_Thong_Theo_Doi_Lien_Camera.docx
    └── images/                 # Sơ đồ độ phân giải cao 300 DPI
        ├── fig_topology_floorplan.png
        ├── fig_custom_reid.png
        └── fig_benchmark.png
```

---

## Kết quả thực nghiệm

Hệ thống được đánh giá định lượng trên cấu hình GPU phổ thông (NVIDIA RTX / GTX):

| Phân hệ xử lý | Độ trễ trung bình (ms) | FPS tương đương | Tỷ trọng tính toán |
| :--- | :---: | :---: | :---: |
| **YOLOv8s Person Detection** | 10.42 ms | ~96.0 FPS | 54.4% |
| **BoT-SORT / ByteTrack** | 2.15 ms | ~465.1 FPS | 11.2% |
| **Custom ReID Feature Extractor** | 6.28 ms | ~159.2 FPS | 32.8% |
| **NetworkX Topology Graph Validation** | 0.31 ms | ~3225.8 FPS | 1.6% |
| **Toàn bộ Pipeline (End-to-End Loop)** | **19.16 ms** | **~52.2 FPS** | **100%** |

---

## Xử lý lỗi thường gặp

- **Không nhận lệnh Python hoặc `python`:** Khi cài đặt Python trên Windows, cần tích chọn ô **`Add python.exe to PATH`**. Mở lại cửa sổ PowerShell mới và gõ `python --version` để kiểm tra.
- **Lỗi `cv2.error: (-2:Unspecified error) The function is not implemented`:** Do cài đặt nhầm gói `opencv-python-headless`. Hãy gỡ bỏ và cài lại gói đầy đủ bằng lệnh: `pip uninstall opencv-python-headless -y` sau đó `pip install opencv-python`.
- **Lỗi không mở được Webcam (`Cannot open source 0`):** Kiểm tra xem có ứng dụng nào khác (Zoom, Teams, Camera app) đang chiếm quyền sử dụng camera hay không; đồng thời kiểm tra quyền cho phép truy cập Camera trong **Windows Settings → Privacy & Security → Camera**.
- **Lỗi tiếng Việt trên PowerShell (`UnicodeEncodeError: 'charmap'`):** Toàn bộ các script trong thư mục `scripts/` đã tích hợp sẵn cơ chế tự động chuyển mã `sys.stdout.reconfigure(encoding='utf-8')`.

---

## Lưu ý & Ghi công

- **Mục đích:** Dự án được thực hiện phục vụ mục đích nghiên cứu, học tập trong khuôn khổ học phần Xử Lý Ảnh Số (Digital Image Processing - DIP), mã đề tài `CVFP26BCPPRJC008`.
- **Bản quyền & Ghi công:** Các mô hình nền tảng YOLOv8 thuộc sở hữu của Ultralytics (giấy phép AGPL-3.0). Các kiến trúc ReID và thuật toán tích hợp tuân thủ các quy chuẩn học thuật mở.
