"""
Mô phỏng trực quan động: Cách máy tính tìm ra trọng số tối ưu (Gradient Descent & Backpropagation).
Hiển thị quả bóng trọng số w lăn từ đỉnh dốc (Loss cao) xuống đáy thung lũng (Loss = 0).
Khởi chạy: python simulate_gradient_descent.py
"""

import sys
import time
import numpy as np
import cv2

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def main():
    W_canvas, H_canvas = 960, 560
    window_name = "Mo phong dong: Cach tim ra Trong so w toi uu (Gradient Descent)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Bài toán: Đầu vào x = 2, Mục tiêu y = 4. Ta cần tìm w sao cho (w * x - y)^2 nhỏ nhất.
    # Hàm Loss: L(w) = (2w - 4)^2 = 4*(w - 2)^2
    # Đáy thung lũng lý tưởng: w = 2.0 (khi đó Loss = 0)
    x_val = 2.0
    y_target = 4.0

    # Trọng số khởi đầu ngẫu nhiên ở đỉnh núi (rất tệ)
    w_start = 4.8
    w = w_start
    lr = 0.06  # Learning rate

    # Tọa độ đồ thị trên canvas
    # w chạy từ 0.0 đến 4.8
    # Loss chạy từ 0 đến 40
    graph_x0, graph_y0 = 80, 440
    graph_w, graph_h = 520, 340

    def w_to_pixel(val_w):
        # map w từ [0.0, 5.0] sang tọa độ X [graph_x0, graph_x0 + graph_w]
        return int(graph_x0 + (val_w / 5.0) * graph_w)

    def loss_to_pixel(val_loss):
        # map loss từ [0, 36] sang tọa độ Y [graph_y0, graph_y0 - graph_h]
        clamped = min(max(val_loss, 0), 36)
        return int(graph_y0 - (clamped / 36.0) * graph_h)

    # Vẽ trước đường cong parabol của hàm Loss
    w_samples = np.linspace(0.2, 4.8, 200)
    curve_pts = []
    for ws in w_samples:
        loss_val = (ws * x_val - y_target) ** 2
        px = w_to_pixel(ws)
        py = loss_to_pixel(loss_val)
        curve_pts.append((px, py))

    history_w = [w]
    step_count = 0
    paused = False

    print("=" * 65)
    print("[*] ĐANG CHẠY MÔ PHỎNG GRADIENT DESCENT (TÌM TRỌNG SỐ)")
    print("[*] Phím SPACE: Tạm dừng / Tiếp tục | Phím 'r': Chạy lại từ đầu | Phím 'q': Thoát")
    print("=" * 65)

    while True:
        canvas = np.ones((H_canvas, W_canvas, 3), dtype=np.uint8) * 252

        # 1. Tiêu đề
        cv2.putText(canvas, "MO PHONG DONG: CACH MAY TINH 'DO DUONG' TIM TRONG SO w TOI UU", 
                    (30, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 160), 2)
        cv2.putText(canvas, "Thuat toan Ha Do Doc (Gradient Descent): w_moi = w_cu - LearningRate * DaoHam", 
                    (30, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (90, 90, 90), 1)

        # 2. Vẽ hệ trục tọa độ đồ thị
        # Trục X
        cv2.line(canvas, (graph_x0 - 10, graph_y0), (graph_x0 + graph_w + 20, graph_y0), (80, 80, 80), 2)
        cv2.putText(canvas, "Trong so w", (graph_x0 + graph_w - 40, graph_y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        # Trục Y
        cv2.line(canvas, (graph_x0, graph_y0 + 10), (graph_x0, graph_y0 - graph_h - 20), (80, 80, 80), 2)
        cv2.putText(canvas, "Muc phat Loss L(w)", (graph_x0 - 60, graph_y0 - graph_h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        # Vẽ các vạch chia w
        for tick_w in [1.0, 2.0, 3.0, 4.0]:
            tx = w_to_pixel(tick_w)
            cv2.line(canvas, (tx, graph_y0 - 4), (tx, graph_y0 + 4), (100, 100, 100), 1)
            cv2.putText(canvas, f"{tick_w:.1f}", (tx - 10, graph_y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1)

        # Đánh dấu đáy thung lũng tối ưu w = 2.0
        best_x = w_to_pixel(2.0)
        cv2.line(canvas, (best_x, graph_y0), (best_x, graph_y0 - graph_h), (0, 180, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Day thung lung: w = 2.0 (Loss = 0)", (best_x - 90, graph_y0 - graph_h - 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 140, 0), 2)

        # Vẽ đường cong hàm Loss (màu xanh dương)
        for i in range(len(curve_pts) - 1):
            cv2.line(canvas, curve_pts[i], curve_pts[i + 1], (220, 140, 0), 2, cv2.LINE_AA)

        # 3. Tính toán hiện tại
        current_loss = (w * x_val - y_target) ** 2
        # Đạo hàm dL/dw = 2 * (w * x - y) * x = 2 * (2w - 4) * 2 = 8*(w - 2)
        grad = 2.0 * (w * x_val - y_target) * x_val

        # Tọa độ quả bóng w trên đồ thị
        ball_x = w_to_pixel(w)
        ball_y = loss_to_pixel(current_loss)

        # Vẽ vệt lịch sử lăn của quả bóng
        for hw in history_w:
            hx = w_to_pixel(hw)
            hy = loss_to_pixel((hw * x_val - y_target) ** 2)
            cv2.circle(canvas, (hx, hy), 3, (180, 180, 255), -1)

        # Vẽ đường tiếp tuyến thể hiện Đạo hàm (Độ dốc)
        tangent_len = 35
        # Slope trên pixel: dx, dy
        dx = 1.0
        dy = grad * 0.15
        norm_v = np.sqrt(dx**2 + dy**2) + 1e-6
        dx, dy = (dx / norm_v) * tangent_len, (dy / norm_v) * tangent_len
        cv2.line(canvas, (int(ball_x - dx), int(ball_y + dy)), 
                         (int(ball_x + dx), int(ball_y - dy)), (0, 0, 220), 2)

        # Vẽ quả bóng trọng số w (Màu đỏ nổi bật)
        cv2.circle(canvas, (ball_x, ball_y), 11, (0, 0, 255), -1)
        cv2.circle(canvas, (ball_x, ball_y), 12, (0, 0, 100), 2)
        cv2.putText(canvas, "w", (ball_x - 4, ball_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 2)

        # Mũi tên hướng lăn
        arrow_dir = -1 if grad > 0 else 1
        if abs(grad) > 0.05:
            cv2.arrowedLine(canvas, (ball_x, ball_y - 20), (ball_x + arrow_dir * 30, ball_y - 20), (0, 0, 255), 2, tipLength=0.3)
            cv2.putText(canvas, "Huong lan", (ball_x - 20, ball_y - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 200), 1)

        # 4. Bảng thông số chi tiết bên phải
        panel_x = 640
        panel_y = 100
        cv2.rectangle(canvas, (panel_x, panel_y), (W_canvas - 30, 480), (255, 255, 255), -1)
        cv2.rectangle(canvas, (panel_x, panel_y), (W_canvas - 30, 480), (200, 200, 200), 2)

        cv2.putText(canvas, f"BUOC LAP THU: {step_count}", (panel_x + 15, panel_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 160), 2)

        # Thông số
        cv2.putText(canvas, f"Trong so w hien tai: {w:.4f}", (panel_x + 15, panel_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1)
        cv2.putText(canvas, f"Muc phat Loss:       {current_loss:.4f}", (panel_x + 15, panel_y + 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 200) if current_loss > 0.1 else (0, 150, 0), 2)
        cv2.putText(canvas, f"Dao ham (Do doc):    {grad:+.4f}", (panel_x + 15, panel_y + 145), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 50, 0), 1)

        # Giải thích trạng thái
        cv2.line(canvas, (panel_x + 15, panel_y + 170), (W_canvas - 45, panel_y + 170), (220, 220, 220), 1)
        if abs(grad) > 0.08:
            status_text = "Trang thai: Dang do doc tim w toi uu..."
            status_col = (0, 100, 200)
        else:
            status_text = "THANH CONG: DA CHAT DAY THUNG LUNG!"
            status_col = (0, 140, 0)
        cv2.putText(canvas, status_text, (panel_x + 15, panel_y + 200), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_col, 2)

        # Công thức tính bước tiếp theo
        w_next = w - lr * grad
        cv2.putText(canvas, "Phep tinh buoc tiep theo:", (panel_x + 15, panel_y + 235), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 50), 1)
        cv2.putText(canvas, f"w_moi = {w:.3f} - ({lr} * {grad:+.3f})", (panel_x + 15, panel_y + 265), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 0, 0), 2)
        cv2.putText(canvas, f"      = {w_next:.4f}", (panel_x + 15, panel_y + 295), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 120, 0), 2)

        # Hướng dẫn
        cv2.putText(canvas, "[SPACE]: Dung/Chay  |  [r]: Chay lai  |  [q]: Thoat", 
                    (panel_x + 15, panel_y + 355), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 120, 120), 1)

        cv2.imshow(window_name, canvas)

        key = cv2.waitKey(150 if not paused else 50) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('r'):
            w = w_start
            history_w = [w]
            step_count = 0

        if not paused and abs(grad) > 0.01:
            w = w_next
            history_w.append(w)
            step_count += 1
            if len(history_w) > 40:
                history_w.pop(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
