"""
Mô phỏng trực quan: CÁCH MÁY TÍNH HỌC VÀ CẬP NHẬT TRỰC TIẾP MA TRẬN TRỌNG SỐ W (3x3).
Hiển thị trực tiếp phép trừ ma trận: W_mới = W_cũ - LearningRate * MaTranDaoHam
Khởi chạy: python simulate_matrix_training.py
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


def draw_matrix_box(canvas, top_left, cell_w, cell_h, matrix, title, color_theme="blue", highlight_diff=None):
    x0, y0 = top_left
    rows, cols = matrix.shape

    # Màu sắc
    if color_theme == "blue":
        header_col = (180, 50, 0)
        border_col = (200, 120, 50)
        bg_col = (250, 245, 235)
    elif color_theme == "red":
        header_col = (0, 0, 180)
        border_col = (50, 50, 200)
        bg_col = (240, 235, 250)
    else:  # green
        header_col = (0, 130, 0)
        border_col = (50, 180, 50)
        bg_col = (235, 250, 235)

    # Tiêu đề ma trận
    cv2.putText(canvas, title, (x0, y0 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.52, header_col, 2)

    # Khung bao quanh ma trận
    cv2.rectangle(canvas, (x0 - 5, y0 - 5), (x0 + cols * cell_w + 5, y0 + rows * cell_h + 5), border_col, 2)

    for r in range(rows):
        for c in range(cols):
            x1 = x0 + c * cell_w
            y1 = y0 + r * cell_h
            x2 = x1 + cell_w
            y2 = y1 + cell_h

            val = matrix[r, c]

            # Tô nền ô
            cv2.rectangle(canvas, (x1, y1), (x2, y2), bg_col, -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (210, 210, 210), 1)

            # Chữ số
            text = f"{val:+.2f}"
            cv2.putText(canvas, text, (x1 + 8, y1 + cell_h // 2 + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)


def main():
    W_canvas, H_canvas = 1040, 540
    window_name = "Mo phong: Cach cap nhat truc tiep MA TRAN TRONG SO W (3x3)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 1. Ma trận trọng số ban đầu W_cũ (khởi tạo ngẫu nhiên lộn xộn trước khi train)
    W_current = np.array([
        [ 0.15, -0.42,  0.88],
        [-0.05,  0.31, -0.74],
        [ 0.62, -0.18,  0.25]
    ], dtype=np.float32)

    # Mục tiêu lý tưởng mà W cần tiến tới (Bộ lọc viền sắc bén sau khi train)
    W_target = np.array([
        [-1.00,  2.00, -1.00],
        [-1.00,  2.00, -1.00],
        [-1.00,  2.00, -1.00]
    ], dtype=np.float32)

    lr = 0.08  # Tốc độ học
    epoch = 1
    step = 0
    paused = False

    print("=" * 65)
    print("[*] ĐANG CHẠY MÔ PHỎNG CẬP NHẬT MA TRẬN TRỌNG SỐ W (3x3)")
    print("[*] Phím SPACE: Tạm dừng / Tiếp tục | Phím 'r': Chạy lại từ đầu | Phím 'q': Thoát")
    print("=" * 65)

    while True:
        canvas = np.ones((H_canvas, W_canvas, 3), dtype=np.uint8) * 252

        # Tiêu đề chính
        cv2.putText(canvas, "MO PHONG: CAP NHAT TRUC TIEP MA TRAN TRONG SO W QUA CAC EPOCH", 
                    (30, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 160), 2)
        cv2.putText(canvas, "Cong thuc ma tran: [ W_moi ] = [ W_cu ]  -  LearningRate * [ Ma tran Dao ham dL/dW ]", 
                    (30, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)

        # Tính Ma trận Đạo hàm Gradient dL/dW (cho từng ô một trong ma trận 3x3)
        # Gradient = (W_current - W_target)
        Grad_matrix = 2.0 * (W_current - W_target)
        current_loss = float(np.mean((W_current - W_target) ** 2) * 2.5)

        # Ma trận mới sau khi trừ
        W_next = W_current - lr * Grad_matrix

        # Tọa độ vẽ các ma trận
        cell_w, cell_h = 68, 42
        y_matrices = 140

        # 1. Vẽ Ma trận W_cũ (bên trái)
        draw_matrix_box(canvas, (40, y_matrices), cell_w, cell_h, W_current, 
                        "1. Ma tran W hien tai (3x3)", color_theme="blue")

        # Dấu trừ (-) và hệ số lr
        cv2.putText(canvas, "-", (270, y_matrices + 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 200), 3)
        cv2.putText(canvas, f"({lr}) *", (295, y_matrices + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 2)

        # 2. Vẽ Ma trận Đạo hàm Gradient dL/dW (ở giữa)
        draw_matrix_box(canvas, (370, y_matrices), cell_w, cell_h, Grad_matrix, 
                        "2. Ma tran Dao ham dL/dW", color_theme="red")

        # Dấu bằng (=)
        cv2.putText(canvas, "=", (600, y_matrices + 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 140, 0), 3)

        # 3. Vẽ Ma trận W_mới (bên phải)
        draw_matrix_box(canvas, (650, y_matrices), cell_w, cell_h, W_next, 
                        "3. Ma tran W moi sau cap nhat", color_theme="green")

        # Khung thông tin huấn luyện phía dưới
        panel_y = 330
        cv2.rectangle(canvas, (40, panel_y), (W_canvas - 40, 500), (255, 255, 255), -1)
        cv2.rectangle(canvas, (40, panel_y), (W_canvas - 40, 500), (200, 200, 200), 2)

        # Cột trái thông tin
        cv2.putText(canvas, f"TRANG THAI HUAN LUYEN (EPOCH {epoch}) - Buoc lap: {step}", 
                    (60, panel_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 160), 2)
        cv2.putText(canvas, f"Muc phat Loss: {current_loss:.4f}", 
                    (60, panel_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.58, 
                    (0, 0, 200) if current_loss > 0.2 else (0, 150, 0), 2)

        # Thanh tiến trình Loss
        bar_x, bar_y, bar_w, bar_h = 60, panel_y + 90, 400, 18
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (230, 230, 230), -1)
        progress = max(0.0, min(1.0, 1.0 - (current_loss / 3.0)))
        fill_w = int(bar_w * progress)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 180, 0), -1)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (150, 150, 150), 1)

        # Cột phải giải thích
        cv2.putText(canvas, "Giai thich tung o trong ma tran:", (520, panel_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 100, 0), 2)
        exp1 = "* Moi o trong ma tran W tuong ung voi 1 trong so w_ij."
        exp2 = "* May tinh tinh dao ham rieng dL/dw_ij cho TUNG O MOT."
        exp3 = "* Lay o cua W_cu tru di (lr * DaoHam) -> Ra o cua W_moi!"
        exp4 = "* Qua moi buoc, ca 9 con so trong ma tran dong loat duoc sua."
        cv2.putText(canvas, exp1, (520, panel_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 60), 1)
        cv2.putText(canvas, exp2, (520, panel_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 60), 1)
        cv2.putText(canvas, exp3, (520, panel_y + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 180), 2)
        cv2.putText(canvas, exp4, (520, panel_y + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 120, 0), 2)

        # Phím tắt
        cv2.putText(canvas, "[SPACE]: Dung/Tiep tuc  |  [r]: Chay lai tu dau  |  [q]: Thoat", 
                    (60, panel_y + 145), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1)

        cv2.imshow(window_name, canvas)

        key = cv2.waitKey(180 if not paused else 50) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('r'):
            W_current = np.array([
                [ 0.15, -0.42,  0.88],
                [-0.05,  0.31, -0.74],
                [ 0.62, -0.18,  0.25]
            ], dtype=np.float32)
            step = 0
            epoch = 1

        if not paused and current_loss > 0.005:
            W_current = W_next.copy()
            step += 1
            if step % 5 == 0:
                epoch += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
