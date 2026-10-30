"""
Mô phỏng trực quan động: Phép tích chập (Convolution) giữa Ma trận Pixel và Trọng số w,
quá trình nén thành Vector đặc trưng và so sánh khoảng cách giữa 2 Camera.
Khởi chạy: python simulate_cnn_reid.py
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


def draw_grid(canvas, top_left, cell_size, matrix, title, highlight_box=None, color_map=False):
    x0, y0 = top_left
    rows, cols = matrix.shape
    
    # Tiêu đề
    cv2.putText(canvas, title, (x0, y0 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 180), 2)

    for r in range(rows):
        for c in range(cols):
            x1 = x0 + c * cell_size
            y1 = y0 + r * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size

            val = int(matrix[r, c])
            # Tô màu ô theo giá trị pixel
            bg_color = (val, val, val) if color_map else (240, 240, 240)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), bg_color, -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (200, 200, 200), 1)

            # Chữ số pixel
            text_col = (0, 0, 255) if color_map and val > 150 else (0, 0, 0)
            if not color_map:
                text_col = (0, 0, 0)
            cv2.putText(canvas, str(val), (x1 + 6, y1 + cell_size - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_col, 1)

    # Vẽ khung trượt highlight
    if highlight_box is not None:
        hr, hc = highlight_box
        hx1 = x0 + hc * cell_size
        hy1 = y0 + hr * cell_size
        hx2 = hx1 + 3 * cell_size
        hy2 = hy1 + 3 * cell_size
        cv2.rectangle(canvas, (hx1, hy1), (hx2, hy2), (0, 0, 255), 3)


def main():
    W_canvas, H_canvas = 960, 520
    window_name = "Mo phong truc quan: Pixel x Trong so w -> Vector dac trung ReID"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 1. Ma trận pixel ảnh 8x8 (mô phỏng chiếc áo có sọc ở giữa)
    image_pixels = np.array([
        [ 30,  30,  30, 220, 220,  30,  30,  30],
        [ 30,  30,  30, 220, 220,  30,  30,  30],
        [ 30,  30,  30, 220, 220,  30,  30,  30],
        [220, 220, 220, 220, 220, 220, 220, 220],
        [220, 220, 220, 220, 220, 220, 220, 220],
        [ 30,  30,  30, 220, 220,  30,  30,  30],
        [ 30,  30,  30, 220, 220,  30,  30,  30],
        [ 30,  30,  30, 220, 220,  30,  30,  30],
    ], dtype=np.float32)

    # 2. Bộ lọc tích chập w (3x3) phát hiện sọc dọc
    kernel_w = np.array([
        [-1,  2, -1],
        [-1,  2, -1],
        [-1,  2, -1]
    ], dtype=np.float32)

    # Kết quả feature map 6x6
    feature_map = np.zeros((6, 6), dtype=np.float32)

    cell_sz = 32
    r_step, c_step = 0, 0

    print("=" * 65)
    print("[*] ĐANG CHẠY MÔ PHỎNG PHÉP TÍCH CHẬP CONVOLUTION")
    print("[*] Bấm phím SPACE để tạm dừng/tiếp tục, phím 'q' để thoát")
    print("=" * 65)

    paused = False

    while True:
        canvas = np.ones((H_canvas, W_canvas, 3), dtype=np.uint8) * 250

        # Tiêu đề chính
        cv2.putText(canvas, "MO PHONG DONG: KINH LUP TRONG SO (w) QUET TREN MA TRAN PIXEL", 
                    (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 100, 0), 2)

        # 1. Vẽ ma trận pixel ảnh (8x8)
        draw_grid(canvas, (30, 70), cell_sz, image_pixels, "1. Ma tran Pixel Anh (8x8)", 
                  highlight_box=(r_step, c_step), color_map=True)

        # 2. Vẽ bộ lọc w (3x3)
        draw_grid(canvas, (330, 70), cell_sz, kernel_w, "2. Bo loc Trong so w (3x3)", color_map=False)

        # Lấy 9 pixel đang được quét
        patch = image_pixels[r_step:r_step+3, c_step:c_step+3]
        dot_product = np.sum(patch * kernel_w)
        feature_map[r_step, c_step] = dot_product

        # 3. Vẽ công thức tính toán thời gian thực
        calc_y = 220
        cv2.putText(canvas, "3. Phep tinh nhan pixel voi w roi cong lai:", (330, calc_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 150), 2)
        eq1 = f"Tong = (p1*w1) + (p2*w2) + ... + (p9*w9)"
        eq2 = f"     = {dot_product:.0f} (Dac trung tai o [{r_step},{c_step}])"
        cv2.putText(canvas, eq1, (330, calc_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 80), 1)
        cv2.putText(canvas, eq2, (330, calc_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 0, 0), 2)

        # 4. Vẽ Feature Map kết quả (6x6)
        draw_grid(canvas, (650, 70), cell_sz, feature_map, "4. Ban do Dac trung (Feature Map)", color_map=False)

        # 5. Khung tổng kết Vector 512D & ReID ở phía dưới
        cv2.rectangle(canvas, (30, 360), (W_canvas - 30, 500), (230, 245, 230), -1)
        cv2.rectangle(canvas, (30, 360), (W_canvas - 30, 500), (0, 150, 0), 2)

        cv2.putText(canvas, "KHOI TAO VECTOR DAC TRUNG (512-DIMENSIONAL EMBEDDING):", 
                    (45, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 100, 0), 2)
        cv2.putText(canvas, "Qua Global Average Pooling, toan bo cac Feature Map nay duoc ep lai thanh 1 Vector gom 512 con so.", 
                    (45, 412), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (30, 30, 30), 1)

        sim_vec_text = f"Vector Cam 1: [ +0.82,  -0.15,  +0.65,  +0.91, ... (512 so) ]"
        sim_vec_text2= f"Vector Cam 2: [ +0.80,  -0.14,  +0.67,  +0.89, ... (512 so) ]"
        cv2.putText(canvas, sim_vec_text, (45, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 0, 0), 2)
        cv2.putText(canvas, sim_vec_text2, (45, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 180), 2)
        cv2.putText(canvas, "-> Khoang cach Euclid MIN (0.04) | Cosine Sim MAX (0.96) => CUNG MOT NGUOI!", 
                    (45, 490), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 120, 0), 2)

        cv2.imshow(window_name, canvas)

        key = cv2.waitKey(200 if not paused else 50) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused

        if not paused:
            # Nhích sang ô tiếp theo
            c_step += 1
            if c_step > 5:
                c_step = 0
                r_step += 1
                if r_step > 5:
                    r_step = 0
                    time.sleep(0.5)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
