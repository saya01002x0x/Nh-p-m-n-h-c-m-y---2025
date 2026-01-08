# Hướng dẫn Cài đặt & Chạy Project Dự đoán Tuổi

## 1. Chuẩn bị
1. **Clone repo**:
   ```bash
   git clone https://github.com/saya01002x0x/Nh-p-m-n-h-c-m-y---2025
   ```
2. **Download Dataset**:
   - Tải dataset từ Kaggle: [Age prediction](https://www.kaggle.com/) (hoặc nguồn tương ứng).
   - Giải nén và lưu vào thư mục project (để train lại nếu cần).

## 2. Cài đặt Môi trường (Anaconda)
Nếu đã có file `environment.yml` trong `Source_code`:
```bash
cd Source_code
conda env create -f environment.yml
conda activate env_machine
```

## 3. Cấu hình & Thêm Model
Project yêu cầu file model đã train để chạy tính năng dự đoán.

### Nếu bạn đã có file model training (ví dụ trong folder `moDel`):
1. Vào thư mục `Source_code`.
2. Tạo thư mục mới tên là `models` (nếu chưa có).
3. Copy file model ở trong thư mục `moDel` vào thư mục `models`.

### Nếu bạn chưa có model:
- Chạy `python train.py` để tự train model mới (đòi hỏi dataset).

## 4. Chạy Ứng dụng
1. Mở terminal/cmd tại thư mục `Source_code`.
2. Kích hoạt môi trường:
   ```bash
   conda activate env_machine
   ```
3. Chạy ứng dụng Camera:
   ```bash
   python camera.py
   ```

## 5. Lưu ý
- Nếu gặp lỗi `Image.ANTIALIAS`, hãy đảm bảo dùng `Pillow` phiên bản mới nhất và code đã xử lý `Image.Resampling.LANCZOS`.
- Model càng train kỹ (val_loss thấp) thì dự đoán càng chính xác.
