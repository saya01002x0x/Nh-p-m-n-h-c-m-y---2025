1.	Clone repo link từ github: https://github.com/TruongVanHien194276/Project_IT3190_Team1.git
2.	Download bộ dataset từ Kaggle và lưu ở project: [Age prediction | Kaggle](https://www.kaggle.com/datasets/mariafrenti/age-prediction)

3. Cài đặt môi trường Anaconda
  + Cài Anaconda tại link https://www.anaconda.com/products/distribution
  + Mở Anaconda Promt (Run as Administrator)
    - Tạo môi trường mới: conda create --name env_machine
    - Cấu hình môi trường: activate env_machine
    - Cài đặt tensorflow: conda install -c conda-forge tensorflow
    - Cài đặt pandas: conda install -c anaconda pandas
    - Cài đặt matplotlib: conda install -c conda-forge matplotlib
    - Cài đặt scikit-learn: conda install -c anaconda scikit-learn
    - Cài đặt OpenCV: conda install -c conda-forge opencv
    
4. Cài đặt các thư viện còn thiếu

5. Mở project bằng pycharm
  + Chọn File -> Settings (Ctrl + Alt + S) -> Project -> Python Interpreter
  + Đổi Python Interpreter sang đường dẫn file python.exe của anaconda (Ví dụ: D:\Program Files\Anaconda\envs\env_machine\python.exe)
  
5. Sửa đường dẫn Model trong file face_detection.py

6. Chạy camera.py để mở app

7. Để test tập dữ liệu sử dụng file test.py (Sửa link Model và link dữ liệu image_dir)
