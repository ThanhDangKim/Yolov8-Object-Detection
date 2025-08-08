# 📌 Dự án YOLOv8 với Google Colab, Flask API và Streamlit Client

## 1. Giới thiệu
Dự án này xây dựng một ứng dụng nhận diện đối tượng sử dụng **YOLOv8** làm mô hình chính, kết hợp **Google Colab** làm server chạy model và **Streamlit** làm giao diện người dùng.  
Ứng dụng hỗ trợ nhận diện:
- Ảnh tĩnh (Image)
- Video từ file
- Video từ YouTube
- Video trực tiếp từ webcam

Cấu trúc hoạt động:
1. **Server**: Chạy trên Google Colab, load mô hình YOLOv8, xây dựng API với Flask để xử lý nhận diện, và dùng **ngrok** để mở port cho máy client truy cập.
2. **Client**: Ứng dụng Streamlit chạy trên máy tính cá nhân, gọi API từ server để hiển thị kết quả nhận diện theo thời gian thực.

---

## 2. Kiến trúc code

**Luồng xử lý server**:
Client gửi request → Flask API nhận dữ liệu → YOLOv8 xử lý → Trả kết quả JSON/hình ảnh → Client hiển thị
```text
Client (Streamlit)  <----HTTP/WebRTC---->  Server (Colab / local)
        |                                        |
        | UI: image / video / youtube / webcam   | Flask API endpoints
        |                                        | YOLOv8 model (ultralytics)
        |                                        | ngrok (expose public URL)
```

### 🖥 Server_BE (Google Colab)
- **YOLOv8**: Load model và xử lý nhận diện ảnh, video, webcam.
- **Flask API**: Xây dựng các endpoint:
  - `/detect/image` : Nhận diện đối tượng trong ảnh.
  - `/detect/video` : Nhận diện đối tượng trong video.
  - `/detect/youtube` : Nhận diện video từ đường dẫn YouTube.
  - `/detect/frame` : Nhận diện từ webcam.
- **Ngrok**: Kết nối Colab ra internet để client có thể gọi API.

### 💻 Client: FE (Streamlit)
- Giao diện đơn giản, có menu chọn:
  - Nhận diện ảnh
  - Nhận diện video từ file
  - Nhận diện video từ YouTube
  - Nhận diện webcam
- Gửi dữ liệu hoặc URL tới API server, nhận kết quả và hiển thị.
- Có tính năng tải xuống kết quả đã xử lý.

---

## 3. Hướng dẫn cài đặt

> **Trước khi bắt đầu:** clone repo về một thư mục trên máy cá nhân.

### **Bước 1 — Clone repo về máy**
```bash
git clone https://github.com/ThanhDangKim/Yolov8-Object-Detection.git
```

### 📍 **Bước 2 — Chạy server**

> Bạn có 2 lựa chọn: chạy server trên Google Colab (recommended) hoặc chạy local bằng Python (chạy file .py).

#### Chạy server trên Google Colab (nếu dùng file Jupyter)

1. Upload file server notebooks (notebook .ipynb chứa code server) lên Colab.
2. Mở notebook server trên Colab và chạy từng cell từ đầu tới cuối cho đến khi:
    - Cài đặt thư viện xong
    - Model YOLO được tải 
    - Flask app được khởi chạy và ngrok xuất ra public URL 
3. Sao chép đường dẫn ngrok (ví dụ https://xxxxxx.ngrok-free.app)

### 🤖 **Bước 3 — Cấu hình FE (Streamlit) và chạy**

1. Vào thư mục FE, mở file streamlit_app.py.
2. Tìm biến cấu hình URL ngrok (NGROK_BACKEND_URL) và thay bằng ngrok public URL thu được từ Bước 2
```bash
NGROK_BACKEND_URL = "https://xxxxxx.ngrok-free.app"
```
3. Cài thư viện cho client:
```bash
pip install -r requirements.txt
```
4. Chạy câu lệnh 
```bash
streamlit run streamlit_app.py
```

---

## 4. Yêu cầu hệ thống
- Python 3.8+
- Kết nối internet ổn định (server và client cần kết nối liên tục)
- GPU (trên Colab để tăng tốc xử lý YOLOv8)

---

## 5. Ghi chú
- Nếu ngrok URL thay đổi sau khi restart server, cần cập nhật lại trong file cấu hình client.
- Với video dài hoặc webcam, thời gian xử lý phụ thuộc vào tốc độ mạng giữa client và server.

---

## ✍ Tác giả: Đặng Kim Thành
📅 Ngày cập nhật: 8/8/2025

