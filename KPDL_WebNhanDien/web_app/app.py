from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
import cv2
import os
import wikipedia

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

# Load model YOLOv8
model = YOLO("best3.pt")  # Đổi thành đường dẫn model của bạn

# Danh sách cây & mô tả
descriptions = {
    "ban": "Hoa Ban là biểu tượng của vùng Tây Bắc Việt Nam.",
    "dao": "Hoa Đào là biểu tượng của mùa xuân, thường xuất hiện vào dịp Tết.",
    "sen can": "Sen Cạn là một họ nhỏ có 3 chi và khoảng 80-90 loài thực vật thân thảo, mềm, bò trên mặt đất",
    "lan": "Đây là hoa lan hồ điệp (Phalaenopsis). Loài lan này có cánh hoa lớn, màu trắng với họng màu tím hồng, đặc trưng của nhiều giống lan hồ điệp lai hiện nay",
    "me dat": "Cây Me Đất có hoa vàng nhỏ, lá hình cỏ ba lá, thường mọc hoang."
}

# Wikipedia links
wiki_links = {
    "ban": "https://vi.wikipedia.org/wiki/Ban_T%C3%A2y_B%E1%BA%AFc",
    "dao": "https://vi.wikipedia.org/wiki/%C4%90%C3%A0o",
    "sen can": "https://vi.wikipedia.org/wiki/H%E1%BB%8D_Sen_c%E1%BA%A1n#:~:text=H%E1%BB%8D%20Sen%20c%E1%BA%A1n%20(danh%20ph%C3%A1p,m%E1%BB%99t%20lo%C3%A0i%20(mashua%20%2D%20T.",
    "lan": "https://vi.wikipedia.org/wiki/Chi_Lan_h%E1%BB%93_%C4%91i%E1%BB%87p#:~:text=Chi%20Lan%20h%E1%BB%93%20%C4%91i%E1%BB%87p%20(danh,nhi%E1%BB%81u%20lo%C3%A0i%20lai%20nh%C3%A2n%20t%E1%BA%A1o.",
    "me dat": "https://vi.wikipedia.org/wiki/H%E1%BB%8D_Chua_me_%C4%91%E1%BA%A5t"
}


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Nhận ảnh từ form
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        # Lưu ảnh tải lên
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], "input.jpg")
        file.save(filepath)

        # Chạy YOLO nhận diện
        results = model(filepath)
        result = results[0]

        # Vẽ bounding box lên ảnh
        img = cv2.imread(filepath)
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Tọa độ hộp
            label_id = int(box.cls[0])  # Nhãn ID
            score = float(box.conf[0])  # Độ tin cậy
            label = model.names[label_id]  # Lấy nhãn chính xác từ model

            # Vẽ hình chữ nhật và nhãn
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{label} ({score:.1%})", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Lưu ảnh kết quả
        output_path = "static/output.jpg"
        cv2.imwrite(output_path, img)

        # Lấy thông tin cây nhận diện đầu tiên
        detected_label = list(descriptions.keys())[int(result.boxes.cls[0])] if result.boxes else "Không xác định"
        description = descriptions.get(detected_label, "Không có thông tin.")
        wiki_url = wiki_links.get(detected_label, "")

        return render_template("index.html", result=True, detected_label=detected_label, description=description, wiki_url=wiki_url)

    return render_template("index.html", result=False)


if __name__ == "__main__":
    app.run(debug=True)
