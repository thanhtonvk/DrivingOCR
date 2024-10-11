from flask import Flask, render_template, request, redirect, url_for, send_file
import os
from werkzeug.utils import secure_filename
from modules.card_recognition import CardRecognition
import cv2
cardRecognition = CardRecognition()

app = Flask(__name__)

# Cấu hình thư mục để lưu trữ hình ảnh
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Đảm bảo thư mục upload tồn tại
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Route cho trang chủ


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        files = request.files.getlist('file')  # Lấy danh sách các file
        uploaded_files = []
        for file in files:
            if file.filename == '':
                continue
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            uploaded_files.append(filename)
        imageList = [cv2.imread(app.config['UPLOAD_FOLDER']+path)
                     for path in uploaded_files]
        results = [cardRecognition.predict(image) for image in imageList]
        print(results)
        data = []
        for path, result in zip(uploaded_files, results):
            data.append({"path": path, "result": result})
        return render_template('index.html', data=data)
    return render_template('index.html')


# @app.route('/image2table', methods=['GET', 'POST'])
# def image2table():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             return 'No file part'
#         files = request.files.getlist('file')  # Lấy danh sách các file
#         uploaded_files = []
#         for file in files:
#             if file.filename == '':
#                 continue
#             filename = secure_filename(file.filename)
#             file.save(os.path.join('data/MB/', filename))
#             uploaded_files.append(filename)
#         filePath = tableDetection.detect(uploaded_files)
#         return send_file(filePath, as_attachment=True)
#     return render_template('index.html')


# Khởi động server Flask
if __name__ == '__main__':
    app.run(debug=True)
