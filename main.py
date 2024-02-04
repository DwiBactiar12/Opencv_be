from flask import Flask, request, jsonify
import cv2
import numpy as np
from base64 import b64encode, b64decode
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/process_video', methods=['POST'])
def process_video():
    data = request.json
    video_data_base64 = data.get('video_data')
    video_data_bytes = b64decode(video_data_base64.split(',')[1])
    nparr = np.frombuffer(video_data_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    faceDeteksi = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    # Lakukan pemrosesan OpenCV di sini

    # Salin frame asli untuk digambar
    frame_with_rectangles = np.copy(frame)

    # Ubah frame ke dalam skala abu-abu
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    wajah = faceDeteksi.detectMultiScale(gray_frame,scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    count_people=0
    for (x, y, w, h) in wajah:
        count_people+=1
        # Gambar persegi panjang hijau di sekitar wajah yang terdeteksi pada frame yang disalin
        cv2.rectangle(frame_with_rectangles, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame_with_rectangles, str(count_people), (x + 40, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0))
   
    # Kodekan frame asli dan frame yang telah dimodifikasi sebagai base64
    retval, buffer = cv2.imencode('.jpg', frame)
    encoded_frame = b64encode(buffer).decode()

    retval, buffer = cv2.imencode('.jpg', frame_with_rectangles)
    encoded_frame_with_rectangles = b64encode(buffer).decode()

    return jsonify({'processed_frame_with_rectangles': encoded_frame_with_rectangles,'person_counter':count_people})

if __name__ == '__main__':
    app.run(debug=True)
