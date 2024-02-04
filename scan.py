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
    upperBodyDetector = cv2.CascadeClassifier('haarcascade_upperbody.xml')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    upper_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')
    
    # Salin frame asli untuk digambar
    frame_with_rectangles = np.copy(frame)

    # Ubah frame ke dalam skala abu-abu
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Deteksi bagian atas tubuh
    upper_bodies = upperBodyDetector.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Nomor untuk setiap orang yang terdeteksi
    person_counter = 0
    
    # Loop melalui setiap bagian atas tubuh yang terdeteksi
    for (x, y, w, h) in upper_bodies:
        person_counter += 1
        # Gambar persegi panjang hijau di sekitar bagian atas tubuh yang terdeteksi pada frame yang disalin
        cv2.rectangle(frame_with_rectangles, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Tambahkan nomor untuk setiap orang yang terdeteksi
        cv2.putText(frame_with_rectangles, str(person_counter), (x + 40, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0))
        
        # Deteksi wajah dalam bagian atas tubuh yang terdeteksi
        roi_gray = gray_frame[y:y+h, x:x+w]
        faces = upper_cascade.detectMultiScale(roi_gray)
        for (fx, fy, fw, fh) in faces:
            # Gambar persegi panjang biru di sekitar wajah yang terdeteksi pada frame yang disalin
            cv2.rectangle(frame_with_rectangles, (x + fx, y + fy), (x + fx + fw, y + fy + fh), (255, 0, 0), 2)
            
            # Deteksi mata dalam wajah yang terdeteksi
            roi_color = frame_with_rectangles[y:y+h, x:x+w]
            roi_gray_face = roi_gray[fy:fy+fh, fx:fx+fw]
            fece = face_cascade.detectMultiScale(roi_gray_face)
            for (ex, ey, ew, eh) in fece:
                # Gambar persegi panjang merah di sekitar mata yang terdeteksi pada frame yang disalin
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)
        
        # Tingkatkan nomor untuk orang berikutnya
        
   
    # Kodekan frame yang telah dimodifikasi dengan persegi panjang dan nomor sebagai base64
    retval, buffer = cv2.imencode('.jpg', frame_with_rectangles)
    encoded_frame_with_rectangles = b64encode(buffer).decode()

    return jsonify({'processed_frame_with_rectangles': encoded_frame_with_rectangles,'person_counter':person_counter})

if __name__ == '__main__':
    app.run(debug=True)
