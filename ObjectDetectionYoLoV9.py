from roboflow import Roboflow
import cv2
import threading
from flask import Flask, Response

# Roboflow API와 프로젝트 정보를 사용하여 모델 로드
rf = Roboflow(api_key="개인api")
project = rf.workspace("개인workspace").project("개인프로젝트명")
version = project.version(1)
model = version.model

# 클래스 ID와 사용자 정의 클래스 이름 매핑
class_names = {
    1: "Glove",
    2: "Helmet",
    3: "Ladder",
    4: "Mask On",
    5: "No Helmet",
    6: "No Mask",
    7: "No Safe Vest",
    8: "Person",
    10: "Cone",
    11: "No Work Uniform",
    13: "Dump Truck",
    15: "Truck",
    17: "Car",
    21: "Truck",
    23: "Regular Vehicle",
    24: "Forklift",
    25: "Large Forklift"
}

# 클래스 ID와 색상 매핑
class_colors = {
    1: (255, 0, 0),    # Blue
    2: (0, 255, 0),    # Green
    3: (0, 0, 255),    # Red
    4: (255, 255, 0),  # Cyan
    5: (255, 0, 255),  # Magenta
    6: (0, 255, 255),  # Yellow
    7: (128, 0, 128),  # Purple
    8: (0, 128, 128),  # Teal
    10: (128, 128, 0), # Olive
    11: (0, 0, 128),   # Maroon
    13: (128, 128, 128), # Gray
    15: (128, 0, 0),   # Dark Red
    17: (0, 128, 0),   # Dark Green
    21: (0, 0, 128),   # Dark Blue
    23: (192, 192, 192), # Silver
    24: (64, 64, 64),  # Dark Gray
    25: (0, 255, 127)  # Spring Green
}

# 예측 결과를 저장하는 변수와 락
predictions = []
predictions_lock = threading.Lock()

def predict_frame_async(frame):
    global predictions
    result = model.predict(frame, confidence=40, overlap=30).json()
    with predictions_lock:
        predictions = result['predictions']

def generate_frames():
    cap = cv2.VideoCapture(0)
    
    # OpenCV 설정: 웹캠 프레임 속도 설정
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 비동기 예측 시작
        threading.Thread(target=predict_frame_async, args=(frame.copy(),)).start()
        
        # 예측 결과 표시
        with predictions_lock:
            current_predictions = predictions
        
        for pred in current_predictions:
            x1 = int(pred['x'] - pred['width'] / 2)
            y1 = int(pred['y'] - pred['height'] / 2)
            x2 = int(pred['x'] + pred['width'] / 2)
            y2 = int(pred['y'] + pred['height'] / 2)
            
            class_id = int(pred['class'])  # 숫자 클래스 ID를 정수형으로 변환
            class_name = class_names.get(class_id, "Unknown")  # 사용자 정의 클래스 이름
            color = class_colors.get(class_id, (0, 255, 0))  # 클래스 색상
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

app = Flask(__name__)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """
    <html>
        <head>
            <title>Webcam Stream</title>
        </head>
        <body>
            <h1>Webcam Stream</h1>
            <img src="/video_feed">
        </body>
    </html>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
