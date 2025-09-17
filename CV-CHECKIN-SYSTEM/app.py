"""
Hệ thống Computer Vision cho Check-in/out với Face Recognition
Sử dụng các thư viện mã nguồn mở miễn phí
Author: AI Assistant
"""

import cv2
import numpy as np
import face_recognition
import pyzbar.pyzbar as pyzbar
from datetime import datetime
import json
import os
import pickle
from flask import Flask, render_template, Response, jsonify, request
import base64
from io import BytesIO
from PIL import Image
import paddleocr
from paddleocr import PaddleOCR
import threading
import time
import re
import signal
import sys
import atexit

# Khởi tạo Flask app
app = Flask(__name__)

class FaceRecognitionSystem:
    """Hệ thống nhận diện khuôn mặt"""
    
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        self.load_known_faces()
        
    def load_known_faces(self):
        """Load database khuôn mặt đã lưu"""
        if os.path.exists('face_database.pkl'):
            try:
                with open('face_database.pkl', 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data['encodings']
                    self.known_face_names = data['names']
                    self.known_face_ids = data['ids']
            except (FileNotFoundError, KeyError, pickle.UnpicklingError) as e:
                print(f"Lỗi khi load face database: {e}")
                self.known_face_encodings = []
                self.known_face_names = []
                self.known_face_ids = []
        
    def save_face_database(self):
        """Lưu database khuôn mặt"""
        data = {
            'encodings': self.known_face_encodings,
            'names': self.known_face_names,
            'ids': self.known_face_ids
        }
        try:
            with open('face_database.pkl', 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Lỗi khi lưu face database: {e}")
    
    def register_face(self, image, name, employee_id):
        """Đăng ký khuôn mặt mới"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image)
        
        if len(face_locations) == 0:
            return False, "Không tìm thấy khuôn mặt"
        
        if len(face_locations) > 1:
            return False, "Phát hiện nhiều khuôn mặt, chỉ chụp 1 người"
        
        face_encoding = face_recognition.face_encodings(rgb_image, face_locations)[0]
        
        # Kiểm tra xem khuôn mặt đã tồn tại chưa
        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
        if any(matches):
            matched_idx = matches.index(True)
            return False, f"Khuôn mặt đã tồn tại với ID: {self.known_face_ids[matched_idx]}"
        
        # Thêm vào database
        self.known_face_encodings.append(face_encoding)
        self.known_face_names.append(name)
        self.known_face_ids.append(employee_id)
        self.save_face_database()
        
        return True, f"Đăng ký thành công cho {name} - ID: {employee_id}"
    
    def verify_face(self, image, expected_id=None):
        """Xác thực khuôn mặt"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image)
        
        if len(face_locations) == 0:
            return False, None, "Không tìm thấy khuôn mặt"
        
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            
            if any(matches):
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    emp_id = self.known_face_ids[best_match_index]
                    confidence = 1 - face_distances[best_match_index]
                    
                    if expected_id and emp_id != expected_id:
                        return False, None, f"Khuôn mặt không khớp với ID: {expected_id}"
                    
                    return True, {
                        'name': name,
                        'id': emp_id,
                        'confidence': float(confidence)
                    }, "Xác thực thành công"
        
        return False, None, "Khuôn mặt không được nhận diện"

class QRCodeScanner:
    """Quét và xử lý mã QR/Barcode"""
    
    @staticmethod
    def scan_qr(image):
        """Quét mã QR từ ảnh"""
        decoded_objects = pyzbar.decode(image)
        results = []
        
        for obj in decoded_objects:
            qr_data = obj.data.decode('utf-8')
            qr_type = obj.type
            
            # Vẽ khung quanh mã QR
            points = obj.polygon
            if len(points) == 4:
                pts = np.array(points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(image, [pts], True, (0, 255, 0), 2)
            
            # Parse dữ liệu nếu là JSON
            try:
                data = json.loads(qr_data)
            except (json.JSONDecodeError, ValueError):
                data = qr_data
            
            results.append({
                'type': qr_type,
                'data': data,
                'raw': qr_data
            })
        
        return results, image

class OCRProcessor:
    """Xử lý OCR cho CCCD/CMND"""
    
    def __init__(self):
        # Tạm thời disable PaddleOCR do vấn đề compatibility
        print("⚠️  Tạm thời disable PaddleOCR do vấn đề compatibility với PaddlePaddle")
        self.ocr = None
    
    def extract_id_card_info(self, image):
        """Trích xuất thông tin từ CCCD/CMND"""
        # Tạm thời disable OCR do vấn đề compatibility
        if self.ocr is None:
            return {
                'so_cccd': '',
                'ho_ten': '',
                'ngay_sinh': '',
                'gioi_tinh': '',
                'que_quan': '',
                'noi_thuong_tru': '',
                'raw_text': [],
                'message': 'Tính năng OCR tạm thời bị disable do vấn đề compatibility với PaddleOCR'
            }
            
        # Chuyển đổi image từ numpy array sang định dạng phù hợp
        if isinstance(image, np.ndarray):
            # OCR với PaddleOCR
            result = self.ocr.ocr(image, cls=True)
            
            # Parse kết quả
            extracted_info = {
                'so_cccd': '',
                'ho_ten': '',
                'ngay_sinh': '',
                'gioi_tinh': '',
                'que_quan': '',
                'noi_thuong_tru': '',
                'raw_text': []
            }
            
            if result and len(result) > 0:
                for line in result[0]:
                    text = line[1][0]
                    confidence = line[1][1]
                    extracted_info['raw_text'].append({
                        'text': text,
                        'confidence': confidence
                    })
                    
                    # Tìm các trường thông tin
                    text_lower = text.lower()
                    if 'số' in text_lower or 'so' in text_lower:
                        # Tìm số CCCD (12 chữ số)
                        import re
                        numbers = re.findall(r'\d{12}', text)
                        if numbers:
                            extracted_info['so_cccd'] = numbers[0]
                    elif 'họ và tên' in text_lower or 'ho va ten' in text_lower:
                        # Lấy tên ở dòng tiếp theo
                        pass
                    elif 'ngày sinh' in text_lower or 'ngay sinh' in text_lower:
                        # Tìm ngày sinh
                        dates = re.findall(r'\d{2}/\d{2}/\d{4}', text)
                        if dates:
                            extracted_info['ngay_sinh'] = dates[0]
            
            return extracted_info
        
        return None

class CheckInOutSystem:
    """Hệ thống Check-in/Check-out tích hợp"""
    
    def __init__(self):
        self.face_system = FaceRecognitionSystem()
        self.qr_scanner = QRCodeScanner()
        self.ocr_processor = OCRProcessor()
        self.check_records = []
        self.load_records()
    
    def load_records(self):
        """Load lịch sử check-in/out"""
        if os.path.exists('check_records.json'):
            try:
                with open('check_records.json', 'r', encoding='utf-8') as f:
                    self.check_records = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"Lỗi khi load check records: {e}")
                self.check_records = []
    
    def save_records(self):
        """Lưu lịch sử check-in/out"""
        try:
            with open('check_records.json', 'w', encoding='utf-8') as f:
                json.dump(self.check_records, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Lỗi khi lưu check records: {e}")
    
    def process_check_in_out(self, image, method='qr'):
        """Xử lý check-in/out"""
        result = {
            'success': False,
            'method': method,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'message': '',
            'data': {}
        }
        
        if method == 'qr':
            # Quét mã QR
            qr_results, _ = self.qr_scanner.scan_qr(image)
            if qr_results:
                qr_data = qr_results[0]['data']
                
                # Lấy employee_id từ QR
                if isinstance(qr_data, dict):
                    employee_id = qr_data.get('employee_id', '')
                else:
                    employee_id = qr_data
                
                # Xác thực khuôn mặt
                face_verified, face_data, message = self.face_system.verify_face(image, employee_id)
                
                if face_verified:
                    result['success'] = True
                    result['data'] = {
                        'qr_data': qr_data,
                        'face_data': face_data,
                        'action': self.determine_action(employee_id)
                    }
                    result['message'] = f"Check-{result['data']['action']} thành công cho {face_data['name']}"
                    
                    # Lưu record
                    self.add_record(employee_id, face_data['name'], result['data']['action'])
                else:
                    result['message'] = f"Xác thực khuôn mặt thất bại: {message}"
            else:
                result['message'] = "Không tìm thấy mã QR"
        
        elif method == 'face':
            # Chỉ dùng face recognition
            face_verified, face_data, message = self.face_system.verify_face(image)
            
            if face_verified:
                result['success'] = True
                result['data'] = {
                    'face_data': face_data,
                    'action': self.determine_action(face_data['id'])
                }
                result['message'] = f"Check-{result['data']['action']} thành công cho {face_data['name']}"
                
                # Lưu record
                self.add_record(face_data['id'], face_data['name'], result['data']['action'])
            else:
                result['message'] = f"Nhận diện khuôn mặt thất bại: {message}"
        
        return result
    
    def determine_action(self, employee_id):
        """Xác định là check-in hay check-out"""
        # Tìm record cuối cùng của employee
        for record in reversed(self.check_records):
            if record['employee_id'] == employee_id:
                if record['action'] == 'in':
                    return 'out'
                else:
                    return 'in'
        return 'in'
    
    def add_record(self, employee_id, name, action):
        """Thêm record check-in/out"""
        record = {
            'employee_id': employee_id,
            'name': name,
            'action': action,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.check_records.append(record)
        self.save_records()
    
    def process_cccd(self, image):
        """Xử lý OCR CCCD"""
        info = self.ocr_processor.extract_id_card_info(image)
        return info

# Khởi tạo hệ thống
system = CheckInOutSystem()
camera = None

def cleanup_camera():
    """Giải phóng camera resource"""
    global camera
    if camera is not None:
        camera.release()
        camera = None
        print("Camera đã được giải phóng")

def signal_handler(sig, frame):
    """Xử lý tín hiệu đóng ứng dụng"""
    print("\nĐang đóng ứng dụng...")
    cleanup_camera()
    sys.exit(0)

# Đăng ký cleanup functions
atexit.register(cleanup_camera)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def generate_frames():
    """Generate video frames cho streaming"""
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("Lỗi: Không thể mở camera")
            return
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Trang chủ"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    """Chụp ảnh và xử lý"""
    data = request.json
    image_data = data['image'].split(',')[1]
    method = data.get('method', 'qr')
    
    # Decode image
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Xử lý check-in/out
    result = system.process_check_in_out(image_bgr, method)
    
    return jsonify(result)

@app.route('/register_face', methods=['POST'])
def register_face():
    """Đăng ký khuôn mặt mới"""
    data = request.json
    image_data = data['image'].split(',')[1]
    name = data['name']
    employee_id = data['employee_id']
    
    # Decode image
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Đăng ký face
    success, message = system.face_system.register_face(image_bgr, name, employee_id)
    
    return jsonify({
        'success': success,
        'message': message
    })

@app.route('/process_cccd', methods=['POST'])
def process_cccd():
    """Xử lý OCR CCCD"""
    data = request.json
    image_data = data['image'].split(',')[1]
    
    # Decode image
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # OCR CCCD
    info = system.process_cccd(image_bgr)
    
    return jsonify({
        'success': info is not None,
        'data': info
    })

@app.route('/get_records')
def get_records():
    """Lấy lịch sử check-in/out"""
    return jsonify(system.check_records[-50:])  # Lấy 50 records gần nhất

if __name__ == '__main__':
    print("""
    ===================================================
    HỆ THỐNG COMPUTER VISION CHECK-IN/OUT
    ===================================================
    Các tính năng:
    1. Check-in/out qua QR Code với xác thực khuôn mặt
    2. Check-in/out chỉ bằng khuôn mặt
    3. OCR CCCD/CMND
    4. Đăng ký khuôn mặt mới
    
    Thư viện sử dụng (miễn phí):
    - face_recognition: Nhận diện khuôn mặt
    - pyzbar: Quét mã QR/Barcode
    - PaddleOCR: OCR tiếng Việt cho CCCD
    - OpenCV: Xử lý ảnh
    - Flask: Web framework
    
    Server đang chạy tại: http://localhost:5000
    ===================================================
    """)
    
    app.run(debug=True, host='0.0.0.0', port=5000)