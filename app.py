import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from flask import Flask, render_template, Response
import threading

class ColorDetectionCamera:
    def __init__(self, n_neighbors=3):
        """
        Inisialisasi sistem deteksi warna menggunakan kamera
        
        Parameters:
        - n_neighbors (int): Jumlah tetangga terdekat untuk klasifikasi
        """
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.scaler = StandardScaler()
        
        # Definisi rentang warna HSV 
        self.color_ranges = {
            'Merah': [
                [(0, 100, 100), (10, 255, 255)],     # Rentang merah bawah
                [(170, 100, 100), (180, 255, 255)]   # Rentang merah atas
            ],
            'Hijau': [
                [(35, 50, 50), (85, 255, 255)],      # Hijau terang
                [(85, 30, 30), (95, 255, 255)]       # Hijau gelap
            ],
            'Biru': [
                [(100, 50, 50), (130, 255, 255)],    # Biru terang
                [(130, 30, 30), (140, 255, 255)]     # Biru gelap
            ],
            'Kuning': [
                [(20, 50, 50), (30, 255, 255)],      # Kuning terang
                [(30, 30, 30), (35, 255, 255)]       # Kuning gelap
            ]
        }
        
        # Variabel untuk streaming video
        self.cap = None
        self.frame = None
        self.is_running = False
        self.lock = threading.Lock()
    
    def prepare_training_data(self):
        """
        Siapkan data training dalam ruang warna HSV
        """
        # Data training warna dalam RGB
        rgb_data = {
            'Merah': [(255, 0, 0), (220, 20, 20), (200, 50, 50)],
            'Hijau': [(0, 255, 0), (50, 200, 50), (100, 150, 100)],
            'Biru': [(0, 0, 255), (50, 50, 200), (100, 100, 150)],
            'Kuning': [(255, 255, 0), (230, 230, 50), (200, 200, 100)]
        }
        
        # Konversi ke HSV dan siapkan DataFrame
        hsv_training_data = []
        for color_name, rgb_values in rgb_data.items():
            for rgb in rgb_values:
                hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]
                hsv_training_data.append({
                    'H': hsv[0],  # Hue
                    'S': hsv[1],  # Saturation
                    'V': hsv[2],  # Value
                    'color_name': color_name
                })
        
        training_data = pd.DataFrame(hsv_training_data)
        
        # Pisahkan fitur dan label
        X = training_data[['H', 'S', 'V']]
        y = training_data['color_name']
        
        # Normalisasi fitur
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data latih dan uji
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Latih model
        self.model.fit(self.X_train, self.y_train)
    
    def detect_color_threshold(self, hsv_frame):
        """
        Deteksi warna menggunakan metode threshold
        
        Parameters:
        - hsv_frame (numpy.ndarray): Frame dalam ruang warna HSV
        
        Returns:
        - list: Daftar tuple (warna terdeteksi, kontur)
        """
        detected_colors = []
        
        color_bgr = {
            'Merah': (0, 0, 255),
            'Hijau': (0, 255, 0),
            'Biru': (255, 0, 0),
            'Kuning': (0, 255, 255)
        }
        
        for color_name, ranges in self.color_ranges.items():
            # Handle semua warna yang memiliki dua rentang
            if color_name in ['Merah', 'Hijau', 'Biru', 'Kuning']:
                mask1 = cv2.inRange(hsv_frame, np.array(ranges[0][0]), np.array(ranges[0][1]))
                mask2 = cv2.inRange(hsv_frame, np.array(ranges[1][0]), np.array(ranges[1][1]))
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(hsv_frame, np.array(ranges[0]), np.array(ranges[1]))
            
            # Temukan kontur
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter kontur berdasarkan area
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Atur ambang batas sesuai kebutuhan
                    detected_colors.append((color_name, contour, color_bgr[color_name]))
        
        return detected_colors
    
    def start_camera_thread(self):
        """
        Memulai thread untuk streaming video
        """
        # Siapkan data training
        self.prepare_training_data()
        
        # Buka kamera
        self.cap = cv2.VideoCapture(0)
        self.is_running = True
        
        def process_frames():
            while self.is_running:
                # Baca frame dari kamera
                ret, frame = self.cap.read()
                
                if not ret:
                    break
                
                # Konversi ke HSV
                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                # Deteksi warna menggunakan threshold
                detected_objects = self.detect_color_threshold(hsv_frame)
                
                # Gambar kotak dan teks untuk setiap objek terdeteksi
                for i, (color, contour, bgr_color) in enumerate(detected_objects):
                    # Gambar kontur
                    cv2.drawContours(frame, [contour], -1, bgr_color, 2)
                    
                    # Dapatkan bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Gambar rectangle
                    cv2.rectangle(frame, (x, y), (x + w, y + h), bgr_color, 2)
                    
                    # Tambahkan teks
                    cv2.putText(frame, color, 
                               (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 
                               0.9, bgr_color, 2)
                
                # Simpan frame dengan threading lock
                with self.lock:
                    self.frame = frame
        
        # Mulai thread
        threading.Thread(target=process_frames, daemon=True).start()
    
    def get_frame(self):
        """
        Mendapatkan frame untuk streaming
        """
        with self.lock:
            if self.frame is not None:
                # Encode frame untuk streaming
                ret, buffer = cv2.imencode('.jpg', self.frame)
                return buffer.tobytes()
            return None
    
    def stop_camera(self):
        """
        Menghentikan kamera dan thread
        """
        self.is_running = False
        if self.cap:
            self.cap.release()

# Inisialisasi global untuk color detector
color_detector = ColorDetectionCamera()

# Inisialisasi Flask
app = Flask(__name__)

@app.route('/')
def index():
    """
    Halaman utama
    """
    return render_template('index.html')

def generate_frames():
    """
    Generator frame untuk streaming video
    """
    while True:
        frame = color_detector.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """
    Endpoint streaming video
    """
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def main():
    # Memulai deteksi kamera
    color_detector.start_camera_thread()
    
    try:
        # Jalankan server Flask
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        # Pastikan kamera ditutup
        color_detector.stop_camera()

if __name__ == '__main__':
    main()