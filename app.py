import sys
import os
import shutil
import socket
import torch
from torch.serialization import add_safe_globals
from flask import Flask, request, jsonify, render_template, send_from_directory, Response
from flask_cors import CORS, cross_origin
from signLanguage.pipeline.training_pipeline import TrainPipeline
from signLanguage.exception import SignException
from signLanguage.utils.main_utils import decodeImage, encodeImageIntoBase64
from signLanguage.constant.application import APP_HOST, APP_PORT

# Corrected magic variables
yolov5_path = os.path.join(os.path.dirname(__file__), 'yolov5')
sys.path.append(yolov5_path)

# Rest of imports
from models.yolo import Model

# Fix 2: Correct Flask initialization
app = Flask(__name__)
CORS(app)

def get_local_ip():
    """Get the local IP address of the machine"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

class ClientApp:
    def __init__(self):
        # Get the directory of the current file (where this ClientApp class is defined)
        self.base_dir = os.path.dirname(__file__)
        self.filename = "inputImage.jpg"

        # Construct absolute paths using self.base_dir
        self.output_dir = os.path.join(self.base_dir, "yolov5", "runs", "detect")
        self.model_path = os.path.join(self.base_dir, "yolov5", 'my_model.pt')

        self.model = self._load_model()
        self._ensure_directories()

    def _load_model(self):
        """Cache model for faster inference"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            with add_safe_globals([Model]):
                return torch.load(self.model_path, map_location=torch.device('cpu'))
        except Exception as e:
            print(f"Failed to load model: {e}")
            return None
    
    def _ensure_directories(self):
        """Create required directories if they don't exist"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, "data"), exist_ok=True)

    def run_yolo_detection(self, image_path):
        """Run YOLO detection on the given image"""
        yolo_cmd = [
            "python",
            os.path.join(self.base_dir, "yolov5", "detect.py"),
            "--weights",
            self.model_path,
            "--img",
            "416",
            "--conf",
            "0.5",
            "--source",
            image_path,
            "--project",
            self.output_dir,
            "--name",  
            "exp"      
        ]
        
        # Join the command parts into a single string
        yolo_cmd_str = " ".join(yolo_cmd)

        # Execute the command
        try:
            exit_code = os.system(yolo_cmd_str)
            if exit_code != 0:
                raise RuntimeError(f"YOLO detection failed with code {exit_code}")
        except Exception as e:
            print(f"Error running YOLO detection: {e}")

@app.route("/train")
def trainRoute():
    try:
        obj = TrainPipeline()
        obj.run_pipeline()
        return jsonify({"message": "Training Successful!", "ip": get_local_ip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/warmup")
def warmup():
    """Pre-load model before first request"""
    clApp = ClientApp()
    return jsonify({"status": "Model loaded"})

@app.route("/predict", methods=['POST', 'GET'])
@cross_origin()
def predictRoute():
    clApp = ClientApp()
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
            
        image = request.json.get('image')
        if not image:
            return jsonify({"error": "No image provided"}), 400

        decodeImage(image, clApp.filename)
        image_path = os.path.join(os.path.dirname(__file__), "data", clApp.filename)
        
        clApp.run_yolo_detection(image_path)

        # Find latest experiment
        exp_folders = [d for d in os.listdir(clApp.output_dir)
                      if d.startswith("exp") and os.path.isdir(os.path.join(clApp.output_dir, d))]
        
        if not exp_folders:
            return jsonify({"error": "No experiment folders found"}), 500
        
        # Get the path of the latest predicted image
        latest_exp_folder = max(exp_folders, key=lambda x: os.path.getmtime(os.path.join(clApp.output_dir, x)))

        latest_image_path = os.path.join(clApp.output_dir, latest_exp_folder, "image0.jpg")  

        if not os.path.exists(latest_image_path):
            return jsonify({"error": "Predicted image not found"}), 500
        
        # Serve the predicted image
        with open(latest_image_path, "rb") as image_file:
            encoded_string = encodeImageIntoBase64(latest_image_path)
            
        result = {"image": encoded_string.decode('utf-8')}
        return jsonify(result)

    except Exception as e:
        print(e)  
        return jsonify({"error": str(e)}), 500

@app.route("/live", methods=['GET'])
@cross_origin()
def predictLive():
    try:
        clApp = ClientApp()
        clApp.run_yolo_detection("0")
        return "Camera starting!!" 
    except ValueError as val:
        print(val)
        return Response("Value not found inside  json data")
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

# Route to serve images from the output directory
@app.route('/output_images/<path:path>')
def send_output_image(path):
    return send_from_directory(os.path.join("yolov5", "runs", "detect"), path)

if __name__ == "__main__":
    print(f"Running on host {APP_HOST} and port {APP_PORT}")
    app.run(host=APP_HOST, port=APP_PORT)
