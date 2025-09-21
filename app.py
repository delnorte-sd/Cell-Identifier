from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
import json
import base64
from io import BytesIO
import random
from sklearn.cluster import DBSCAN
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class SyntheticCellGenerator:
    """Generates realistic synthetic cell images with GFP-like fluorescence"""
    
    def __init__(self, image_size=(512, 512)):
        self.image_size = image_size
        
    def generate_cell_image(self, num_cells=None, background_noise_level=0.1, allow_overlap=True, cell_shape='pill'):
        """Generate a synthetic cell image with fluorescent cells"""
        if num_cells is None:
            num_cells = random.randint(50, 100)
        
        self.allow_overlap = allow_overlap
        self.cell_shape = cell_shape
        
        img = np.zeros((*self.image_size, 3), dtype=np.float32)
        cell_data = []
        
        attempts = 0
        max_attempts = num_cells * 10
        while len(cell_data) < num_cells and attempts < max_attempts:
            cell_info = self._add_cell(img)
            if cell_info:
                cell_data.append(cell_info)
            attempts += 1
        
        noise = np.random.normal(0, background_noise_level, img.shape)
        img += noise
        
        img = cv2.GaussianBlur(img, (5, 5), 1.5)
        
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        
        return img, cell_data
    
    def _add_cell(self, img):
        """Add a single cell to the image"""
        h, w = img.shape[:2]
        
        center_x = random.randint(50, w - 50)
        center_y = random.randint(50, h - 50)
        
        if getattr(self, 'cell_shape', 'pill') == 'ellipse':
            return self._add_ellipse_cell(img, center_x, center_y, h, w)
        else:
            return self._add_pill_cell(img, center_x, center_y, h, w)
    
    def _add_pill_cell(self, img, center_x, center_y, h, w):
        """Add a pill-shaped cell"""
        length = random.randint(8, 20)
        width = random.randint(3, 6)
        angle = random.randint(0, 180)
        intensity = random.uniform(0.8, 1.2)
        
        if self._check_overlap(img, center_x, center_y, length):
            return None
        
        mask = np.zeros((h, w), dtype=np.uint8)
        
        cos_a = np.cos(np.radians(angle))
        sin_a = np.sin(np.radians(angle))
        half_length = length // 2
        
        x1 = int(center_x - half_length * cos_a)
        y1 = int(center_y - half_length * sin_a)
        x2 = int(center_x + half_length * cos_a)
        y2 = int(center_y + half_length * sin_a)
        
        cv2.line(mask, (x1, y1), (x2, y2), 255, width)
        cv2.circle(mask, (x1, y1), width//2, 255, -1)
        cv2.circle(mask, (x2, y2), width//2, 255, -1)
        
        y, x = np.ogrid[:h, :w]
        dx = x - center_x
        dy = y - center_y
        x_rot = cos_a * dx + sin_a * dy
        y_rot = -sin_a * dx + cos_a * dy
        dist_from_center = np.abs(y_rot)
        
        cell_intensity = intensity * (1.0 - 0.2 * (dist_from_center / (width/2)))
        cell_intensity = np.clip(cell_intensity, 0.6 * intensity, intensity)
        
        mask_bool = mask > 0
        img[mask_bool, 0] += cell_intensity[mask_bool] * (26/255)
        img[mask_bool, 1] += cell_intensity[mask_bool] * (251/255)
        img[mask_bool, 2] += cell_intensity[mask_bool] * (115/255)
        
        area = np.sum(mask_bool)
        
        return {
            'center': (center_x, center_y),
            'length': length,
            'width': width,
            'angle': angle,
            'intensity': intensity,
            'area': area
        }
    
    def _add_ellipse_cell(self, img, center_x, center_y, h, w):
        """Add an ellipse-shaped cell with variable intensity"""
        major_axis = random.randint(10, 25)
        minor_axis = random.randint(6, int(major_axis * 0.8))
        angle = random.randint(0, 180)
        intensity = random.uniform(0.8, 1.0)
        
        if self._check_overlap(img, center_x, center_y, major_axis):
            return None
        
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (center_x, center_y), (major_axis//2, minor_axis//2), angle, 0, 360, 255, -1)
        
        y, x = np.ogrid[:h, :w]
        
        cos_a = np.cos(np.radians(angle))
        sin_a = np.sin(np.radians(angle))
        dx = x - center_x
        dy = y - center_y
        
        x_rot = cos_a * dx + sin_a * dy
        y_rot = -sin_a * dx + cos_a * dy
        
        ellipse_dist = np.sqrt((x_rot / (major_axis/2))**2 + (y_rot / (minor_axis/2))**2)
        
        cell_intensity = intensity * np.exp(-ellipse_dist * 2)
        
        mask_bool = mask > 0
        img[mask_bool, 0] += cell_intensity[mask_bool] * (26/255)
        img[mask_bool, 1] += cell_intensity[mask_bool] * (251/255)
        img[mask_bool, 2] += cell_intensity[mask_bool] * (115/255)
        
        area = np.sum(mask_bool)
        
        return {
            'center': (center_x, center_y),
            'major_axis': major_axis,
            'minor_axis': minor_axis,
            'angle': angle,
            'intensity': intensity,
            'area': area
        }
    
    def _check_overlap(self, img, x, y, radius):
        """Check if new cell overlaps significantly with existing cells"""
        # If overlap is disabled, always reject any overlap
        if not getattr(self, 'allow_overlap', True):
            h, w = img.shape[:2]
            y_min = max(0, y - radius)
            y_max = min(h, y + radius)
            x_min = max(0, x - radius)
            x_max = min(w, x + radius)
            
            region = img[y_min:y_max, x_min:x_max, 1]  # Check green channel
            return np.max(region) > 0.01  # Very low threshold for no overlap
        
        # Original overlap logic for when overlap is allowed
        h, w = img.shape[:2]
        # Check a smaller region to allow more overlap
        check_radius = radius // 2  # Only check half the radius
        y_min = max(0, y - check_radius)
        y_max = min(h, y + check_radius)
        x_min = max(0, x - check_radius)
        x_max = min(w, x + check_radius)
        
        region = img[y_min:y_max, x_min:x_max, 1]  # Check green channel
        
        # Very permissive overlap detection - only prevent cells directly on top of each other
        max_intensity = np.max(region)
        
        # Only reject if there's a very bright spot (almost complete overlap)
        severe_overlap = max_intensity > 1.0  # Very high threshold
        
        return severe_overlap

class SimpleCellAnalyzer:
    """Simplified cell detection and analysis using traditional computer vision"""
    
    def __init__(self):
        self.min_cell_area = 50
        self.max_cell_area = 3000
        self.intensity_threshold = 50
        self.blur_kernel = 5
        
    def set_parameters(self, min_area=None, max_area=None, intensity_thresh=None, blur_size=None):
        """Set detection parameters"""
        if min_area is not None:
            self.min_cell_area = min_area
        if max_area is not None:
            self.max_cell_area = max_area
        if intensity_thresh is not None:
            self.intensity_threshold = intensity_thresh
        if blur_size is not None:
            self.blur_kernel = blur_size
        
    def analyze_image(self, image_path):
        """Analyze cell image and return detection results"""
        # Load image
        if isinstance(image_path, str):
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            img = image_path
            
        if img is None:
            return None, None
            
        # Preprocessing with user-defined blur kernel
        blur_size = self.blur_kernel if self.blur_kernel % 2 == 1 else self.blur_kernel + 1  # Ensure odd
        img_blur = cv2.GaussianBlur(img, (blur_size, blur_size), 0)
        
        # Thresholding with user-defined intensity threshold
        _, thresh = cv2.threshold(img_blur, self.intensity_threshold, 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by user-defined area range
        valid_contours = []
        cell_data = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_cell_area < area < self.max_cell_area:
                valid_contours.append(contour)
                
                # Calculate cell properties
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = 0, 0
                
                # Create mask for this cell
                mask = np.zeros(img.shape, dtype=np.uint8)
                cv2.fillPoly(mask, [contour], 255)
                
                # Calculate mean intensity
                mean_intensity = cv2.mean(img, mask=mask)[0]

                # Classify brightness
                if mean_intensity < 73:
                    brightness = "Low"
                elif mean_intensity < 80:
                    brightness = "Medium"
                else:
                    brightness = "High"

                cell_data.append({
                    'center': (cx, cy),
                    'area': area,
                    'mean_intensity': mean_intensity,
                    'brightness': brightness,
                    'contour': contour.tolist()
                })
        
        return valid_contours, cell_data
    
    def create_result_image(self, original_img, contours, cell_data):
        """Create visualization of detection results"""
        if len(original_img.shape) == 2:
            result_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
        else:
            result_img = original_img.copy()
        
        # Draw contours and labels
        for i, (contour, cell_info) in enumerate(zip(contours, cell_data)):
            # Draw contour
            cv2.drawContours(result_img, [np.array(contour, dtype=np.int32)], -1, (0, 255, 0), 2)
            
            # Draw center point
            center = cell_info['center']
            cv2.circle(result_img, center, 3, (255, 0, 0), -1)
            
            # Add label with cell number and intensity
            label = f"Cell {i+1}: {cell_info['mean_intensity']:.1f}"
            # Add label with cell number, intensity, and brightness
            label = f"Cell {i+1}: {cell_info['mean_intensity']:.1f} ({cell_info['brightness']})"
            cv2.putText(result_img, label, (center[0] - 40, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        
        return result_img

# Initialize components
cell_generator = SyntheticCellGenerator()
cell_analyzer = SimpleCellAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_synthetic', methods=['POST'])
def generate_synthetic():
    """Generate synthetic cell images"""
    try:
        data = request.get_json()
        num_cells = data.get('num_cells')
        if num_cells is None:
            num_cells = random.randint(50, 100)
        else:
            num_cells = int(num_cells)

        noise_level = float(data.get('noise_level', 0.1))
        allow_overlap = data.get('allow_overlap', True)
        cell_shape = data.get('cell_shape', 'pill')
                
        synthetic_img, cell_data = cell_generator.generate_cell_image(
            num_cells=num_cells,
            background_noise_level=noise_level,
            allow_overlap=allow_overlap,
            cell_shape=cell_shape
        )
        
        pil_img = Image.fromarray(cv2.cvtColor(synthetic_img, cv2.COLOR_BGR2RGB))
        buffer = BytesIO()
        pil_img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        serializable_cell_data = []
        for cell in cell_data:
            if cell_shape == 'pill':
                serializable_cell = {
                    'center': (int(cell['center'][0]), int(cell['center'][1])),
                    'length': int(cell['length']),
                    'width': int(cell['width']),
                    'angle': int(cell['angle']),
                    'intensity': float(cell['intensity']),
                    'area': int(cell['area'])
                }
            else:
                serializable_cell = {
                    'center': (int(cell['center'][0]), int(cell['center'][1])),
                    'major_axis': int(cell['major_axis']),
                    'minor_axis': int(cell['minor_axis']),
                    'angle': int(cell['angle']),
                    'intensity': float(cell['intensity']),
                    'area': int(cell['area'])
                }
            serializable_cell_data.append(serializable_cell)
        
        return jsonify({
            'success': True,
            'image': f"data:image/png;base64,{img_str}",
            'cell_count': len(serializable_cell_data),
            'cell_data': serializable_cell_data,
            'cell_shape': cell_shape
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    """Analyze uploaded cell image"""
    try:
        # Get parameters from request
        if request.is_json:
            data = request.get_json()
            min_area = data.get('min_cell_area', 50)
            max_area = data.get('max_cell_area', 3000)
            intensity_thresh = data.get('intensity_threshold', 50)
            blur_kernel = data.get('blur_kernel', 5)
        else:
            min_area = int(request.form.get('min_cell_area', 50))
            max_area = int(request.form.get('max_cell_area', 3000))
            intensity_thresh = int(request.form.get('intensity_threshold', 50))
            blur_kernel = int(request.form.get('blur_kernel', 5))

        # Set analyzer parameters
        cell_analyzer.set_parameters(min_area, max_area, intensity_thresh, blur_kernel)

        # Check for sample image request
        if request.is_json and request.get_json().get('sample'):
            sample_path = os.path.join(os.path.dirname(__file__), 'static/sample.jpeg')
            if not os.path.exists(sample_path):
                return jsonify({'success': False, 'error': 'sample.jpeg not found'})
            filepath = sample_path
        else:
            if 'file' not in request.files:
                return jsonify({'success': False, 'error': 'No file uploaded'})
            file = request.files['file']
            if file.filename == '':
                return jsonify({'success': False, 'error': 'No file selected'})
            filename = f"{uuid.uuid4().hex}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

        # Analyze image
        contours, cell_data = cell_analyzer.analyze_image(filepath)
        if contours is None:
            return jsonify({'success': False, 'error': 'Could not process image'})
        original_img = cv2.imread(filepath)
        result_img = cell_analyzer.create_result_image(original_img, contours, cell_data)
        _, buffer = cv2.imencode('.png', result_img)
        img_str = base64.b64encode(buffer).decode()
        if cell_data:
            intensities = [cell['mean_intensity'] for cell in cell_data]
            avg_intensity = np.mean(intensities)
            std_intensity = np.std(intensities)
        else:
            avg_intensity = 0
            std_intensity = 0

        # Clean up uploaded file if not sample
        if not (request.is_json and request.get_json().get('sample')):
            os.remove(filepath)

        return jsonify({
            'success': True,
            'result_image': f"data:image/png;base64,{img_str}",
            'cell_count': len(cell_data),
            'average_intensity': float(avg_intensity),
            'intensity_std': float(std_intensity),
            'cell_data': cell_data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/demo_analysis')
def demo_analysis():
    """Generate and analyze a synthetic image for demo"""
    try:
        # Generate synthetic image
        synthetic_img, true_cell_data = cell_generator.generate_cell_image(
            num_cells=random.randint(50, 100)
        )
        
        # Analyze the synthetic image
        contours, detected_cell_data = cell_analyzer.analyze_image(synthetic_img)
        
        # Create result visualization
        result_img = cell_analyzer.create_result_image(synthetic_img, contours, detected_cell_data)
        
        # Convert images to base64
        synthetic_pil = Image.fromarray(synthetic_img)
        buffer1 = BytesIO()
        synthetic_pil.save(buffer1, format='PNG')
        synthetic_str = base64.b64encode(buffer1.getvalue()).decode()
        
        _, result_buffer = cv2.imencode('.png', result_img)
        result_str = base64.b64encode(result_buffer).decode()
        
        # Calculate accuracy metrics
        true_count = len(true_cell_data)
        detected_count = len(detected_cell_data)
        
        # Simple accuracy calculation
        detection_accuracy = min(detected_count / true_count if true_count > 0 else 0, 1.0) * 100
        
        return jsonify({
            'success': True,
            'original_image': f"data:image/png;base64,{synthetic_str}",
            'result_image': f"data:image/png;base64,{result_str}",
            'true_count': true_count,
            'detected_count': detected_count,
            'detection_accuracy': detection_accuracy,
            'cell_data': detected_cell_data
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)