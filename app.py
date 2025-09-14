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
        
    def generate_cell_image(self, num_cells=None, background_noise_level=0.1):
        """Generate a synthetic cell image with fluorescent cells"""
        if num_cells is None:
            num_cells = random.randint(5, 25)
        
        # Create base image
        img = np.zeros(self.image_size, dtype=np.float32)
        cell_data = []
        
        # Generate cells
        for i in range(num_cells):
            cell_info = self._add_cell(img)
            if cell_info:
                cell_data.append(cell_info)
        
        # Add background noise
        noise = np.random.normal(0, background_noise_level, self.image_size)
        img += noise
        
        # Apply gaussian blur to simulate microscope optics
        img = cv2.GaussianBlur(img, (5, 5), 1.5)
        
        # Normalize to 0-255 range
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        
        return img, cell_data
    
    def _add_cell(self, img):
        """Add a single cell to the image"""
        h, w = img.shape
        
        # Random cell properties
        center_x = random.randint(50, w - 50)
        center_y = random.randint(50, h - 50)
        radius = random.randint(15, 35)
        intensity = random.uniform(0.4, 1.0)  # GFP intensity
        
        # Check for overlap with existing cells
        if self._check_overlap(img, center_x, center_y, radius):
            return None
        
        # Create cell mask
        y, x = np.ogrid[:h, :w]
        mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
        
        # Add cell with intensity gradient (brighter center)
        cell_img = np.zeros_like(img)
        distances = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        cell_intensity = intensity * np.exp(-distances / (radius * 0.6))
        cell_img[mask] = cell_intensity[mask]
        
        img += cell_img
        
        return {
            'center': (center_x, center_y),
            'radius': radius,
            'intensity': intensity,
            'area': np.sum(mask)
        }
    
    def _check_overlap(self, img, x, y, radius):
        """Check if new cell overlaps with existing cells"""
        h, w = img.shape
        y_min = max(0, y - radius)
        y_max = min(h, y + radius)
        x_min = max(0, x - radius)
        x_max = min(w, x + radius)
        
        region = img[y_min:y_max, x_min:x_max]
        return np.max(region) > 0.1  # Threshold for existing cell detection

class SimpleCellAnalyzer:
    """Simplified cell detection and analysis using traditional computer vision"""
    
    def __init__(self):
        self.min_cell_area = 10
        self.max_cell_area = 30000
        
    def analyze_image(self, image_path):
        """Analyze cell image and return detection results"""
        # Load image
        if isinstance(image_path, str):
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            img = image_path
            
        if img is None:
            return None, None
            
        # Preprocessing
        img_blur = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Thresholding
        _, thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
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
                if mean_intensity < 85:
                    brightness = "Low"
                elif mean_intensity < 170:
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
        num_cells = data.get('num_cells', random.randint(5, 25))
        noise_level = data.get('noise_level', 0.1)
        
        # Generate synthetic image
        synthetic_img, cell_data = cell_generator.generate_cell_image(
            num_cells=num_cells,
            background_noise_level=noise_level
        )
        
        # Convert to base64 for web display
        pil_img = Image.fromarray(synthetic_img)
        buffer = BytesIO()
        pil_img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        # Convert numpy types to Python native types for JSON serialization
        serializable_cell_data = []
        for cell in cell_data:
            serializable_cell = {
                'center': (int(cell['center'][0]), int(cell['center'][1])),
                'radius': int(cell['radius']),
                'intensity': float(cell['intensity']),
                'area': int(cell['area'])
            }
            serializable_cell_data.append(serializable_cell)
        
        return jsonify({
            'success': True,
            'image': f"data:image/png;base64,{img_str}",
            'cell_count': len(serializable_cell_data),
            'cell_data': serializable_cell_data
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    """Analyze uploaded cell image"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Save uploaded file
        filename = f"{uuid.uuid4().hex}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Analyze image
        contours, cell_data = cell_analyzer.analyze_image(filepath)
        
        if contours is None:
            return jsonify({'success': False, 'error': 'Could not process image'})
        
        # Create result visualization
        original_img = cv2.imread(filepath)
        result_img = cell_analyzer.create_result_image(original_img, contours, cell_data)
        
        # Convert result to base64
        _, buffer = cv2.imencode('.png', result_img)
        img_str = base64.b64encode(buffer).decode()
        
        # Calculate statistics
        if cell_data:
            intensities = [cell['mean_intensity'] for cell in cell_data]
            avg_intensity = np.mean(intensities)
            std_intensity = np.std(intensities)
        else:
            avg_intensity = 0
            std_intensity = 0
        
        # Clean up uploaded file
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
            num_cells=random.randint(8, 15)
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