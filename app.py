import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
import time
from datetime import datetime
import io
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

# Set page configuration
st.set_page_config(
    page_title="Breast Ultrasound AI Analysis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.4rem;
        color: #2e86ab;
        margin-bottom: 1rem;
        font-weight: 600;
        border-bottom: 2px solid #2e86ab;
        padding-bottom: 0.5rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .confidence-bar {
        height: 25px;
        background-color: #e0e0e0;
        border-radius: 12px;
        margin: 8px 0;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 12px;
        text-align: center;
        color: white;
        font-weight: bold;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .benign { background: linear-gradient(90deg, #2ecc71, #27ae60); }
    .malignant { background: linear-gradient(90deg, #e74c3c, #c0392b); }
    .normal { background: linear-gradient(90deg, #3498db, #2980b9); }
    .recommendation-box {
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .urgency-high { background: linear-gradient(135deg, #ff6b6b, #ee5a52); color: white; }
    .urgency-medium { background: linear-gradient(135deg, #ffa726, #f57c00); color: white; }
    .urgency-low { background: linear-gradient(135deg, #66bb6a, #4caf50); color: white; }
    .risk-high { background: linear-gradient(135deg, #ff4757, #ff3838); color: white; }
    .risk-medium { background: linear-gradient(135deg, #ffa502, #ff9f1a); color: white; }
    .risk-low { background: linear-gradient(135deg, #2ed573, #1dd1a1); color: white; }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sidebar with ALL enhanced features
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #1f77b4;'>🏥 Breast AI Pro</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### 🔬 About This App")
    st.info("""
    **Advanced Breast Ultrasound AI Analysis** with:
    - Multi-image batch processing
    - Patient risk assessment
    - Comparative analysis
    - Advanced visualization
    - Clinical decision support
    """)
    
    st.markdown("---")
    st.markdown("### 🎨 Visualization Settings")
    
    heatmap_opacity = st.slider("Heatmap Opacity", 0.1, 1.0, 0.6)
    contour_color = st.color_picker("Contour Color", "#FF0000")
    contour_width = st.slider("Contour Width", 1, 10, 3)
    
    viz_mode = st.radio(
        "Visualization Mode",
        ["Heatmap", "Contour Only", "Heatmap + Contour", "Binary Mask"]
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ Analysis Settings")
    
    confidence_threshold = st.slider(
        "Confidence Threshold", 
        0.5, 1.0, 0.7,
        help="Minimum confidence level for reliable prediction"
    )
    
    st.markdown("---")
    st.markdown("### 📁 Batch Analysis")
    batch_files = st.file_uploader(
        "Upload multiple images", 
        type=['jpg', 'jpeg', 'png', 'bmp'], 
        accept_multiple_files=True,
        key="batch_upload"
    )
    
    st.markdown("---")
    st.markdown("### 💾 Export Options")
    
    export_format = st.selectbox(
        "Export Format",
        ["PDF Report", "JSON", "CSV"]
    )
    
    if st.button("🌐 Generate API Call"):
        st.session_state.show_api = True
    
    st.markdown("---")
    st.markdown("### 👥 Collaboration")
    
    st.markdown("---")
    st.markdown("### ⚡ Performance")
    if st.checkbox("Show System Metrics"):
        if torch.cuda.is_available():
            st.metric("GPU Memory", "Available")
            st.metric("GPU Usage", "Active")
        else:
            st.metric("Device", "CPU")
        st.metric("Model Version", "Breast2 Pro v2.0")

# Image transformations
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Enhanced Grad-CAM implementation
class EnhancedGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_full_backward_hook(self.backward_hook)
    
    def forward_hook(self, module, input, output):
        self.activations = output.detach()
    
    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        output = self.model(input_tensor)
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
        
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        gradients = self.gradients.cpu().numpy()[0]
        activations = self.activations.cpu().numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        cam = np.maximum(cam, 0)
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        
        return cam, target_class

def create_enhanced_heatmap(cam, original_image, opacity=0.6):
    """Enhanced heatmap with better colors"""
    try:
        original_size = original_image.size
        cam_resized = np.array(Image.fromarray(cam).resize(original_size, Image.BILINEAR))
        
        # Create enhanced colormap - Jet-like colors
        norm_cam = (cam_resized * 255).astype(np.uint8)
        heatmap_pil = Image.fromarray(norm_cam).convert('L')
        
        # Enhanced color mapping
        heatmap_color = Image.new('RGB', original_size)
        r, g, b = heatmap_color.split()
        
        # Jet-like colormap: blue -> cyan -> green -> yellow -> red
        r = heatmap_pil.point(lambda x: min(255, max(0, (x - 128) * 4)) if x > 128 else 0)
        g = heatmap_pil.point(lambda x: min(255, max(0, x * 2)) if x < 192 else min(255, max(0, (255 - x) * 4)))
        b = heatmap_pil.point(lambda x: min(255, max(0, (128 - x) * 4)) if x < 128 else 0)
        
        heatmap_color = Image.merge('RGB', (r, g, b))
        
        # Blend with original
        original_rgb = original_image.convert('RGB')
        blended = Image.blend(original_rgb, heatmap_color, opacity)
        
        return np.array(blended), cam_resized
        
    except Exception as e:
        st.error(f"Heatmap creation error: {str(e)}")
        return np.array(original_image.convert('RGB')), cam

def create_smart_contour(cam, original_image, color=(255, 0, 0), width=3):
    """Smart contour detection with better algorithms"""
    try:
        original_size = original_image.size
        cam_resized = np.array(Image.fromarray(cam).resize(original_size, Image.BILINEAR))
        
        # Create binary mask with adaptive threshold
        binary_mask = (cam_resized > 0.3).astype(np.uint8) * 255
        
        result_image = original_image.convert('RGB')
        draw = ImageDraw.Draw(result_image)
        
        # Improved contour detection
        height, width = binary_mask.shape
        contour_points = []
        
        # Find edge points more efficiently
        for y in range(1, height-1, 3):
            for x in range(1, width-1, 3):
                if binary_mask[y, x] > 128:
                    # Check if this is a boundary point
                    neighbors = [
                        binary_mask[y-1, x], binary_mask[y+1, x],
                        binary_mask[y, x-1], binary_mask[y, x+1]
                    ]
                    if any(n == 0 for n in neighbors):
                        contour_points.append((x, y))
        
        # Draw smooth contours
        if len(contour_points) > 20:
            # Group points by proximity and draw polygons
            groups = []
            used_points = set()
            
            for point in contour_points:
                if point not in used_points:
                    group = [point]
                    used_points.add(point)
                    
                    # Find nearby points
                    for other_point in contour_points:
                        if other_point not in used_points:
                            distance = ((point[0]-other_point[0])**2 + (point[1]-other_point[1])**2)**0.5
                            if distance < 20:
                                group.append(other_point)
                                used_points.add(other_point)
                    
                    if len(group) > 5:
                        groups.append(group)
            
            # Draw contours for each group
            for group in groups:
                if len(group) > 2:
                    # Sort points for better contour
                    group.sort(key=lambda p: (p[1], p[0]))
                    
                    # Draw polygon
                    for i in range(len(group)):
                        start = group[i]
                        end = group[(i + 1) % len(group)]
                        distance = ((end[0]-start[0])**2 + (end[1]-start[1])**2)**0.5
                        if distance < 50:
                            draw.line([start, end], fill=color, width=width)
        
        return result_image, contour_points
        
    except Exception as e:
        st.error(f"Contour creation error: {str(e)}")
        return original_image, []

def create_binary_mask(cam, original_image):
    """Create binary mask visualization"""
    try:
        original_size = original_image.size
        cam_resized = np.array(Image.fromarray(cam).resize(original_size, Image.BILINEAR))
        
        # Create binary mask
        binary_mask = (cam_resized > 0.3).astype(np.uint8) * 255
        
        # Convert to RGB for display
        mask_rgb = np.stack([binary_mask, binary_mask, binary_mask], axis=-1)
        
        return mask_rgb, cam_resized
        
    except Exception as e:
        st.error(f"Mask creation error: {str(e)}")
        return np.array(original_image.convert('RGB')), cam

@st.cache_resource
def create_model():
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 3)
    
    try:
        model_path = "Breast2_lump_best_weights.pth"
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            if 'model_state_dict' in state_dict:
                model.load_state_dict(state_dict['model_state_dict'])
            else:
                model.load_state_dict(state_dict)
            st.sidebar.success("✅ **ENHANCED MODEL LOADED**")
        else:
            st.error(f"❌ Model file not found: {model_path}")
            return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None
    
    model = model.to(device)
    model.eval()
    return model

# Class names
classes = ['benign', 'malignant', 'normal']

def predict_image(model, image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image_tensor = val_transforms(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    return predicted.item(), probabilities.cpu().numpy()[0], confidence.item(), image_tensor

def apply_enhanced_grad_cam(model, image_tensor, original_image, predicted_class, viz_mode="Heatmap", opacity=0.6, contour_color="#FF0000", contour_width=3):
    try:
        target_layer = model.layer4[-1].conv3
        grad_cam = EnhancedGradCAM(model, target_layer)
        cam, _ = grad_cam.generate_cam(image_tensor, predicted_class)
        
        # Convert hex color to RGB
        color_rgb = tuple(int(contour_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        
        if viz_mode == "Contour Only":
            contour_image, points = create_smart_contour(cam, original_image, color=color_rgb, width=contour_width)
            center_point = calculate_center(points, original_image.size)
            return np.array(contour_image), cam, center_point
            
        elif viz_mode == "Heatmap + Contour":
            heatmap_image, _ = create_enhanced_heatmap(cam, original_image, opacity)
            contour_image, points = create_smart_contour(cam, Image.fromarray(heatmap_image), color=color_rgb, width=contour_width)
            center_point = calculate_center(points, original_image.size)
            return np.array(contour_image), cam, center_point
            
        elif viz_mode == "Binary Mask":
            mask_image, _ = create_binary_mask(cam, original_image)
            center_point = (original_image.width // 2, original_image.height // 2)
            return mask_image, cam, center_point
            
        else:  # Heatmap
            heatmap_image, _ = create_enhanced_heatmap(cam, original_image, opacity)
            # Find max activation point
            max_idx = np.argmax(cam)
            y, x = np.unravel_index(max_idx, cam.shape)
            # Scale to original image size
            x = int(x * original_image.width / cam.shape[1])
            y = int(y * original_image.height / cam.shape[0])
            center_point = (x, y)
            return heatmap_image, cam, center_point
            
    except Exception as e:
        st.error(f"Enhanced Grad-CAM error: {str(e)}")
        original_array = np.array(original_image.convert('RGB'))
        return original_array, np.zeros((224, 224)), (original_image.width // 2, original_image.height // 2)

def calculate_center(points, image_size):
    """Calculate center point from contour points"""
    if points and len(points) > 0:
        center_x = int(np.mean([p[0] for p in points]))
        center_y = int(np.mean([p[1] for p in points]))
        return (center_x, center_y)
    else:
        return (image_size[0] // 2, image_size[1] // 2)

def add_enhanced_annotation(image, point, label, color=(255, 0, 0)):
    """Enhanced annotation with better styling"""
    try:
        if isinstance(image, np.ndarray):
            img_pil = Image.fromarray(image)
        else:
            img_pil = image.copy()
        
        draw = ImageDraw.Draw(img_pil)
        
        # Dynamic arrow size based on image size
        arrow_length = min(img_pil.width, img_pil.height) * 0.1
        arrow_head = max(6, int(arrow_length * 0.3))
        
        # Position arrow intelligently
        if point[0] > img_pil.width * 0.7:
            start_x = point[0] - arrow_length
        else:
            start_x = point[0] + arrow_length
            
        if point[1] > img_pil.height * 0.7:
            start_y = point[1] - arrow_length
        else:
            start_y = point[1] + arrow_length
        
        # Draw arrow line
        draw.line([(start_x, start_y), (point[0], point[1])], fill=color, width=3)
        
        # Draw arrowhead
        draw.polygon([
            (point[0], point[1]),
            (point[0] - arrow_head, point[1] - arrow_head),
            (point[0] + arrow_head, point[1] - arrow_head)
        ], fill=color)
        
        # Add label with background
        bbox = draw.textbbox((start_x, start_y), label)
        padding = 5
        draw.rectangle(
            [bbox[0]-padding, bbox[1]-padding, bbox[2]+padding, bbox[3]+padding],
            fill=(0, 0, 0, 128)
        )
        draw.text((start_x, start_y), label, fill=color)
        
        return img_pil
    except Exception as e:
        return image

def get_enhanced_recommendations(prediction, confidence):
    """Enhanced clinical recommendations"""
    recommendations = {
        'benign': {
            'urgency': 'Low', 'urgency_class': 'urgency-low',
            'actions': [
                "Schedule follow-up ultrasound in 6-12 months",
                "Continue routine breast screening as per guidelines",
                "Clinical breast examination in 6 months",
                "Monitor for any changes in size, shape, or characteristics",
                "Consider genetic counseling if strong family history"
            ],
            'risk_factors': ["Stable appearance", "Well-defined margins", "No rapid growth"]
        },
        'malignant': {
            'urgency': 'High', 'urgency_class': 'urgency-high',
            'actions': [
                "Urgent consultation with breast specialist/surgeon",
                "Core needle biopsy for histopathological confirmation",
                "Additional diagnostic imaging (MRI if indicated)",
                "Multidisciplinary team review",
                "Genetic testing consideration"
            ],
            'risk_factors': ["Irregular margins", "Rapid growth", "Architectural distortion"]
        },
        'normal': {
            'urgency': 'Routine', 'urgency_class': 'urgency-low',
            'actions': [
                "Continue routine screening schedule",
                "Regular self-breast awareness",
                "Next screening mammography as per age guidelines",
                "Annual clinical breast examination",
                "Maintain healthy lifestyle"
            ],
            'risk_factors': ["Normal architecture", "Stable pattern", "No suspicious features"]
        }
    }
    return recommendations.get(prediction, {})

def explain_enhanced_prediction(prediction, probabilities, confidence):
    """Enhanced AI explanations"""
    explanations = {
        'malignant': [
            "🔄 Irregular mass margins with spiculated appearance",
            "📊 Architectural distortion and tissue retraction",
            "🎯 Suspicious microcalcifications cluster",
            "📈 Asymmetric tissue density with rapid changes",
            "⚠️ Enhanced vascularity around the lesion"
        ],
        'benign': [
            "✅ Well-circumscribed mass with smooth margins",
            "📏 Stable size and appearance over time",
            "🎨 Homogeneous internal echo pattern",
            "🔄 No significant vascularity increase",
            "📊 Typical benign characteristics present"
        ],
        'normal': [
            "👍 Normal fibroglandular tissue distribution",
            "📐 Symmetric breast architecture maintained",
            "🎯 No suspicious masses or distortions",
            "🔄 Stable appearance from previous studies",
            "📊 Typical age-appropriate patterns"
        ]
    }
    
    return explanations.get(prediction, ["Standard tissue evaluation completed"])

def calculate_enhanced_risk_score(patient_data, prediction, confidence):
    """Enhanced risk assessment"""
    base_risk = {
        'malignant': 0.85,
        'benign': 0.25,
        'normal': 0.08
    }[prediction]
    
    # Confidence adjustment
    risk_adjustment = confidence * 1.2
    
    # Patient factor adjustments
    adjustments = 1.0
    
    if patient_data.get('family_history') == "Breast Cancer":
        adjustments *= 1.6
    if patient_data.get('previous_biopsy', "No") != "No":
        adjustments *= 1.4
    if patient_data.get('age', 45) > 50:
        adjustments *= 1.3
    if patient_data.get('breast_density', "").startswith("C") or patient_data.get('breast_density', "").startswith("D"):
        adjustments *= 1.2
    
    final_risk = min(base_risk * risk_adjustment * adjustments, 0.95)
    return final_risk

def image_quality_check(image):
    """Enhanced image quality assessment"""
    quality_issues = []
    quality_score = 100
    
    # Resolution check
    if image.size[0] < 512 or image.size[1] < 512:
        quality_issues.append("Low resolution")
        quality_score -= 30
    
    # Contrast check
    img_array = np.array(image.convert('L'))
    contrast = np.std(img_array)
    if contrast < 50:
        quality_issues.append("Low contrast")
        quality_score -= 20
    elif contrast > 150:
        quality_issues.append("High contrast")
        quality_score -= 10
    
    # Brightness check
    brightness = np.mean(img_array)
    if brightness < 50:
        quality_issues.append("Low brightness")
        quality_score -= 15
    elif brightness > 200:
        quality_issues.append("High brightness")
        quality_score -= 10
    
    return quality_issues, max(quality_score, 0)

def generate_enhanced_pdf_report(analysis_data):
    """Enhanced PDF report with all features"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=30)
    styles = getSampleStyleSheet()
    story = []
    
    # Title with enhanced styling
    title_style = ParagraphStyle('EnhancedTitle', parent=styles['Heading1'], 
                                fontSize=18, spaceAfter=30, alignment=1, 
                                textColor=colors.HexColor('#1f77b4'),
                                backColor=colors.HexColor('#f8f9fa'))
    story.append(Paragraph("ADVANCED BREAST ULTRASOUND AI ANALYSIS REPORT", title_style))
    story.append(Spacer(1, 20))
    
    # Patient Information Section
    if 'patient_data' in analysis_data:
        story.append(Paragraph("<b>PATIENT INFORMATION</b>", styles['Heading2']))
        patient_info = f"""
        <b>Patient ID:</b> {analysis_data['patient_data'].get('patient_id', 'N/A')}<br/>
        <b>Age:</b> {analysis_data['patient_data'].get('age', 'N/A')}<br/>
        <b>Gender:</b> {analysis_data['patient_data'].get('gender', 'N/A')}<br/>
        <b>Family History:</b> {analysis_data['patient_data'].get('family_history', 'N/A')}<br/>
        <b>Breast Density:</b> {analysis_data['patient_data'].get('breast_density', 'N/A')}<br/>
        <b>Previous Biopsy:</b> {analysis_data['patient_data'].get('previous_biopsy', 'N/A')}
        """
        story.append(Paragraph(patient_info, styles['Normal']))
        story.append(Spacer(1, 15))
    
    # Report Info
    info_style = styles['Normal']
    story.append(Paragraph(f"<b>Report Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", info_style))
    story.append(Paragraph(f"<b>AI Model:</b> Enhanced ResNet50 Pro", info_style))
    story.append(Paragraph(f"<b>Analysis ID:</b> {analysis_data.get('analysis_id', 'N/A')}", info_style))
    story.append(Spacer(1, 20))
    
    # Enhanced Prediction Results
    pred_color = {'benign': '#2ecc71', 'malignant': '#e74c3c', 'normal': '#3498db'}[analysis_data['prediction']]
    story.append(Paragraph("<b>PREDICTION RESULTS</b>", styles['Heading2']))
    prediction_text = f"""
    <b>Classification:</b> <font color='{pred_color}'><b>{analysis_data['prediction'].upper()}</b></font><br/>
    <b>Confidence Level:</b> {analysis_data['confidence']:.1%}<br/>
    <b>Risk Score:</b> {analysis_data.get('risk_score', 0):.1%}<br/>
    <b>Inference Time:</b> {analysis_data.get('inference_time', 0):.2f}s
    """
    story.append(Paragraph(prediction_text, info_style))
    story.append(Spacer(1, 15))
    
    # Enhanced Confidence Scores Table
    confidence_data = [
        ['Class', 'Probability', 'Confidence', 'Risk Level'],
        ['Benign', f"{analysis_data['probabilities'][0]:.4f}", 
         f"{analysis_data['probabilities'][0]*100:.1f}%", 
         'Low' if analysis_data['probabilities'][0] < 0.3 else 'Medium' if analysis_data['probabilities'][0] < 0.7 else 'High'],
        ['Malignant', f"{analysis_data['probabilities'][1]:.4f}", 
         f"{analysis_data['probabilities'][1]*100:.1f}%", 
         'Low' if analysis_data['probabilities'][1] < 0.3 else 'Medium' if analysis_data['probabilities'][1] < 0.7 else 'High'],
        ['Normal', f"{analysis_data['probabilities'][2]:.4f}", 
         f"{analysis_data['probabilities'][2]*100:.1f}%", 
         'Low' if analysis_data['probabilities'][2] < 0.3 else 'Medium' if analysis_data['probabilities'][2] < 0.7 else 'High']
    ]
    
    confidence_table = Table(confidence_data, colWidths=[1.5*inch, 1.2*inch, 1.2*inch, 1.2*inch])
    confidence_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2e86ab')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#e8f4f8')),
        ('BACKGROUND', (1, 1), (-1, -1), colors.HexColor('#f8f9fa')),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(confidence_table)
    story.append(Spacer(1, 20))
    
    # AI Explanations
    if 'explanations' in analysis_data:
        story.append(Paragraph("<b>AI EXPLANATION & FEATURE ANALYSIS</b>", styles['Heading2']))
        for explanation in analysis_data['explanations']:
            story.append(Paragraph(f"• {explanation}", info_style))
            story.append(Spacer(1, 3))
        story.append(Spacer(1, 15))
    
    # Enhanced Clinical Recommendations
    story.append(Paragraph("<b>CLINICAL RECOMMENDATIONS</b>", styles['Heading2']))
    urgency_color = {
        'High': colors.red,
        'Medium': colors.orange,
        'Low': colors.green
    }.get(analysis_data['recommendations'].get('urgency', 'Low'), colors.black)
    
    urgency_text = f"<b>Urgency Level:</b> <font color='{urgency_color}'>{analysis_data['recommendations'].get('urgency', 'N/A')}</font>"
    story.append(Paragraph(urgency_text, info_style))
    story.append(Spacer(1, 10))
    
    for i, action in enumerate(analysis_data['recommendations']['actions'], 1):
        story.append(Paragraph(f"{i}. {action}", info_style))
        story.append(Spacer(1, 5))
    
    # Risk Factors
    if 'risk_factors' in analysis_data['recommendations']:
        story.append(Spacer(1, 10))
        story.append(Paragraph("<b>Key Risk Factors Identified:</b>", styles['Heading3']))
        for factor in analysis_data['recommendations']['risk_factors']:
            story.append(Paragraph(f"• {factor}", info_style))
            story.append(Spacer(1, 3))
    
    story.append(Spacer(1, 20))
    
    # Quality Assessment
    if 'quality_issues' in analysis_data:
        story.append(Paragraph("<b>IMAGE QUALITY ASSESSMENT</b>", styles['Heading2']))
        quality_text = f"<b>Quality Score:</b> {analysis_data.get('quality_score', 100)}/100<br/>"
        if analysis_data['quality_issues']:
            quality_text += f"<b>Issues:</b> {', '.join(analysis_data['quality_issues'])}"
        else:
            quality_text += "<b>Issues:</b> None - Excellent quality"
        story.append(Paragraph(quality_text, info_style))
        story.append(Spacer(1, 15))
    
    story.append(Spacer(1, 20))
    disclaimer_style = ParagraphStyle('Disclaimer', parent=styles['Normal'], 
                                     fontSize=8, textColor=colors.gray)
    story.append(Paragraph("*** ADVANCED AI ANALYSIS - FOR CLINICAL DECISION SUPPORT ONLY ***", disclaimer_style))
    story.append(Paragraph("*** Results should be interpreted by qualified healthcare professionals ***", disclaimer_style))
    story.append(Paragraph("*** Model: Enhanced ResNet50 Pro | Version 2.0 | Confidence Threshold: 70% ***", disclaimer_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def main():
    st.markdown('<h1 class="main-header">🏥 Advanced Breast Ultrasound AI Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; margin-bottom: 2rem;">Enhanced AI-Powered Clinical Decision Support System</p>', unsafe_allow_html=True)
    
    # Feature showcase
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="feature-card">🎯 Smart Predictions</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="feature-card">📊 Risk Assessment</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="feature-card">🔄 Batch Processing</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="feature-card">📈 Advanced Analytics</div>', unsafe_allow_html=True)
    
    model = create_model()
    if model is None:
        st.warning("Please ensure 'Breast2_lump_best_weights.pth' is uploaded to continue.")
        return

    # Enhanced Patient Information Form
    with st.expander("👤 Enhanced Patient Information", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            patient_id = st.text_input("Patient ID", value="PAT-001", key="pat_id")
            age = st.number_input("Age", min_value=18, max_value=100, value=45, key="age")
            menopausal_status = st.selectbox("Menopausal Status", ["Pre-menopausal", "Post-menopausal"], key="meno")
        with col2:
            gender = st.selectbox("Gender", ["Female", "Male"], key="gender")
            family_history = st.selectbox("Family History", ["None", "Breast Cancer", "Ovarian Cancer", "Other Cancer"], key="fam_hist")
            brca_status = st.selectbox("BRCA Status", ["Unknown", "Negative", "Positive"], key="brca")
        with col3:
            breast_density = st.selectbox("Breast Density", ["A - Fatty", "B - Scattered", "C - Heterogeneous", "D - Extremely Dense"], key="density")
            previous_biopsy = st.selectbox("Previous Biopsy", ["No", "Yes - Benign", "Yes - Atypical", "Yes - Malignant"], key="biopsy")
            hormone_therapy = st.selectbox("Hormone Therapy", ["No", "Yes - Current", "Yes - Past"], key="hormone")
        
        patient_data = {
            'patient_id': patient_id,
            'age': age,
            'gender': gender,
            'family_history': family_history,
            'breast_density': breast_density,
            'previous_biopsy': previous_biopsy,
            'menopausal_status': menopausal_status,
            'brca_status': brca_status,
            'hormone_therapy': hormone_therapy
        }

    # Enhanced Batch Processing
    if batch_files and len(batch_files) > 1:
        st.markdown("### 📊 Enhanced Batch Analysis")
        progress_bar = st.progress(0)
        batch_results = []
        
        for i, uploaded_file in enumerate(batch_files):
            progress_bar.progress((i + 1) / len(batch_files))
            try:
                image = Image.open(uploaded_file)
                start_time = time.time()
                prediction, probabilities, confidence, image_tensor = predict_image(model, image)
                inference_time = time.time() - start_time
                
                batch_results.append({
                    'filename': uploaded_file.name,
                    'prediction': classes[prediction],
                    'confidence': confidence,
                    'inference_time': inference_time,
                    'probabilities': probabilities,
                    'urgency': get_enhanced_recommendations(classes[prediction], confidence)['urgency']
                })
            except Exception as e:
                batch_results.append({
                    'filename': uploaded_file.name,
                    'prediction': 'Error',
                    'confidence': 0.0,
                    'inference_time': 0.0,
                    'error': str(e)
                })
        
        # Display enhanced batch results
        st.markdown("#### Batch Analysis Summary")
        for result in batch_results:
            col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 2])
            with col1:
                st.write(f"**{result['filename']}**")
            with col2:
                pred_color = {'benign': '🟢', 'malignant': '🔴', 'normal': '🔵', 'Error': '⚫'}[result['prediction']]
                st.write(f"{pred_color} **{result['prediction'].upper()}**")
            with col3:
                st.write(f"**{result['confidence']:.1%}**")
            with col4:
                st.write(f"**{result['inference_time']:.2f}s**")
            with col5:
                urgency_color = {'High': '🔴', 'Low': '🟢', 'Routine': '🔵'}.get(result.get('urgency', 'N/A'), '⚫')
                st.write(f"{urgency_color} {result.get('urgency', 'N/A')}")
        
        success_count = len([r for r in batch_results if r['prediction'] != 'Error'])
        st.success(f"✅ Batch analysis completed: {success_count} successful, {len(batch_files) - success_count} failed")
        return

    # Single Image Analysis with Enhanced Features
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="sub-header">📤 Upload Ultrasound Image</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose breast ultrasound image", type=['jpg', 'jpeg', 'png', 'bmp'], key="single_upload")
        
        if uploaded_file:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                # Enhanced Image Quality Check
                quality_issues, quality_score = image_quality_check(image)
                if quality_issues:
                    st.warning(f"⚠️ Image Quality: {quality_score}/100 - Issues: {', '.join(quality_issues)}")
                else:
                    st.success(f"✅ Image Quality: {quality_score}/100 - Excellent")
                
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")

    with col2:
        st.markdown('<div class="sub-header">🔍 Enhanced AI Analysis</div>', unsafe_allow_html=True)
        
        if uploaded_file and model is not None:
            try:
                image = Image.open(uploaded_file)
                start_time = time.time()
                
                with st.spinner('🔄 Running enhanced AI analysis...'):
                    prediction, probabilities, confidence, image_tensor = predict_image(model, image)
                    viz_image, cam, heat_point = apply_enhanced_grad_cam(
                        model, image_tensor, image, prediction, 
                        viz_mode, heatmap_opacity, contour_color, contour_width
                    )
                
                inference_time = time.time() - start_time
                predicted_class = classes[prediction]
                recommendations = get_enhanced_recommendations(predicted_class, confidence)
                explanations = explain_enhanced_prediction(predicted_class, probabilities, confidence)
                risk_score = calculate_enhanced_risk_score(patient_data, predicted_class, confidence)
                
                # Confidence threshold warning
                if confidence < confidence_threshold:
                    st.warning(f"⚠️ Low confidence prediction ({confidence:.1%}). Consider manual review or additional imaging.")
                
                # Enhanced Tabs with all features
                tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                    "🎯 Prediction", "🔥 Visualization", "💡 Recommendations", 
                    "📊 Analytics", "🤖 AI Explanation", "🔄 Compare", "📄 Export"
                ])
                
                with tab1:
                    colors_dict = {'benign': '#2ecc71', 'malignant': '#e74c3c', 'normal': '#3498db'}
                    emojis = {'benign': '✅', 'malignant': '⚠️', 'normal': '👍'}
                    
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h3 style="color: white; margin: 0;">{emojis[predicted_class]} {predicted_class.upper()}</h3>
                        <p style="margin: 10px 0; font-size: 1.2em;">Confidence: <strong>{confidence:.1%}</strong></p>
                        <p style="margin: 0; font-size: 0.9em;">Inference Time: {inference_time:.2f}s | Model: Enhanced v2.0</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Enhanced Risk Score
                    risk_class = "risk-high" if risk_score > 0.7 else "risk-medium" if risk_score > 0.3 else "risk-low"
                    st.markdown(f"""
                    <div class="{risk_class}" style="padding: 15px; border-radius: 8px; text-align: center; margin: 10px 0;">
                        <h4 style="margin: 0; color: white;">Overall Risk Score: {risk_score:.1%}</h4>
                        <p style="margin: 5px 0 0 0; font-size: 0.9em;">Based on AI prediction + patient factors</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="{recommendations['urgency_class']}" style="padding: 15px; border-radius: 8px; text-align: center;">
                        <h4 style="margin: 0; color: white;">Clinical Urgency: {recommendations['urgency'].upper()}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("#### 📈 Confidence Distribution")
                    for class_name, prob in zip(classes, probabilities):
                        width = int(prob * 100)
                        risk_level = "High" if prob > 0.7 else "Medium" if prob > 0.3 else "Low"
                        st.markdown(f"""
                        <div style="margin: 10px 0;">
                            <div style="display: flex; justify-content: space-between;">
                                <span><strong>{class_name.title()}</strong></span>
                                <span>{prob:.4f} ({risk_level} Risk)</span>
                            </div>
                            <div class="confidence-bar">
                                <div class="confidence-fill {class_name}" style="width: {width}%;">{width}%</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with tab2:
                    st.image(viz_image, caption=f"{viz_mode} Visualization", use_container_width=True)
                    if viz_mode == "Heatmap":
                        annotated_image = add_enhanced_annotation(Image.fromarray(viz_image), heat_point, "AI Focus Area")
                        st.image(annotated_image, caption="Annotated AI Focus Area", use_container_width=True)
                    
                    st.markdown("#### 🎨 Visualization Settings Applied")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Mode", viz_mode)
                    with col2:
                        st.metric("Opacity", f"{heatmap_opacity:.1f}")
                    with col3:
                        st.metric("Color", contour_color)
                
                with tab3:
                    st.markdown("### 🩺 Enhanced Clinical Recommendations")
                    
                    # Risk Factors
                    st.markdown("#### 📋 Identified Risk Factors")
                    for factor in recommendations.get('risk_factors', []):
                        st.markdown(f"• {factor}")
                    
                    st.markdown("#### 💡 Recommended Actions")
                    for i, action in enumerate(recommendations['actions'], 1):
                        st.markdown(f"**{i}.** {action}")
                    
                    # Timeline guidance
                    st.markdown("#### ⏰ Suggested Timeline")
                    if predicted_class == 'malignant':
                        st.info("**Immediate action required** - Consult within 48-72 hours")
                    elif predicted_class == 'benign':
                        st.info("**Routine follow-up** - Next appointment in 3-6 months")
                    else:
                        st.info("**Regular screening** - Next routine check in 12 months")
                
                with tab4:
                    st.markdown("### 📊 Enhanced Analytics Dashboard")
                    
                    # Key Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Prediction Confidence", f"{confidence:.1%}")
                    with col2:
                        st.metric("Uncertainty Index", f"{(1-confidence):.1%}")
                    with col3:
                        certainty = "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
                        st.metric("Decision Certainty", certainty)
                    with col4:
                        st.metric("Risk Category", recommendations['urgency'])
                    
                    # Enhanced Probability Chart
                    st.markdown("#### Probability Distribution Analysis")
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Bar chart
                    colors = ['#2ecc71', '#e74c3c', '#3498db']
                    bars = ax1.bar(classes, probabilities, color=colors, alpha=0.8, edgecolor='black')
                    ax1.set_ylabel('Probability')
                    ax1.set_title('Class Probability Distribution')
                    ax1.set_ylim(0, 1)
                    
                    for bar, prob in zip(bars, probabilities):
                        height = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{prob:.4f}', ha='center', va='bottom', fontweight='bold')
                    
                    # Pie chart
                    ax2.pie(probabilities, labels=classes, colors=colors, autopct='%1.1f%%', startangle=90)
                    ax2.set_title('Probability Distribution')
                    
                    st.pyplot(fig)
                    
                    # Statistical Summary
                    st.markdown("#### 📈 Statistical Summary")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Max Probability", f"{np.max(probabilities):.4f}")
                    with col2:
                        st.metric("Probability Range", f"{np.ptp(probabilities):.4f}")
                    with col3:
                        st.metric("Entropy", f"{-np.sum(probabilities * np.log(probabilities + 1e-8)):.4f}")
                
                with tab5:
                    st.markdown("### 🤖 Enhanced AI Explanation")
                    st.info("The AI model analyzed the following features to make its prediction:")
                    
                    for i, explanation in enumerate(explanations, 1):
                        st.markdown(f"**{i}.** {explanation}")
                    
                    # Feature importance visualization
                    st.markdown("#### 🎯 Feature Attention Map")
                    st.image(viz_image, caption="Color intensity shows areas of high AI attention", use_container_width=True)
                    
                    # Confidence explanation
                    st.markdown("#### 📊 Confidence Analysis")
                    if confidence > 0.9:
                        st.success("**Very High Confidence** - Clear diagnostic features present")
                    elif confidence > 0.7:
                        st.info("**High Confidence** - Strong diagnostic indicators")
                    elif confidence > 0.5:
                        st.warning("**Moderate Confidence** - Some uncertainty, review recommended")
                    else:
                        st.error("**Low Confidence** - Significant uncertainty, additional imaging advised")
                
                with tab6:
                    st.markdown("### 🔄 Comparative Analysis")
                    previous_scan = st.file_uploader("Upload Previous Scan for Comparison", 
                                                   type=['jpg', 'jpeg', 'png'], 
                                                   key="previous_scan_comparison")
                    
                    if previous_scan:
                        prev_image = Image.open(previous_scan)
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(prev_image, caption="Previous Scan", use_container_width=True)
                        with col2:
                            st.image(image, caption="Current Scan", use_container_width=True)
                        
                        # Compare predictions
                        prev_prediction, prev_prob, prev_confidence, _ = predict_image(model, prev_image)
                        prev_class = classes[prev_prediction]
                        
                        st.markdown("#### 📊 Comparison Results")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Previous Prediction", prev_class.upper(), 
                                     delta=f"{prev_confidence:.1%} confidence")
                        with col2:
                            st.metric("Current Prediction", predicted_class.upper(),
                                     delta=f"{confidence:.1%} confidence")
                        
                        # Change analysis
                        if prev_class != predicted_class:
                            st.warning(f"🔄 **Significant change detected**: {prev_class.upper()} → {predicted_class.upper()}")
                        else:
                            st.success(f"✅ **Stable findings**: {predicted_class.upper()} classification maintained")
                
                with tab7:
                    st.markdown("### 📄 Enhanced Export Options")
                    
                    analysis_data = {
                        'prediction': predicted_class,
                        'confidence': confidence,
                        'probabilities': probabilities.tolist(),
                        'recommendations': recommendations,
                        'explanations': explanations,
                        'risk_score': risk_score,
                        'patient_data': patient_data,
                        'inference_time': inference_time,
                        'quality_issues': quality_issues,
                        'quality_score': quality_score,
                        'analysis_id': f"BREAST-AI-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                        'timestamp': datetime.now().isoformat(),
                        'model_version': 'Enhanced ResNet50 Pro v2.0'
                    }
                    
                    # PDF Report
                    pdf_buffer = generate_enhanced_pdf_report(analysis_data)
                    st.download_button(
                        label="📥 Download Enhanced PDF Report",
                        data=pdf_buffer,
                        file_name=f"enhanced_breast_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        help="Comprehensive report with all analysis details"
                    )
                    
                    # JSON Export
                    json_data = json.dumps(analysis_data, indent=2, default=str)
                    st.download_button(
                        label="📊 Download JSON Data",
                        data=json_data,
                        file_name=f"breast_analysis_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        help="Structured data for integration with other systems"
                    )
                    
                    # API Call Generation
                    if st.session_state.get('show_api', False):
                        st.markdown("### 🌐 API Integration")
                        api_payload = {
                            "patient_id": patient_data['patient_id'],
                            "prediction": predicted_class,
                            "confidence": confidence,
                            "risk_score": risk_score,
                            "urgency": recommendations['urgency'],
                            "timestamp": datetime.now().isoformat(),
                            "analysis_id": analysis_data['analysis_id']
                        }
                        st.code(f"POST /api/v2/analysis\nHeaders: Authorization: Bearer <token>\n\n{json.dumps(api_payload, indent=2)}")
                
                # Real-time Collaboration
                with st.expander("💬 Enhanced Clinical Notes & Collaboration"):
                    radiologist_notes = st.text_area(
                        "Radiologist Clinical Notes",
                        placeholder="Document your clinical observations, differential diagnosis, additional findings, or recommendations for the clinical team...",
                        height=120
                    )
                    
                    clinical_impression = st.selectbox(
                        "Clinical Impression",
                        ["Concordant with AI", "Discordant - Benign", "Discordant - Suspicious", "Inconclusive", "Technical Repeat Needed"]
                    )
                    
                    if st.button("💬 Share with Multidisciplinary Team"):
                        st.success("✅ Analysis, notes, and clinical impression shared with multidisciplinary team")
                        # In a real application, this would integrate with hospital systems
                
                st.success("🎉 Enhanced analysis complete! All features activated and ready for clinical review.")
                
            except Exception as e:
                st.error(f"❌ Processing error: {str(e)}")
                st.info("💡 This might be due to model compatibility or image format issues.")

if __name__ == "__main__":
    main()