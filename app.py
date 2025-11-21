import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Set page configuration
st.set_page_config(
    page_title="Kidney Disease Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid;
    }
    .cyst-pred { border-left-color: #ff6b6b; background-color: #ffebee; }
    .normal-pred { border-left-color: #4caf50; background-color: #e8f5e8; }
    .stone-pred { border-left-color: #ff9800; background-color: #fff3e0; }
    .tumor-pred { border-left-color: #9c27b0; background-color: #f3e5f5; }
    .confidence-bar {
        height: 20px;
        background-color: #e0e0e0;
        border-radius: 10px;
        margin: 5px 0;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-weight: bold;
    }
    .gradcam-container {
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
        gap: 20px;
        margin: 20px 0;
    }
    .gradcam-image {
        text-align: center;
        margin: 10px;
    }
</style>
""", unsafe_allow_html=True)

def reassemble_model_chunks():
    """Reassemble model chunks into a single file"""
    try:
        # Folder containing your chunks
        folder = "model_chunks"  # e.g., the folder with your .pth.000, .pth.001, ...

        # Output file
        output_file = "Bestresnet50_kidney_best_weights.pth"

        # Check if output file already exists
        if os.path.exists(output_file):
            st.sidebar.success("✅ Model file already exists")
            return output_file

        # Check if chunks folder exists
        if not os.path.exists(folder):
            st.sidebar.error(f"❌ Chunks folder '{folder}' not found")
            return None

        # List and sort chunks
        chunks = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.startswith("Bestresnet50_kidney_best_weights.pth.")])
        
        if not chunks:
            st.sidebar.error("❌ No model chunks found")
            return None

        st.sidebar.info(f"🔍 Found {len(chunks)} model chunks")

        # Reassemble
        with st.sidebar.expander("🔄 Model Assembly Progress"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            with open(output_file, "wb") as outfile:
                for i, chunk in enumerate(chunks):
                    status_text.text(f"Assembling chunk {i+1}/{len(chunks)}: {os.path.basename(chunk)}")
                    with open(chunk, "rb") as infile:
                        outfile.write(infile.read())
                    progress_bar.progress((i + 1) / len(chunks))

            status_text.text("✅ Model assembly completed!")
        
        st.sidebar.success(f"✅ Model reassembled successfully: {output_file}")
        return output_file

    except Exception as e:
        st.sidebar.error(f"❌ Error reassembling model: {str(e)}")
        return None

class GradCAM:
    """Grad-CAM implementation for ResNet50"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook the gradients and activations
        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_full_backward_hook(self.save_gradients)
    
    def save_activations(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class=None):
        # Forward pass
        model_output = self.model(input_image)
        
        if target_class is None:
            target_class = torch.argmax(model_output, dim=1)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        one_hot = torch.zeros_like(model_output)
        one_hot[0][target_class] = 1
        model_output.backward(gradient=one_hot)
        
        # Global average pooling of gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight the activations by corresponding gradients
        for i in range(self.activations.size(1)):
            self.activations[:, i, :, :] *= pooled_gradients[i]
        
        # Average the weighted activations
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = torch.clamp(heatmap, min=0)  # ReLU
        
        # Normalize heatmap
        heatmap -= heatmap.min()
        heatmap /= heatmap.max()
        
        return heatmap.cpu().numpy()

class KidneyDiseaseClassifier:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = ['Cyst', 'Normal', 'Stone', 'Tumor']
        self.model = self.load_model(model_path)
        self.transform = self.get_transforms()
        
        # Initialize Grad-CAM
        if self.model is not None:
            self.gradcam = GradCAM(self.model, self.model.layer4[-1])
        
    def load_model(self, model_path):
        """Load the trained ResNet50 model"""
        try:
            # Use ResNet50 to match your trained weights
            model = models.resnet50(weights=None)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 4)  # 4 classes
            
            if model_path and os.path.exists(model_path):
                # Load the state dict
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                
                # Load the weights
                model.load_state_dict(state_dict)
                st.success("✅ ResNet50 model loaded successfully!")
            else:
                st.warning("⚠️ Model file not found. Using untrained model.")
                
            model.to(self.device)
            model.eval()
            return model
            
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            return None
    
    def get_transforms(self):
        """Define image transformations"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def predict(self, image):
        """Make prediction on a single image"""
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply transformations
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted_idx = torch.max(probabilities, 0)
                
            predicted_class = self.class_names[predicted_idx.item()]
            confidence_value = confidence.item()
            
            # Get all class probabilities
            all_probs = {
                self.class_names[i]: probabilities[i].item() 
                for i in range(len(self.class_names))
            }
            
            return predicted_class, confidence_value, all_probs, input_tensor, predicted_idx
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            return None, None, None, None, None
    
    def generate_gradcam_heatmap(self, input_tensor, target_class, original_image):
        """Generate Grad-CAM heatmap overlay"""
        try:
            # Generate CAM
            cam = self.gradcam.generate_cam(input_tensor, target_class)
            
            # Convert CAM to heatmap
            cam = cv2.resize(cam, (224, 224))
            cam = np.uint8(255 * cam)
            heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
            
            # Convert original image to numpy
            original_np = np.array(original_image.resize((224, 224)))
            
            # Blend heatmap with original image
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            superimposed_img = heatmap * 0.4 + original_np * 0.6
            superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
            
            return superimposed_img, heatmap, cam
            
        except Exception as e:
            st.error(f"Error generating Grad-CAM: {str(e)}")
            return None, None, None

def plot_probability_chart(probabilities):
    """Create a bar chart of class probabilities"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    classes = list(probabilities.keys())
    probs = list(probabilities.values())
    colors = ['#ff6b6b', '#4caf50', '#ff9800', '#9c27b0']
    
    bars = ax.bar(classes, probs, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_xlabel('Classes', fontsize=12)
    ax.set_title('Prediction Probabilities', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_gradcam_comparison(original_img, heatmap_img, superimposed_img):
    """Plot comparison of original, heatmap, and overlay"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(heatmap_img)
    axes[1].set_title('Attention Heatmap', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(superimposed_img)
    axes[2].set_title('Grad-CAM Overlay', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig

def get_class_info(predicted_class):
    """Return information about each kidney condition"""
    info = {
        'Cyst': {
            'description': 'Fluid-filled sacs in the kidneys',
            'symptoms': 'Usually asymptomatic, may cause pain if large',
            'treatment': 'Monitoring, drainage if symptomatic',
            'severity': 'Usually benign',
            'attention_areas': 'Fluid-filled regions, cyst walls'
        },
        'Normal': {
            'description': 'Healthy kidney with no abnormalities',
            'symptoms': 'No symptoms',
            'treatment': 'Regular checkups recommended',
            'severity': 'Normal',
            'attention_areas': 'Uniform tissue, clear boundaries'
        },
        'Stone': {
            'description': 'Hard deposits of minerals and salts',
            'symptoms': 'Severe pain, blood in urine, nausea',
            'treatment': 'Medication, lithotripsy, surgery',
            'severity': 'Can be painful but treatable',
            'attention_areas': 'Bright echogenic foci, shadowing'
        },
        'Tumor': {
            'description': 'Abnormal growth in kidney tissue',
            'symptoms': 'Blood in urine, pain, lump in abdomen',
            'treatment': 'Surgery, targeted therapy, immunotherapy',
            'severity': 'Requires immediate medical attention',
            'attention_areas': 'Mass lesions, irregular borders'
        }
    }
    return info.get(predicted_class, {})

def main():
    # Header
    st.markdown('<h1 class="main-header">🩺 Kidney Disease Classification with Grad-CAM</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        "This AI tool classifies kidney ultrasound images and highlights important regions using Grad-CAM visualization."
    )
    
    st.sidebar.title("Model Information")
    st.sidebar.text("Architecture: ResNet50")
    st.sidebar.text("Classes: 4 (Cyst, Normal, Stone, Tumor)")
    st.sidebar.text("Visualization: Grad-CAM")
    st.sidebar.text("Training Accuracy: ~98%")
    
    st.sidebar.title("Grad-CAM Explanation")
    st.sidebar.markdown("""
    **Red areas** = High attention (model focus)  
    **Blue areas** = Low attention  
    The heatmap shows where the model looks to make its decision.
    """)
    
    # Model assembly section
    st.sidebar.title("🔧 Model Assembly")
    
    if st.sidebar.button("🔄 Reassemble Model from Chunks", use_container_width=True):
        reassemble_model_chunks()
    
    # Model file selection
    st.sidebar.title("Model Settings")
    model_path = st.sidebar.text_input(
        "Model file path",
        value="Bestresnet50_kidney_best_weights.pth",
        help="Path to your trained model weights file"
    )
    
    # Check if model file exists
    if model_path and os.path.exists(model_path):
        st.sidebar.success(f"✅ Model file found: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
    else:
        st.sidebar.warning("⚠️ Model file not found. Please reassemble from chunks or provide correct path.")
    
    # Initialize classifier
    classifier = KidneyDiseaseClassifier(model_path)
    
    # Disclaimer
    st.sidebar.warning(
        "⚠️ **Disclaimer:** This tool is for educational purposes only. "
        "Always consult healthcare professionals for medical diagnosis."
    )
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a kidney ultrasound image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Supported formats: JPG, JPEG, PNG, BMP"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            original_image = Image.open(uploaded_file)
            st.image(original_image, caption="Uploaded Image", use_column_width=True)
            
            # Add a predict button
            if st.button("🔍 Analyze Image with Grad-CAM", type="primary", use_container_width=True):
                if classifier.model is None:
                    st.error("❌ Model not loaded properly. Please check your model file.")
                else:
                    with st.spinner("Analyzing image with Grad-CAM..."):
                        # Make prediction
                        predicted_class, confidence, all_probs, input_tensor, predicted_idx = classifier.predict(original_image)
                        
                        if predicted_class:
                            # Generate Grad-CAM
                            superimposed_img, heatmap_img, cam = classifier.generate_gradcam_heatmap(
                                input_tensor, predicted_idx, original_image
                            )
                            
                            # Display results
                            st.subheader("📊 Prediction Results")
                            
                            # Color-coded prediction box
                            class_colors = {
                                'Cyst': 'cyst-pred',
                                'Normal': 'normal-pred', 
                                'Stone': 'stone-pred',
                                'Tumor': 'tumor-pred'
                            }
                            
                            st.markdown(f"""
                            <div class="prediction-box {class_colors.get(predicted_class, '')}">
                                <h3>Prediction: {predicted_class}</h3>
                                <p><strong>Confidence:</strong> {confidence:.1%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Confidence visualization
                            st.subheader("Confidence Levels")
                            for class_name, prob in all_probs.items():
                                color = {
                                    'Cyst': '#ff6b6b',
                                    'Normal': '#4caf50', 
                                    'Stone': '#ff9800',
                                    'Tumor': '#9c27b0'
                                }.get(class_name, '#666666')
                                
                                st.write(f"**{class_name}**")
                                st.markdown(f"""
                                <div class="confidence-bar">
                                    <div class="confidence-fill" style="width: {prob*100}%; background-color: {color};">
                                        {prob:.1%}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Store variables for second column
                            st.session_state.predicted_class = predicted_class
                            st.session_state.confidence = confidence
                            st.session_state.all_probs = all_probs
                            st.session_state.superimposed_img = superimposed_img
                            st.session_state.heatmap_img = heatmap_img
                            st.session_state.original_image = original_image
                            st.session_state.uploaded_file_name = uploaded_file.name
    
    with col2:
        if hasattr(st.session_state, 'predicted_class'):
            predicted_class = st.session_state.predicted_class
            confidence = st.session_state.confidence
            all_probs = st.session_state.all_probs
            superimposed_img = st.session_state.superimposed_img
            heatmap_img = st.session_state.heatmap_img
            original_image = st.session_state.original_image
            
            # Grad-CAM Visualization
            st.subheader("🎯 Model Attention (Grad-CAM)")
            
            if superimposed_img is not None:
                # Create comparison plot
                gradcam_fig = plot_gradcam_comparison(
                    original_image.resize((224, 224)),
                    heatmap_img,
                    superimposed_img
                )
                st.pyplot(gradcam_fig)
                
                st.info("🔍 **Interpretation:** Red areas show where the model focused most for its prediction.")
            
            # Probability chart
            st.subheader("Probability Distribution")
            prob_chart = plot_probability_chart(all_probs)
            st.pyplot(prob_chart)
            
            # Class information
            st.subheader("ℹ️ Condition Information")
            class_info = get_class_info(predicted_class)
            
            if class_info:
                st.write(f"**Description:** {class_info['description']}")
                st.write(f"**Common Symptoms:** {class_info['symptoms']}")
                st.write(f"**Treatment Options:** {class_info['treatment']}")
                st.write(f"**Severity:** {class_info['severity']}")
                st.write(f"**Expected Attention Areas:** {class_info['attention_areas']}")
            
            # Recommendations based on prediction
            st.subheader("💡 Recommendations")
            recommendations = {
                'Cyst': "Most cysts are benign. Follow up with your doctor for monitoring.",
                'Normal': "Continue regular health checkups and maintain a healthy lifestyle.",
                'Stone': "Consult a urologist. Drink plenty of water and follow medical advice.",
                'Tumor': "Urgently consult a nephrologist or oncologist for further evaluation."
            }
            st.info(recommendations.get(predicted_class, "Consult a healthcare professional."))
            
            # Download results
            st.subheader("📥 Download Results")
            results_text = f"""
Kidney Disease Classification Results
====================================
Image: {st.session_state.uploaded_file_name}
Prediction: {predicted_class}
Confidence: {confidence:.1%}

Probability Breakdown:
{''.join([f'- {cls}: {prob:.1%}\n' for cls, prob in all_probs.items()])}

Condition Information:
- Description: {class_info.get('description', 'N/A')}
- Symptoms: {class_info.get('symptoms', 'N/A')}
- Treatment: {class_info.get('treatment', 'N/A')}
- Severity: {class_info.get('severity', 'N/A')}
- Expected Attention Areas: {class_info.get('attention_areas', 'N/A')}

Recommendation:
{recommendations.get(predicted_class, 'Consult a healthcare professional.')}

Generated by Kidney Disease Classifier with Grad-CAM
            """
            
            st.download_button(
                label="Download Results as Text",
                data=results_text,
                file_name=f"kidney_analysis_{st.session_state.uploaded_file_name.split('.')[0]}.txt",
                mime="text/plain",
                use_container_width=True
            )
        else:
            # Demo section when no image is uploaded
            st.subheader("🎯 How It Works")
            st.markdown("""
            1. **Upload** a kidney ultrasound image
            2. **AI Analysis** processes the image using ResNet50 + Grad-CAM
            3. **Visualize Attention** see where the model focuses
            4. **Get Results** with confidence scores and medical information
            
            **Grad-CAM Features:**
            - 🔴 **Red** = High attention (model focus)
            - 🔵 **Blue** = Low attention
            - Shows exactly what the AI looks at
            
            **Supported Conditions:**
            - 🟥 **Cyst**: Fluid-filled sacs
            - 🟩 **Normal**: Healthy kidney  
            - 🟧 **Stone**: Mineral deposits
            - 🟪 **Tumor**: Abnormal growths
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Medical Disclaimer:** This AI tool is designed for educational and research purposes. "
        "It should not replace professional medical diagnosis. Always consult qualified "
        "healthcare providers for medical decisions."
    )

if __name__ == "__main__":
    main()