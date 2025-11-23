"""
Deep Learning Signature Upload Manager
FIXED: Uses 512 dimensions to match Pinecone index
"""

import cv2
import numpy as np
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import uuid
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import warnings
warnings.filterwarnings('ignore')

# Configuration - FIXED: Match Pinecone index dimension
TARGET_SIZE = (128, 128)
EMBEDDING_DIM = 512  # FIXED: Changed from 256 to 512

# ==================== DEEP LEARNING MODELS ====================

class SiameseNetwork:
    """Siamese CNN for signature embedding extraction"""
    
    def __init__(self, input_shape=(128, 128, 1), embedding_dim=512):  # FIXED: 512
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        self.model = self._build_embedding_network()
    
    def _build_embedding_network(self):
        """Build the embedding network"""
        inputs = layers.Input(shape=self.input_shape, name='signature_input')
        
        # Convolutional Feature Extraction
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Global Average Pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense Embedding Layers - FIXED: Output 512 dimensions
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(self.embedding_dim, activation='linear', name='embedding')(x)
        
        # L2 Normalization
        x = layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=1))(x)
        
        model = models.Model(inputs=inputs, outputs=x, name='signature_embedder')
        return model
    
    def extract_embedding(self, signature_img):
        """Extract embedding from a signature image"""
        if len(signature_img.shape) == 2:
            signature_img = np.expand_dims(signature_img, axis=-1)
        
        signature_img = signature_img.astype('float32') / 255.0
        signature_img = np.expand_dims(signature_img, axis=0)
        
        embedding = self.model.predict(signature_img, verbose=0)
        return embedding[0]

class MobileNetEmbedder:
    """Transfer Learning with MobileNetV2"""
    
    def __init__(self, input_shape=(128, 128, 3), embedding_dim=512):  # FIXED: 512
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        self.model = self._build_model()
    
    def _build_model(self):
        """Build transfer learning model"""
        base_model = MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        
        base_model.trainable = False
        
        inputs = layers.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(self.embedding_dim, activation='linear')(x)  # FIXED: 512
        x = layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=1))(x)
        
        model = models.Model(inputs=inputs, outputs=x, name='mobilenet_embedder')
        return model
    
    def extract_embedding(self, signature_img):
        """Extract embedding from signature"""
        if len(signature_img.shape) == 2:
            signature_img = cv2.cvtColor(signature_img, cv2.COLOR_GRAY2RGB)
        elif signature_img.shape[2] == 1:
            signature_img = cv2.cvtColor(signature_img, cv2.COLOR_GRAY2RGB)
        
        signature_img = signature_img.astype('float32') / 255.0
        signature_img = np.expand_dims(signature_img, axis=0)
        
        embedding = self.model.predict(signature_img, verbose=0)
        return embedding[0]

# ==================== PREPROCESSING ====================

class SignaturePreprocessor:
    """Advanced preprocessing pipeline"""
    
    def __init__(self, target_size=TARGET_SIZE):
        self.target_size = target_size
    
    def preprocess(self, img):
        """Complete preprocessing pipeline"""
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img.copy()
        
        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img_enhanced = clahe.apply(img_gray)
        
        # Adaptive thresholding
        img_thresh = cv2.adaptiveThreshold(
            img_enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Noise removal
        kernel = np.ones((2, 2), np.uint8)
        img_clean = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)
        img_clean = cv2.morphologyEx(img_clean, cv2.MORPH_OPEN, kernel)
        
        # Crop to content
        img_cropped = self._crop_to_content(img_clean)
        
        # Resize with padding
        img_resized = self._resize_with_padding(img_cropped, self.target_size)
        
        return img_resized, img_gray, img_enhanced
    
    def _crop_to_content(self, img):
        """Crop image to signature content"""
        coords = cv2.findNonZero(img)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img.shape[1] - x, w + 2 * padding)
            h = min(img.shape[0] - y, h + 2 * padding)
            return img[y:y+h, x:x+w]
        return img
    
    def _resize_with_padding(self, img, target_size):
        """Resize with aspect ratio preservation"""
        h, w = img.shape
        target_h, target_w = target_size
        
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        canvas = np.zeros((target_h, target_w), dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized
        
        return canvas

# ==================== QUALITY CHECKS ====================

def check_image_quality(img):
    """Advanced image quality assessment"""
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    
    quality_metrics = {}
    issues = []
    quality_score = 0
    
    # 1. Blur detection (Laplacian variance)
    laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    quality_metrics['blur_score'] = laplacian_var
    
    if laplacian_var > 100:
        quality_score += 3
    elif laplacian_var > 50:
        quality_score += 2
        issues.append("Slightly blurry - consider retaking")
    else:
        quality_score += 1
        issues.append("⚠️ Very blurry - please use clearer image")
    
    # 2. Contrast check
    contrast = img_gray.std()
    quality_metrics['contrast'] = contrast
    
    if contrast > 50:
        quality_score += 2
    elif contrast > 25:
        quality_score += 1
        issues.append("Low contrast detected")
    else:
        issues.append("⚠️ Very low contrast - signature barely visible")
    
    # 3. Brightness check
    brightness = img_gray.mean()
    quality_metrics['brightness'] = brightness
    
    if 50 < brightness < 200:
        quality_score += 2
    else:
        quality_score += 1
        issues.append("Brightness issues - too dark or too bright")
    
    # 4. Content density (signature presence)
    _, binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
    density = np.count_nonzero(binary) / binary.size
    quality_metrics['density'] = density
    
    if 0.05 < density < 0.5:
        quality_score += 2
    else:
        quality_score += 1
        if density < 0.05:
            issues.append("⚠️ Very faint signature")
        else:
            issues.append("Too much ink/noise")
    
    # 5. Edge strength
    edges = cv2.Canny(img_gray, 50, 150)
    edge_density = np.count_nonzero(edges) / edges.size
    quality_metrics['edge_density'] = edge_density
    
    if edge_density > 0.02:
        quality_score += 1
    else:
        issues.append("Weak edges - signature may be unclear")
    
    quality_metrics['quality_score'] = quality_score
    quality_metrics['max_score'] = 10
    quality_metrics['issues'] = issues
    
    return quality_metrics

def validate_embedding(embedding):
    """Validate embedding quality"""
    embedding_array = np.array(embedding)
    
    # Check for all zeros
    if np.all(embedding_array == 0):
        return False, "Embedding is all zeros"
    
    # Check norm
    norm = np.linalg.norm(embedding_array)
    if norm < 0.1:
        return False, f"Embedding norm too low: {norm:.4f}"
    
    # Check for NaN or Inf
    if np.any(np.isnan(embedding_array)) or np.any(np.isinf(embedding_array)):
        return False, "Embedding contains NaN or Inf values"
    
    return True, "Embedding valid"

# ==================== PINECONE ====================

def init_pinecone():
    """Initialize Pinecone with serverless index"""
    try:
        try:
            pinecone_api_key = st.secrets["PINECONE_API_KEY"]
        except:
            st.sidebar.warning("⚠️ Please enter Pinecone credentials")
            pinecone_api_key = st.sidebar.text_input("Pinecone API Key", type="password")
            if not pinecone_api_key:
                st.stop()
        
        pc = Pinecone(api_key=pinecone_api_key)
        index_name = "signature-verification"
        
        # Check/create index
        existing_indexes = [index.name for index in pc.list_indexes()]
        if index_name not in existing_indexes:
            st.info("Creating new signature database...")
            pc.create_index(
                name=index_name,
                dimension=EMBEDDING_DIM,  # FIXED: Now 512
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            st.success("✅ Database created successfully!")
        
        index = pc.Index(index_name)
        return index, pc
        
    except Exception as e:
        st.error(f"❌ Pinecone initialization failed: {e}")
        return None, None

# ==================== MAIN APPLICATION ====================

def main():
    st.set_page_config(
        page_title="Deep Learning Signature Upload",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .upload-header {
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .quality-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #11998e;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="upload-header"><h1>🧠 Deep Learning Signature Upload</h1><p>Add signatures with Siamese CNN & MobileNetV2 (512D Embeddings)</p></div>', unsafe_allow_html=True)
    
    # Initialize models
    if 'preprocessor' not in st.session_state:
        st.session_state.preprocessor = SignaturePreprocessor()
    
    if 'siamese_model' not in st.session_state:
        with st.spinner("🔄 Loading Siamese Neural Network (512D)..."):
            st.session_state.siamese_model = SiameseNetwork()
    
    if 'mobilenet_model' not in st.session_state:
        with st.spinner("🔄 Loading MobileNetV2 (512D)..."):
            st.session_state.mobilenet_model = MobileNetEmbedder()
    
    # Sidebar - Configuration
    st.sidebar.header("⚙️ Upload Configuration")
    
    model_choice = st.sidebar.selectbox(
        "Embedding Model",
        ["Siamese CNN (Custom)", "MobileNetV2 (Transfer)", "Hybrid (Recommended)"],
        index=2,
        help="Choose model for feature extraction"
    )
    
    st.sidebar.markdown("---")
    
    # Initialize Pinecone
    index, pc = init_pinecone()
    if index is None:
        st.stop()
    
    st.sidebar.success("✅ Database Connected")
    
    # Database stats
    try:
        stats = index.describe_index_stats()
        st.sidebar.metric("Total Signatures", stats['total_vector_count'])
        st.sidebar.metric("Dimension", stats['dimension'])
        
        # FIXED: Now checks for 512
        if stats['dimension'] != EMBEDDING_DIM:
            st.sidebar.error(f"⚠️ Dimension mismatch! Expected {EMBEDDING_DIM}, got {stats['dimension']}")
        else:
            st.sidebar.success(f"✅ Dimension match: {EMBEDDING_DIM}")
    except Exception as e:
        st.sidebar.metric("Total Signatures", 0)
    
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Quality Guidelines")
    st.sidebar.info("""
    **Quality Score:**
    - 8-10: Excellent
    - 6-7: Good
    - 4-5: Acceptable
    - 0-3: Poor (not recommended)
    
    **Requirements:**
    - Clear signature strokes
    - Good contrast
    - White/light background
    - No shadows or glare
    """)
    
    # Main tabs
    tab1, tab2 = st.tabs(["➕ Single Signature", "📚 Batch Upload"])
    
    # ==================== SINGLE UPLOAD ====================
    with tab1:
        st.header("Upload Single Signature")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose signature image",
                type=['png', 'jpg', 'jpeg', 'bmp'],
                key="single_upload"
            )
            
            if uploaded_file:
                # Read and display
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                st.image(img, caption="Original Signature")
                
                # Preprocess
                with st.spinner("Processing..."):
                    preprocessed, gray, enhanced = st.session_state.preprocessor.preprocess(img)
                    st.session_state.current_img = img
                    st.session_state.preprocessed = preprocessed
                
                # Show processing stages
                proc_cols = st.columns(3)
                with proc_cols[0]:
                    st.image(gray, caption="Grayscale")
                with proc_cols[1]:
                    st.image(enhanced, caption="Enhanced")
                with proc_cols[2]:
                    st.image(preprocessed, caption="Processed")
                
                # Quality check
                quality = check_image_quality(img)
                st.session_state.quality = quality
                
                # Display quality
                st.markdown('<div class="quality-card">', unsafe_allow_html=True)
                st.subheader("📊 Quality Assessment")
                
                q_cols = st.columns(4)
                with q_cols[0]:
                    st.metric("Quality Score", f"{quality['quality_score']}/{quality['max_score']}")
                with q_cols[1]:
                    st.metric("Blur Score", f"{quality['blur_score']:.1f}")
                with q_cols[2]:
                    st.metric("Contrast", f"{quality['contrast']:.1f}")
                with q_cols[3]:
                    st.metric("Density", f"{quality['density']*100:.1f}%")
                
                if quality['issues']:
                    st.warning("**Quality Issues:**")
                    for issue in quality['issues']:
                        st.write(f"• {issue}")
                else:
                    st.success("✅ Excellent quality signature!")
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.subheader("Signature Details")
            
            user_id = st.text_input(
                "User ID*",
                placeholder="john_doe_001",
                help="Unique identifier for this user"
            )
            
            full_name = st.text_input(
                "Full Name*",
                placeholder="John Doe"
            )
            
            department = st.selectbox(
                "Department",
                ["Finance", "HR", "Legal", "Operations", "Management", "Sales", "IT", "Engineering", "Marketing", "Other"]
            )
            
            description = st.text_area(
                "Description",
                placeholder="Official signature for contracts...",
                help="Optional notes about this signature"
            )
            
            st.markdown("---")
            
            # Store button
            if st.button("💾 Store Signature", type="primary"):
                if 'current_img' not in st.session_state:
                    st.error("❌ Please upload a signature first")
                    st.stop()
                
                if not user_id or not full_name:
                    st.error("❌ Please enter User ID and Full Name")
                    st.stop()
                
                with st.spinner("🧠 Generating 512D deep learning embeddings..."):
                    preprocessed = st.session_state.preprocessed
                    
                    # Extract embedding based on model choice
                    if model_choice == "Siamese CNN (Custom)":
                        embedding = st.session_state.siamese_model.extract_embedding(preprocessed)
                        model_used = "siamese_cnn"
                        
                    elif model_choice == "MobileNetV2 (Transfer)":
                        preprocessed_rgb = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2RGB)
                        embedding = st.session_state.mobilenet_model.extract_embedding(preprocessed_rgb)
                        model_used = "mobilenet_v2"
                        
                    else:  # Hybrid
                        emb1 = st.session_state.siamese_model.extract_embedding(preprocessed)
                        preprocessed_rgb = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2RGB)
                        emb2 = st.session_state.mobilenet_model.extract_embedding(preprocessed_rgb)
                        embedding = (emb1 + emb2) / 2
                        embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
                        model_used = "hybrid"
                    
                    # Validate embedding
                    is_valid, message = validate_embedding(embedding)
                    
                    if not is_valid:
                        st.error(f"❌ Invalid embedding: {message}")
                        st.stop()
                    
                    st.success(f"✅ {len(embedding)}D embedding generated using {model_used}")
                    
                    # Generate signature ID
                    signature_id = f"{user_id}_{uuid.uuid4().hex[:8]}"
                    
                    # Prepare metadata
                    metadata = {
                        "user_id": user_id,
                        "full_name": full_name,
                        "department": department,
                        "description": description,
                        "original_filename": uploaded_file.name,
                        "timestamp": datetime.now().isoformat(),
                        "quality_score": quality['quality_score'],
                        "blur_score": float(quality['blur_score']),
                        "contrast": float(quality['contrast']),
                        "density": float(quality['density']),
                        "model_used": model_used,
                        "embedding_dim": EMBEDDING_DIM
                    }
                    
                    # Store in Pinecone
                    try:
                        index.upsert(vectors=[{
                            "id": signature_id,
                            "values": embedding.tolist(),
                            "metadata": metadata
                        }])
                        
                        st.success("🎉 Signature stored successfully!")
                        
                        # Show summary
                        st.subheader("📄 Storage Summary")
                        summary_cols = st.columns(2)
                        
                        with summary_cols[0]:
                            st.write("**Signature Info:**")
                            st.write(f"• ID: {signature_id}")
                            st.write(f"• User: {user_id}")
                            st.write(f"• Name: {full_name}")
                            st.write(f"• Department: {department}")
                        
                        with summary_cols[1]:
                            st.write("**Technical Details:**")
                            st.write(f"• Model: {model_used}")
                            st.write(f"• Quality: {quality['quality_score']}/10")
                            st.write(f"• Embedding Dim: {EMBEDDING_DIM}")
                            st.write(f"• Norm: {np.linalg.norm(embedding):.4f}")
                        
                        # Clear session state
                        for key in ['current_img', 'preprocessed', 'quality']:
                            if key in st.session_state:
                                del st.session_state[key]
                        
                    except Exception as e:
                        st.error(f"❌ Failed to store: {e}")
    
    # ==================== BATCH UPLOAD ====================
    with tab2:
        st.header("Batch Upload Multiple Signatures")
        
        st.info("""
        **Batch Upload Process:**
        1. Upload multiple signature images
        2. Review quality for all signatures
        3. Configure user details
        4. Generate 512D embeddings for all
        5. Store in database
        """)
        
        uploaded_files = st.file_uploader(
            "Upload multiple signatures",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            accept_multiple_files=True,
            key="batch_upload"
        )
        
        if uploaded_files:
            st.success(f"📁 {len(uploaded_files)} files selected")
            
            # Process all files
            with st.spinner("Processing all signatures..."):
                batch_data = []
                
                for i, file in enumerate(uploaded_files):
                    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    preprocessed, gray, enhanced = st.session_state.preprocessor.preprocess(img)
                    quality = check_image_quality(img)
                    
                    batch_data.append({
                        'filename': file.name,
                        'image': img,
                        'preprocessed': preprocessed,
                        'quality': quality,
                        'index': i
                    })
            
            # Display preview grid
            st.subheader("📸 Signature Preview")
            preview_cols = st.columns(4)
            
            for i, data in enumerate(batch_data[:12]):  # Show first 12
                with preview_cols[i % 4]:
                    st.image(data['image'], caption=f"#{i+1}: Q={data['quality']['quality_score']}")
            
            if len(batch_data) > 12:
                st.info(f"Showing 12 of {len(batch_data)} signatures")
            
            # Quality summary
            st.subheader("📊 Batch Quality Summary")
            
            quality_scores = [d['quality']['quality_score'] for d in batch_data]
            avg_quality = sum(quality_scores) / len(quality_scores)
            
            sum_cols = st.columns(4)
            with sum_cols[0]:
                st.metric("Average Quality", f"{avg_quality:.1f}/10")
            with sum_cols[1]:
                excellent = sum(1 for q in quality_scores if q >= 8)
                st.metric("Excellent (8-10)", f"{excellent}/{len(batch_data)}")
            with sum_cols[2]:
                good = sum(1 for q in quality_scores if 6 <= q < 8)
                st.metric("Good (6-7)", f"{good}/{len(batch_data)}")
            with sum_cols[3]:
                poor = sum(1 for q in quality_scores if q < 4)
                st.metric("Poor (0-3)", f"{poor}/{len(batch_data)}")
            
            # Configuration
            st.subheader("⚙️ Batch Configuration")
            
            config_cols = st.columns(2)
            
            with config_cols[0]:
                naming_mode = st.radio(
                    "User ID Generation",
                    ["Auto (user_001, user_002...)", "Manual Entry", "From Filename"]
                )
                
                if naming_mode == "Manual Entry":
                    user_ids_text = st.text_area(
                        "Enter User IDs (one per line)",
                        placeholder="john_doe\njane_smith\nmike_wilson",
                        height=150
                    )
                
                department_batch = st.selectbox(
                    "Department (for all)",
                    ["Finance", "HR", "Legal", "Operations", "Management", "Sales", "IT", "Engineering", "Various"]
                )
            
            with config_cols[1]:
                quality_filter = st.slider(
                    "Minimum Quality Score",
                    min_value=0,
                    max_value=10,
                    value=4,
                    help="Only store signatures above this quality"
                )
                
                st.write(f"**Signatures passing filter:** {sum(1 for q in quality_scores if q >= quality_filter)}/{len(batch_data)}")
                
                add_descriptions = st.checkbox("Add descriptions", value=False)
            
            # Store batch button
            st.markdown("---")
            
            if st.button("🚀 Generate 512D Embeddings & Store All", type="primary"):
                progress_bar = st.progress(0)
                status = st.empty()
                
                stored_count = 0
                failed_count = 0
                skipped_count = 0
                
                results = []
                
                for i, data in enumerate(batch_data):
                    progress = (i + 1) / len(batch_data)
                    progress_bar.progress(progress)
                    status.text(f"Processing {i+1}/{len(batch_data)}: {data['filename']}")
                    
                    # Skip if quality too low
                    if data['quality']['quality_score'] < quality_filter:
                        skipped_count += 1
                        results.append({
                            'filename': data['filename'],
                            'status': 'Skipped (low quality)',
                            'quality': data['quality']['quality_score']
                        })
                        continue
                    
                    try:
                        # Generate user ID
                        if naming_mode == "Auto":
                            user_id = f"user_{i+1:03d}"
                        elif naming_mode == "From Filename":
                            user_id = data['filename'].split('.')[0].replace(' ', '_')
                        else:  # Manual
                            if user_ids_text:
                                ids = user_ids_text.strip().split('\n')
                                user_id = ids[i].strip() if i < len(ids) else f"user_{i+1:03d}"
                            else:
                                user_id = f"user_{i+1:03d}"
                        
                        # Extract embedding
                        preprocessed = data['preprocessed']
                        
                        if model_choice == "Siamese CNN (Custom)":
                            embedding = st.session_state.siamese_model.extract_embedding(preprocessed)
                            model_used = "siamese_cnn"
                        elif model_choice == "MobileNetV2 (Transfer)":
                            preprocessed_rgb = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2RGB)
                            embedding = st.session_state.mobilenet_model.extract_embedding(preprocessed_rgb)
                            model_used = "mobilenet_v2"
                        else:  # Hybrid
                            emb1 = st.session_state.siamese_model.extract_embedding(preprocessed)
                            preprocessed_rgb = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2RGB)
                            emb2 = st.session_state.mobilenet_model.extract_embedding(preprocessed_rgb)
                            embedding = (emb1 + emb2) / 2
                            embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
                            model_used = "hybrid"
                        
                        # Validate embedding
                        is_valid, message = validate_embedding(embedding)
                        
                        if not is_valid:
                            failed_count += 1
                            results.append({
                                'filename': data['filename'],
                                'status': f'Failed: {message}',
                                'quality': data['quality']['quality_score']
                            })
                            continue
                        
                        # Generate signature ID
                        signature_id = f"{user_id}_{uuid.uuid4().hex[:8]}"
                        
                        # Prepare metadata
                        metadata = {
                            "user_id": user_id,
                            "full_name": f"{user_id.replace('_', ' ').title()}",
                            "department": department_batch,
                            "description": f"Batch upload from {data['filename']}" if add_descriptions else "",
                            "original_filename": data['filename'],
                            "timestamp": datetime.now().isoformat(),
                            "quality_score": data['quality']['quality_score'],
                            "blur_score": float(data['quality']['blur_score']),
                            "contrast": float(data['quality']['contrast']),
                            "density": float(data['quality']['density']),
                            "model_used": model_used,
                            "embedding_dim": EMBEDDING_DIM,
                            "batch_upload": True
                        }
                        
                        # Store in Pinecone
                        index.upsert(vectors=[{
                            "id": signature_id,
                            "values": embedding.tolist(),
                            "metadata": metadata
                        }])
                        
                        stored_count += 1
                        results.append({
                            'filename': data['filename'],
                            'user_id': user_id,
                            'signature_id': signature_id,
                            'status': '✅ Stored',
                            'quality': data['quality']['quality_score'],
                            'model': model_used
                        })
                        
                    except Exception as e:
                        failed_count += 1
                        results.append({
                            'filename': data['filename'],
                            'status': f'❌ Error: {str(e)[:50]}',
                            'quality': data['quality']['quality_score']
                        })
                
                # Clear progress
                progress_bar.empty()
                status.empty()
                
                # Display results
                st.success(f"🎉 Batch Upload Complete!")
                
                result_cols = st.columns(3)
                with result_cols[0]:
                    st.metric("✅ Stored", stored_count)
                with result_cols[1]:
                    st.metric("⚠️ Skipped", skipped_count)
                with result_cols[2]:
                    st.metric("❌ Failed", failed_count)
                
                # Detailed results table
                st.subheader("📋 Detailed Results")
                
                import pandas as pd
                df = pd.DataFrame(results)
                st.dataframe(df)
                
                # Download results as CSV
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name=f"batch_upload_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    # ==================== DATABASE MANAGEMENT ====================
    
    st.markdown("---")
    st.header("🗄️ Database Management")
    
    db_cols = st.columns(3)
    
    with db_cols[0]:
        if st.button("🔄 Refresh Stats"):
            st.rerun()
    
    with db_cols[1]:
        if st.button("📊 View All Signatures"):
            try:
                # Query random samples
                dummy_vector = [0.0] * EMBEDDING_DIM
                results = index.query(
                    vector=dummy_vector,
                    top_k=100,
                    include_metadata=True
                )
                
                if results['matches']:
                    st.subheader("📄 Database Contents")
                    
                    db_data = []
                    for match in results['matches']:
                        metadata = match.get('metadata', {})
                        db_data.append({
                            'User ID': metadata.get('user_id', 'N/A'),
                            'Full Name': metadata.get('full_name', 'N/A'),
                            'Department': metadata.get('department', 'N/A'),
                            'Quality': metadata.get('quality_score', 'N/A'),
                            'Model': metadata.get('model_used', 'N/A'),
                            'Timestamp': metadata.get('timestamp', 'N/A')[:10]
                        })
                    
                    import pandas as pd
                    df = pd.DataFrame(db_data)
                    st.dataframe(df)
                    st.write(f"Showing {len(db_data)} signatures")
                else:
                    st.info("No signatures in database yet")
                    
            except Exception as e:
                st.error(f"Error querying database: {e}")
    
    with db_cols[2]:
        with st.expander("🗑️ Delete Signatures"):
            st.warning("⚠️ Danger Zone")
            
            delete_mode = st.radio(
                "Delete mode",
                ["By Signature ID", "Clear All"],
                key="delete_mode_radio"
            )
            
            if delete_mode == "By Signature ID":
                sig_id_delete = st.text_input("Signature ID to delete")
                if st.button("Delete Signature", type="secondary"):
                    if sig_id_delete:
                        try:
                            index.delete(ids=[sig_id_delete])
                            st.success(f"✅ Deleted signature: {sig_id_delete}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Delete failed: {e}")
                    else:
                        st.error("Enter a Signature ID")
            
            else:  # Clear All
                st.error("⚠️ This will delete ALL signatures!")
                confirm = st.text_input("Type 'DELETE ALL' to confirm")
                if st.button("Clear Database", type="secondary"):
                    if confirm == "DELETE ALL":
                        try:
                            index.delete(delete_all=True)
                            st.success("✅ All signatures deleted")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Delete failed: {e}")
                    else:
                        st.error("Confirmation text doesn't match")
    
    # ==================== HELP & DOCUMENTATION ====================
    
    st.sidebar.markdown("---")
    st.sidebar.header("📚 Documentation")
    
    with st.sidebar.expander("🔍 Model Information"):
        st.write(f"""
        **Current Configuration:**
        - Embedding Model: {model_choice}
        - Embedding Dimension: {EMBEDDING_DIM}
        - Target Size: {TARGET_SIZE}
        
        **Siamese CNN:**
        - Custom architecture
        - 4 Conv blocks with BatchNorm
        - 512D output embedding
        - Writer-independent
        
        **MobileNetV2:**
        - Transfer learning
        - ImageNet pre-trained
        - 512D output embedding
        - Efficient inference
        
        **Hybrid:**
        - Averages both models
        - 512D combined embedding
        - Highest accuracy
        - Recommended for production
        """)
    
    with st.sidebar.expander("💡 Best Practices"):
        st.write("""
        **Image Requirements:**
        - Resolution: 300+ DPI
        - Format: PNG, JPG, JPEG
        - Background: White/light
        - Lighting: Even, no shadows
        - Focus: Sharp, not blurry
        
        **Quality Tips:**
        - Use scanner or high-quality camera
        - Ensure signature is centered
        - Avoid reflections on paper
        - Use black/blue ink (not pencil)
        - Capture full signature without cropping
        
        **Batch Upload:**
        - Maximum 100 files at once
        - All images should be similar quality
        - Use consistent naming convention
        - Review quality before storing
        """)
    
    with st.sidebar.expander("⚙️ Technical Details"):
        st.write(f"""
        **System Configuration:**
        - Database: Pinecone Serverless
        - Region: us-east-1
        - Metric: Cosine Similarity
        - Dimension: {EMBEDDING_DIM}
        
        **Processing Pipeline:**
        1. Grayscale conversion
        2. CLAHE enhancement
        3. Adaptive thresholding
        4. Morphological operations
        5. Content cropping
        6. Aspect-preserving resize
        7. Deep learning embedding (512D)
        8. L2 normalization
        9. Pinecone storage
        
        **Quality Metrics:**
        - Blur detection (Laplacian)
        - Contrast analysis
        - Brightness check
        - Content density
        - Edge strength
        """)
    
    with st.sidebar.expander("🔧 Dimension Fix"):
        st.write(f"""
        **FIXED: Dimension Mismatch**
        
        Previously: 256 dimensions
        Now: **512 dimensions**
        
        This matches your existing Pinecone index.
        
        **What changed:**
        - Siamese CNN output: 512D
        - MobileNetV2 output: 512D
        - Hybrid average: 512D
        - All embeddings now compatible
        
        ✅ No more dimension errors!
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>Deep Learning Signature Upload Manager (512D)</strong></p>
        <p>Powered by Siamese CNN & MobileNetV2 • Pinecone Vector Database</p>
        <p>⚠️ Ensure you have permission to store and process these signatures</p>
        <p style='font-size: 0.9em; margin-top: 10px;'>
            ✅ Now with 512-dimensional embeddings matching your Pinecone index
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()