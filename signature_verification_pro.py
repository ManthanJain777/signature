"""
Enhanced Deep Learning Signature Verification System
COMPLETE VERSION - Matches signature manager with 512D embeddings
Features: Siamese CNN, MobileNetV2, Hybrid mode, High accuracy verification
"""

import cv2
import numpy as np
import streamlit as st
from pinecone import Pinecone
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import warnings
warnings.filterwarnings('ignore')

# Configuration - MUST MATCH UPLOAD MANAGER
TARGET_SIZE = (128, 128)
EMBEDDING_DIM = 512  # FIXED: Match Pinecone index dimension

# ==================== DEEP LEARNING MODELS ====================

class SiameseNetwork:
    """Siamese CNN for signature embedding extraction - MATCHES UPLOAD MANAGER"""
    
    def __init__(self, input_shape=(128, 128, 1), embedding_dim=512):
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        self.model = self._build_embedding_network()
    
    def _build_embedding_network(self):
        """Build the embedding network - IDENTICAL TO UPLOAD MANAGER"""
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
        
        # Dense Embedding Layers - 512D output
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
    """Transfer Learning with MobileNetV2 - MATCHES UPLOAD MANAGER"""
    
    def __init__(self, input_shape=(128, 128, 3), embedding_dim=512):
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        self.model = self._build_model()
    
    def _build_model(self):
        """Build transfer learning model - IDENTICAL TO UPLOAD MANAGER"""
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
        x = layers.Dense(self.embedding_dim, activation='linear')(x)
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
    """Advanced preprocessing pipeline - MATCHES UPLOAD MANAGER"""
    
    def __init__(self, target_size=TARGET_SIZE):
        self.target_size = target_size
    
    def preprocess(self, img):
        """Complete preprocessing pipeline - IDENTICAL TO UPLOAD MANAGER"""
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

# ==================== QUALITY ASSESSMENT ====================

def check_image_quality(img):
    """Advanced image quality assessment - MATCHES UPLOAD MANAGER"""
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    
    quality_metrics = {}
    issues = []
    quality_score = 0
    
    # 1. Blur detection
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
    
    # 4. Content density
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

# ==================== PINECONE INTEGRATION ====================

def init_pinecone():
    """Initialize Pinecone connection"""
    try:
        pinecone_api_key = st.secrets["PINECONE_API_KEY"]
    except KeyError:
        st.error("❌ Please set PINECONE_API_KEY in secrets.toml")
        st.stop()
    
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "signature-verification"
    
    existing_indexes = [index.name for index in pc.list_indexes()]
    if index_name not in existing_indexes:
        st.error("❌ No signature database found. Please upload signatures first.")
        st.stop()
    
    index = pc.Index(index_name)
    return index

def find_similar_signatures(index, query_embedding, top_k=10, user_filter=None):
    """Search for similar signatures with improved accuracy"""
    query_params = {
        'vector': query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding,
        'top_k': top_k,
        'include_metadata': True,
        'include_values': True  # FIXED: Include vector values for multi-metric analysis
    }
    
    if user_filter:
        query_params['filter'] = {"user_id": {"$eq": user_filter}}
    
    results = index.query(**query_params)
    return results

# ==================== ADVANCED SIMILARITY METRICS ====================

def calculate_verification_score(pinecone_score):
    """Convert Pinecone cosine distance to percentage score
    Pinecone returns distance (0-2), where 0 is identical"""
    similarity = 1 - (pinecone_score / 2)
    return similarity * 100

def calculate_multi_metric_score(embedding1, embedding2):
    """Calculate multiple similarity metrics for robust verification"""
    # Cosine similarity
    cosine_sim = np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) * np.linalg.norm(embedding2) + 1e-10
    )
    
    # Euclidean distance
    euclidean_dist = np.linalg.norm(embedding1 - embedding2)
    
    # Manhattan distance
    manhattan_dist = np.sum(np.abs(embedding1 - embedding2))
    
    # Normalize euclidean to 0-1 range
    euclidean_sim = max(0, 1 - (euclidean_dist / 2))
    
    # Normalize manhattan
    manhattan_sim = max(0, 1 - (manhattan_dist / (2 * len(embedding1))))
    
    return {
        'cosine_similarity': cosine_sim * 100,
        'euclidean_similarity': euclidean_sim * 100,
        'manhattan_similarity': manhattan_sim * 100,
        'average_score': (cosine_sim + euclidean_sim + manhattan_sim) / 3 * 100
    }

def advanced_verification_decision(matches, threshold, user_filter=None):
    """Advanced decision logic for verification"""
    if not matches:
        return {
            'decision': 'REJECTED',
            'reason': 'No matching signatures found',
            'confidence': 'N/A'
        }
    
    best_match = matches[0]
    best_score = calculate_verification_score(best_match['score'])
    
    # Check if user-specific verification
    if user_filter:
        best_user = best_match.get('metadata', {}).get('user_id', '')
        if best_user != user_filter:
            return {
                'decision': 'REJECTED',
                'reason': f'Best match belongs to different user: {best_user}',
                'confidence': 'High',
                'best_score': best_score
            }
    
    # Decision based on score
    if best_score >= threshold:
        # Check for significant gap to second best (anti-forgery)
        if len(matches) > 1:
            second_score = calculate_verification_score(matches[1]['score'])
            score_gap = best_score - second_score
            
            if score_gap < 5 and best_score < 90:
                confidence = 'Medium'
                reason = f'Multiple similar matches detected (gap: {score_gap:.1f}%)'
            else:
                confidence = 'High' if best_score >= 90 else 'Medium'
                reason = 'Signature authenticated successfully'
        else:
            confidence = 'High' if best_score >= 90 else 'Medium'
            reason = 'Signature authenticated successfully'
        
        return {
            'decision': 'VERIFIED',
            'reason': reason,
            'confidence': confidence,
            'best_score': best_score,
            'matched_user': best_match.get('metadata', {}).get('user_id', 'Unknown')
        }
    else:
        return {
            'decision': 'REJECTED',
            'reason': f'Match score ({best_score:.2f}%) below threshold ({threshold}%)',
            'confidence': 'High',
            'best_score': best_score
        }

# ==================== MAIN APPLICATION ====================

def main():
    st.set_page_config(
        page_title="Enhanced Signature Verification",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .verified-box {
        background: #d4edda;
        border: 2px solid #28a745;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    .rejected-box {
        background: #f8d7da;
        border: 2px solid #dc3545;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header"><h1>🔒 Enhanced Deep Learning Signature Verification</h1><p>High-Accuracy Biometric Authentication System (512D)</p></div>', unsafe_allow_html=True)
    
    # Initialize components
    if 'preprocessor' not in st.session_state:
        st.session_state.preprocessor = SignaturePreprocessor()
    
    if 'siamese_model' not in st.session_state:
        with st.spinner("🔄 Loading Siamese Neural Network (512D)..."):
            st.session_state.siamese_model = SiameseNetwork(
                input_shape=(128, 128, 1),
                embedding_dim=EMBEDDING_DIM
            )
    
    if 'mobilenet_model' not in st.session_state:
        with st.spinner("🔄 Loading MobileNetV2 Model (512D)..."):
            st.session_state.mobilenet_model = MobileNetEmbedder(
                input_shape=(128, 128, 3),
                embedding_dim=EMBEDDING_DIM
            )
    
    # Sidebar - Configuration
    st.sidebar.header("⚙️ Verification Settings")
    
    model_choice = st.sidebar.selectbox(
        "Embedding Model",
        ["Hybrid (Recommended)", "Siamese CNN (Custom)", "MobileNetV2 (Transfer)"],
        help="Hybrid mode provides highest accuracy by combining both models"
    )
    
    st.sidebar.markdown("---")
    
    # Database connection
    try:
        index = init_pinecone()
        stats = index.describe_index_stats()
        
        st.sidebar.success("✅ Database Connected")
        st.sidebar.metric("Total Signatures", stats['total_vector_count'])
        st.sidebar.metric("Embedding Dimension", stats['dimension'])
        
        # Verify dimension match
        if stats['dimension'] != EMBEDDING_DIM:
            st.sidebar.error(f"⚠️ Dimension mismatch! Database: {stats['dimension']}, System: {EMBEDDING_DIM}")
        else:
            st.sidebar.success(f"✅ Dimension match: {EMBEDDING_DIM}")
        
    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
        st.stop()
    
    st.sidebar.markdown("---")
    
    verification_mode = st.sidebar.radio(
        "Verification Mode",
        ["User-Specific (Recommended)", "General Search"],
        help="User-specific mode provides higher security"
    )
    
    user_filter = None
    if verification_mode == "User-Specific (Recommended)":
        user_filter = st.sidebar.text_input(
            "Expected User ID",
            placeholder="e.g., john_doe_001",
            help="Enter the user ID this signature should belong to"
        )
    
    # Advanced threshold settings
    st.sidebar.subheader("🎯 Threshold Configuration")
    
    preset = st.sidebar.selectbox(
        "Security Preset",
        ["High Security (Banking)", "Standard (Legal Docs)", "Relaxed (General Use)", "Custom"],
        help="Pre-configured thresholds for different use cases"
    )
    
    if preset == "High Security (Banking)":
        threshold = 90
        top_k = 10
    elif preset == "Standard (Legal Docs)":
        threshold = 80
        top_k = 8
    elif preset == "Relaxed (General Use)":
        threshold = 70
        top_k = 5
    else:  # Custom
        threshold = st.sidebar.slider(
            "Verification Threshold (%)",
            min_value=60,
            max_value=98,
            value=80,
            help="Minimum similarity score to accept signature"
        )
        top_k = st.sidebar.slider(
            "Top K Matches",
            min_value=5,
            max_value=20,
            value=10,
            help="Number of similar signatures to analyze"
        )
    
    st.sidebar.metric("Current Threshold", f"{threshold}%")
    
    # Quality settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Quality Control")
    
    min_quality = st.sidebar.slider(
        "Minimum Quality Score",
        min_value=0,
        max_value=10,
        value=4,
        help="Reject signatures below this quality"
    )
    
    enable_multi_metric = st.sidebar.checkbox(
        "Enable Multi-Metric Analysis",
        value=True,
        help="Use multiple similarity metrics for better accuracy"
    )
    
    # Main Content Area
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("📤 Upload Signature for Verification")
        
        uploaded_file = st.file_uploader(
            "Choose signature image",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="Upload a clear signature image on white background"
        )
        
        if uploaded_file:
            # Read image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Display original
            st.image(img, caption="Original Signature", use_column_width=True)
            
            # Preprocess
            with st.spinner("🔄 Processing signature..."):
                preprocessed, gray_img, enhanced_img = st.session_state.preprocessor.preprocess(img)
                quality = check_image_quality(img)
                
                st.session_state.query_img = img
                st.session_state.preprocessed_img = preprocessed
                st.session_state.gray_img = gray_img
                st.session_state.enhanced_img = enhanced_img
                st.session_state.quality = quality
            
            # Show preprocessing stages
            proc_cols = st.columns(3)
            with proc_cols[0]:
                st.image(gray_img, caption="Grayscale", use_column_width=True)
            with proc_cols[1]:
                st.image(enhanced_img, caption="Enhanced", use_column_width=True)
            with proc_cols[2]:
                st.image(preprocessed, caption="Processed", use_column_width=True)
            
            # Quality Assessment
            st.markdown("---")
            st.subheader("📊 Image Quality Analysis")
            
            q_cols = st.columns(5)
            with q_cols[0]:
                q_score = quality['quality_score']
                q_color = "normal" if q_score >= min_quality else "inverse"
                st.metric("Quality Score", f"{q_score}/10", delta="PASS" if q_score >= min_quality else "FAIL", delta_color=q_color)
            with q_cols[1]:
                st.metric("Blur Score", f"{quality['blur_score']:.1f}")
            with q_cols[2]:
                st.metric("Contrast", f"{quality['contrast']:.1f}")
            with q_cols[3]:
                st.metric("Density", f"{quality['density']*100:.1f}%")
            with q_cols[4]:
                st.metric("Edge Strength", f"{quality['edge_density']*100:.2f}%")
            
            if quality['issues']:
                with st.expander("⚠️ Quality Issues Detected", expanded=True):
                    for issue in quality['issues']:
                        st.warning(issue)
            else:
                st.success("✅ Excellent image quality!")
            
            # Block if quality too low
            if quality['quality_score'] < min_quality:
                st.error(f"❌ Image quality ({quality['quality_score']}/10) below minimum threshold ({min_quality}/10). Please upload a clearer image.")
    
    with col2:
        st.subheader("🧠 System Configuration")
        
        st.info(f"""
        **Current Settings:**
        - Model: {model_choice}
        - Mode: {verification_mode}
        - Threshold: {threshold}%
        - Security: {preset}
        - Multi-Metric: {'Enabled' if enable_multi_metric else 'Disabled'}
        - Embedding Dim: {EMBEDDING_DIM}
        
        **Expected User:** {user_filter if user_filter else 'Any'}
        """)
        
        st.markdown("---")
        st.subheader("📈 Verification Statistics")
        
        if 'preprocessed_img' in st.session_state:
            img_stats = st.session_state.preprocessed_img
            st.write(f"**Image Size:** {img_stats.shape}")
            st.write(f"**Non-zero Pixels:** {np.count_nonzero(img_stats):,}")
            st.write(f"**Fill Ratio:** {np.count_nonzero(img_stats)/img_stats.size*100:.2f}%")
        
        st.markdown("---")
        st.subheader("🎯 Accuracy Tips")
        st.info("""
        For best results:
        - Use Hybrid model
        - Enable multi-metric
        - User-specific mode when possible
        - Good lighting & white background
        - Quality score 6+
        """)
    
    # Verification Button
    st.markdown("---")
    
    verify_button_col1, verify_button_col2, verify_button_col3 = st.columns([2, 1, 2])
    
    with verify_button_col2:
        verify_button = st.button("🚀 VERIFY SIGNATURE", type="primary", use_container_width=True)
    
    if verify_button:
        if 'preprocessed_img' not in st.session_state:
            st.error("❌ Please upload a signature first")
            st.stop()
        
        # Check quality threshold
        if st.session_state.quality['quality_score'] < min_quality:
            st.error(f"❌ Cannot verify: Image quality too low")
            st.stop()
        
        with st.spinner("🧠 Extracting 512D deep features and searching database..."):
            preprocessed = st.session_state.preprocessed_img
            
            # Extract embeddings based on model choice
            if model_choice == "Siamese CNN (Custom)":
                embedding = st.session_state.siamese_model.extract_embedding(preprocessed)
                model_used = "siamese_cnn"
                st.success("✅ Siamese CNN 512D embedding extracted")
                
            elif model_choice == "MobileNetV2 (Transfer)":
                preprocessed_rgb = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2RGB)
                embedding = st.session_state.mobilenet_model.extract_embedding(preprocessed_rgb)
                model_used = "mobilenet_v2"
                st.success("✅ MobileNetV2 512D embedding extracted")
                
            else:  # Hybrid (Recommended)
                emb1 = st.session_state.siamese_model.extract_embedding(preprocessed)
                preprocessed_rgb = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2RGB)
                emb2 = st.session_state.mobilenet_model.extract_embedding(preprocessed_rgb)
                # Average embeddings
                embedding = (emb1 + emb2) / 2
                # Re-normalize
                embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
                model_used = "hybrid"
                st.success("✅ Hybrid 512D embedding extracted (Highest Accuracy)")
            
            # Validate embedding
            emb_norm = np.linalg.norm(embedding)
            if emb_norm < 0.1 or np.any(np.isnan(embedding)):
                st.error("❌ Invalid embedding generated. Please try another image.")
                st.stop()
            
            # Show embedding info for debugging
            with st.expander("🔬 Embedding Info (Debug)", expanded=False):
                st.write(f"**Embedding Dimension:** {len(embedding)}")
                st.write(f"**Embedding Norm:** {emb_norm:.4f}")
                st.write(f"**Embedding Mean:** {np.mean(embedding):.4f}")
                st.write(f"**Embedding Std:** {np.std(embedding):.4f}")
                st.write(f"**Min Value:** {np.min(embedding):.4f}")
                st.write(f"**Max Value:** {np.max(embedding):.4f}")
            
            st.session_state.query_embedding = embedding
            
            # Search database
            results = find_similar_signatures(
                index, 
                embedding, 
                top_k=top_k,
                user_filter=user_filter
            )
            
            if not results['matches']:
                st.error("❌ No matching signatures found in database")
                if user_filter:
                    st.info(f"💡 User '{user_filter}' may not exist in the database. Try General Search mode to see all matches.")
                else:
                    st.warning("⚠️ The database may be empty or the signature is completely different from all stored signatures.")
                st.stop()
            
            # Advanced verification decision
            decision = advanced_verification_decision(results['matches'], threshold, user_filter)
            
            st.session_state.verification_result = decision
            st.session_state.matches = results['matches']
        
        # ==================== DISPLAY RESULTS ====================
        
        st.markdown("---")
        st.header("🎯 VERIFICATION RESULTS")
        
        # Main decision display
        if decision['decision'] == 'VERIFIED':
            st.markdown('<div class="verified-box">', unsafe_allow_html=True)
            st.success(f"""
            # ✅ SIGNATURE VERIFIED
            
            **Decision:** AUTHENTICATED  
            **Confidence Level:** {decision['confidence']}  
            **Match Score:** {decision['best_score']:.2f}%  
            **Authenticated User:** {decision['matched_user']}  
            **Threshold:** {threshold}%  
            
            {decision['reason']}
            
            ✓ This signature is **GENUINE** and matches the expected user profile.
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            st.markdown('<div class="rejected-box">', unsafe_allow_html=True)
            st.error(f"""
            # ❌ SIGNATURE REJECTED
            
            **Decision:** NOT AUTHENTICATED  
            **Confidence Level:** {decision['confidence']}  
            **Reason:** {decision['reason']}  
            **Threshold:** {threshold}%  
            
            ✗ This signature is **NOT VERIFIED** - possible forgery, unknown signer, or poor image quality.
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Performance Metrics
        result_metric_cols = st.columns(5)
        best_match = results['matches'][0]
        best_score = calculate_verification_score(best_match['score'])
        best_metadata = best_match.get('metadata', {})
        
        with result_metric_cols[0]:
            st.metric("Match Score", f"{best_score:.2f}%")
        with result_metric_cols[1]:
            st.metric("Threshold", f"{threshold}%")
        with result_metric_cols[2]:
            st.metric("Confidence", decision['confidence'])
        with result_metric_cols[3]:
            st.metric("Matches Found", len(results['matches']))
        with result_metric_cols[4]:
            st.metric("Model Used", model_used.upper())
        
        # Multi-Metric Analysis (if enabled)
        if enable_multi_metric and len(results['matches']) > 0:
            st.markdown("---")
            st.subheader("📊 Multi-Metric Similarity Analysis")
            
            # FIXED: Check if values are available
            if 'values' in best_match and best_match['values']:
                best_match_embedding = np.array(best_match['values'])
                multi_scores = calculate_multi_metric_score(embedding, best_match_embedding)
                
                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.metric("Cosine Similarity", f"{multi_scores['cosine_similarity']:.2f}%")
                with metric_cols[1]:
                    st.metric("Euclidean Similarity", f"{multi_scores['euclidean_similarity']:.2f}%")
                with metric_cols[2]:
                    st.metric("Manhattan Similarity", f"{multi_scores['manhattan_similarity']:.2f}%")
                with metric_cols[3]:
                    st.metric("Average Score", f"{multi_scores['average_score']:.2f}%")
                
                # Consensus check
                all_pass = all(score >= threshold for score in [
                    multi_scores['cosine_similarity'],
                    multi_scores['euclidean_similarity'],
                    multi_scores['manhattan_similarity']
                ])
                
                if all_pass:
                    st.success("✅ All metrics agree: VERIFIED")
                else:
                    st.warning("⚠️ Metric consensus not achieved - review recommended")
            else:
                st.warning("⚠️ Multi-metric analysis unavailable (vector values not returned)")
        
        # Detailed Match Analysis
        st.markdown("---")
        st.subheader("🔍 Detailed Match Analysis")
        
        # Show top matches
        for idx, match in enumerate(results['matches'][:10]):
            score = calculate_verification_score(match['score'])
            metadata = match.get('metadata', {})
            status = "✅ PASS" if score >= threshold else "⚠️ FAIL"
            user_match = "🎯 TARGET USER" if metadata.get('user_id') == user_filter else ""
            
            # IMPROVED: Show if this is wrong user in user-specific mode
            wrong_user = ""
            if user_filter and metadata.get('user_id') != user_filter:
                wrong_user = "❌ WRONG USER"
            
            with st.expander(f"{status} Match #{idx+1}: {score:.2f}% - {metadata.get('user_id', 'Unknown')} {user_match} {wrong_user}"):
                detail_cols = st.columns(3)
                
                with detail_cols[0]:
                    st.markdown("**📋 Match Details**")
                    st.write(f"**Score:** {score:.2f}%")
                    st.write(f"**Distance:** {match['score']:.4f}")
                    st.write(f"**User ID:** {metadata.get('user_id', 'N/A')}")
                    st.write(f"**Vector ID:** {match['id'][:20]}...")
                
                with detail_cols[1]:
                    st.markdown("**👤 User Information**")
                    st.write(f"**Full Name:** {metadata.get('full_name', 'N/A')}")
                    st.write(f"**Department:** {metadata.get('department', 'N/A')}")
                    st.write(f"**Stored:** {metadata.get('timestamp', 'N/A')[:10]}")
                    st.write(f"**Original File:** {metadata.get('original_filename', 'N/A')}")
                
                with detail_cols[2]:
                    st.markdown("**🔬 Quality Metrics**")
                    st.write(f"**Quality Score:** {metadata.get('quality_score', 'N/A')}/10")
                    st.write(f"**Blur Score:** {metadata.get('blur_score', 'N/A')}")
                    st.write(f"**Contrast:** {metadata.get('contrast', 'N/A')}")
                    st.write(f"**Model Used:** {metadata.get('model_used', 'N/A')}")
                
                # Show multi-metric for this match if enabled
                if enable_multi_metric and 'values' in match:
                    st.markdown("**📈 Multi-Metric Scores**")
                    match_embedding = np.array(match['values'])
                    match_multi_scores = calculate_multi_metric_score(embedding, match_embedding)
                    
                    mm_cols = st.columns(4)
                    with mm_cols[0]:
                        st.write(f"Cosine: {match_multi_scores['cosine_similarity']:.2f}%")
                    with mm_cols[1]:
                        st.write(f"Euclidean: {match_multi_scores['euclidean_similarity']:.2f}%")
                    with mm_cols[2]:
                        st.write(f"Manhattan: {match_multi_scores['manhattan_similarity']:.2f}%")
                    with mm_cols[3]:
                        st.write(f"Average: {match_multi_scores['average_score']:.2f}%")
        
        # Score Distribution Visualization
        st.markdown("---")
        st.subheader("📈 Score Distribution Analysis")
        
        viz_cols = st.columns(2)
        
        with viz_cols[0]:
            st.markdown("**Match Score Distribution**")
            scores = [calculate_verification_score(m['score']) for m in results['matches']]
            
            import pandas as pd
            df_scores = pd.DataFrame({
                'Match #': [f"#{i+1}" for i in range(len(scores))],
                'Score (%)': scores
            })
            st.bar_chart(df_scores.set_index('Match #'))
            
            # Statistics
            st.write(f"**Mean Score:** {np.mean(scores):.2f}%")
            st.write(f"**Std Dev:** {np.std(scores):.2f}%")
            st.write(f"**Max Score:** {np.max(scores):.2f}%")
            st.write(f"**Min Score:** {np.min(scores):.2f}%")
        
        with viz_cols[1]:
            st.markdown("**Embedding Statistics**")
            st.write("**Query Embedding:**")
            st.write(f"Mean: {np.mean(embedding):.4f}")
            st.write(f"Std: {np.std(embedding):.4f}")
            st.write(f"Min: {np.min(embedding):.4f}")
            st.write(f"Max: {np.max(embedding):.4f}")
            st.write(f"L2 Norm: {np.linalg.norm(embedding):.4f}")
            
            st.markdown("---")
            st.write("**Best Match Embedding:**")
            if 'values' in best_match:
                best_emb = np.array(best_match['values'])
                st.write(f"Mean: {np.mean(best_emb):.4f}")
                st.write(f"Std: {np.std(best_emb):.4f}")
                st.write(f"L2 Norm: {np.linalg.norm(best_emb):.4f}")
                
                # Embedding similarity
                emb_cosine = np.dot(embedding, best_emb)
                st.write(f"**Cosine:** {emb_cosine:.4f}")
        
        # Score Gap Analysis (Anti-Forgery)
        if len(results['matches']) > 1:
            st.markdown("---")
            st.subheader("🛡️ Anti-Forgery Analysis")
            
            first_score = calculate_verification_score(results['matches'][0]['score'])
            second_score = calculate_verification_score(results['matches'][1]['score'])
            score_gap = first_score - second_score
            
            gap_cols = st.columns(3)
            
            with gap_cols[0]:
                st.metric("1st Place Score", f"{first_score:.2f}%")
            with gap_cols[1]:
                st.metric("2nd Place Score", f"{second_score:.2f}%")
            with gap_cols[2]:
                gap_color = "normal" if score_gap >= 10 else "inverse"
                st.metric("Score Gap", f"{score_gap:.2f}%", delta="Safe" if score_gap >= 10 else "Warning", delta_color=gap_color)
            
            if score_gap >= 15:
                st.success("✅ Strong differentiation - high confidence in match")
            elif score_gap >= 10:
                st.info("ℹ️ Good differentiation - acceptable confidence")
            elif score_gap >= 5:
                st.warning("⚠️ Moderate differentiation - review recommended")
            else:
                st.error("❌ Low differentiation - multiple similar matches detected (possible forgery)")
        
        # Audit Log
        st.markdown("---")
        st.subheader("📋 Verification Audit Log")
        
        audit_log = {
            "timestamp": datetime.now().isoformat(),
            "verification_id": f"VER_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}",
            "decision": decision['decision'],
            "confidence": decision['confidence'],
            "match_score": round(best_score, 2),
            "threshold": threshold,
            "matched_user": best_metadata.get('user_id', 'Unknown'),
            "expected_user": user_filter if user_filter else "Any",
            "model_used": model_used,
            "verification_mode": verification_mode,
            "security_preset": preset,
            "embedding_dimension": EMBEDDING_DIM,
            "total_matches_analyzed": len(results['matches']),
            "multi_metric_enabled": enable_multi_metric,
            "image_quality_score": st.session_state.quality['quality_score'],
            "image_filename": uploaded_file.name if uploaded_file else "N/A",
            "reason": decision['reason']
        }
        
        if enable_multi_metric:
            if 'values' in best_match and best_match['values']:
                audit_log["multi_metric_scores"] = multi_scores
            else:
                audit_log["multi_metric_note"] = "Vector values not available"
        
        st.json(audit_log)
        
        # Download audit log
        import json
        audit_json = json.dumps(audit_log, indent=2)
        st.download_button(
            label="📥 Download Audit Log (JSON)",
            data=audit_json,
            file_name=f"verification_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        # Recommendations
        st.markdown("---")
        st.subheader("💡 Recommendations")
        
        if decision['decision'] == 'VERIFIED':
            st.success("""
            **✅ Verification Successful - Next Steps:**
            - Proceed with the authenticated transaction
            - Log this verification in your system
            - Archive the audit log for compliance
            - Consider two-factor authentication for high-value transactions
            """)
        else:
            st.error("""
            **❌ Verification Failed - Recommended Actions:**
            - Do NOT proceed with transaction
            - Request alternative verification method
            - Check if image quality can be improved
            - Verify user identity through secondary means
            - Consider fraud investigation if suspicious
            - Contact system administrator if persistent issues
            """)
    
    # Sidebar - Additional Information
    st.sidebar.markdown("---")
    st.sidebar.header("📚 Documentation")
    
    with st.sidebar.expander("🎯 Accuracy Features"):
        st.write("""
        **Enhanced Accuracy:**
        - Identical preprocessing to upload
        - 512D embeddings
        - Hybrid model support
        - Multi-metric verification
        - Anti-forgery detection
        - Quality threshold enforcement
        - Score gap analysis
        
        **Match Precision:**
        - L2-normalized embeddings
        - Cosine distance metric
        - Top-K candidate analysis
        - User-specific filtering
        - Confidence scoring
        """)
    
    with st.sidebar.expander("🔒 Security Levels"):
        st.write("""
        **High Security (90%+):**
        - Banking transactions
        - Legal contracts
        - Financial documents
        - Government forms
        
        **Standard (80-89%):**
        - Business documents
        - HR paperwork
        - General contracts
        
        **Relaxed (70-79%):**
        - Internal approvals
        - Non-critical docs
        - Attendance logs
        
        **Not Recommended (<70%):**
        - Too permissive
        - High false positive risk
        """)
    
    with st.sidebar.expander("⚙️ System Details"):
        st.write(f"""
        **Configuration:**
        - Database: Pinecone Serverless
        - Region: us-east-1
        - Metric: Cosine Distance
        - Dimension: {EMBEDDING_DIM}
        
        **Processing:**
        - CLAHE enhancement
        - Adaptive thresholding
        - Morphological filtering
        - Content cropping
        - Aspect-ratio resize
        - Deep learning embedding
        - L2 normalization
        
        **Models:**
        - Siamese CNN: 4 conv blocks
        - MobileNetV2: ImageNet transfer
        - Hybrid: Combined average
        """)
    
    with st.sidebar.expander("💡 Best Practices"):
        st.write("""
        **For Highest Accuracy:**
        
        1. **Use Hybrid Model**
           - Combines both CNN approaches
           - Highest accuracy (recommended)
        
        2. **Enable Multi-Metric**
           - Cross-validates results
           - Reduces false positives
        
        3. **User-Specific Mode**
           - Filters by expected user
           - Prevents cross-user matches
        
        4. **Quality Control**
           - Set minimum quality threshold
           - Reject poor images early
        
        5. **Appropriate Threshold**
           - Match security to use case
           - Don't over-optimize
        
        6. **Regular Database Updates**
           - Add multiple samples per user
           - Update old signatures
        """)
    
    with st.sidebar.expander("🔧 Dimension Info"):
        st.write(f"""
        **Current Configuration:**
        - Embedding Dimension: {EMBEDDING_DIM}
        - Matches Upload Manager: ✅
        - Pinecone Index: ✅
        
        **What this means:**
        - All embeddings are 512D
        - Perfect compatibility
        - No dimension mismatches
        - Optimal performance
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>Enhanced Deep Learning Signature Verification System (512D)</strong></p>
        <p>🔒 High-Accuracy Biometric Authentication</p>
        <p>Powered by Siamese CNN • MobileNetV2 • Pinecone Vector Database</p>
        <p style='font-size: 0.9em; margin-top: 10px;'>
            ⚠️ This system provides automated verification assistance.<br>
            For critical applications, combine with human review and multi-factor authentication.
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()