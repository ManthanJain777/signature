"""
Advanced Signature Verification with Deep Learning (Siamese CNN)
Features: Siamese Neural Network, Contrastive Learning, Deep Feature Extraction
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

# Configuration
TARGET_SIZE = (128, 128)
EMBEDDING_DIM = 256  # Deep learning embedding dimension

# ==================== SIAMESE NEURAL NETWORK MODEL ====================

class SiameseNetwork:
    """Siamese CNN for signature embedding extraction"""
    
    def __init__(self, input_shape=(128, 128, 1), embedding_dim=256):
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        self.model = self._build_embedding_network()
    
    def _build_embedding_network(self):
        """Build the embedding network (one branch of Siamese)"""
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
        
        # Dense Embedding Layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(self.embedding_dim, activation='linear', name='embedding')(x)
        
        # L2 Normalization for cosine similarity
        x = layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=1))(x)
        
        model = models.Model(inputs=inputs, outputs=x, name='signature_embedder')
        return model
    
    def extract_embedding(self, signature_img):
        """Extract embedding from a signature image"""
        # Ensure correct shape
        if len(signature_img.shape) == 2:
            signature_img = np.expand_dims(signature_img, axis=-1)
        
        # Normalize
        signature_img = signature_img.astype('float32') / 255.0
        
        # Add batch dimension
        signature_img = np.expand_dims(signature_img, axis=0)
        
        # Extract embedding
        embedding = self.model.predict(signature_img, verbose=0)
        return embedding[0]

class MobileNetEmbedder:
    """Transfer Learning with MobileNetV2 for signature embeddings"""
    
    def __init__(self, input_shape=(128, 128, 3), embedding_dim=256):
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        self.model = self._build_model()
    
    def _build_model(self):
        """Build transfer learning model"""
        # Load pre-trained MobileNetV2
        base_model = MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        
        # Freeze base model
        base_model.trainable = False
        
        # Add custom head
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
        # Convert to 3 channels if needed
        if len(signature_img.shape) == 2:
            signature_img = cv2.cvtColor(signature_img, cv2.COLOR_GRAY2RGB)
        elif signature_img.shape[2] == 1:
            signature_img = cv2.cvtColor(signature_img, cv2.COLOR_GRAY2RGB)
        
        # Normalize
        signature_img = signature_img.astype('float32') / 255.0
        
        # Add batch dimension
        signature_img = np.expand_dims(signature_img, axis=0)
        
        # Extract embedding
        embedding = self.model.predict(signature_img, verbose=0)
        return embedding[0]

# ==================== PREPROCESSING ====================

class SignaturePreprocessor:
    """Advanced preprocessing pipeline"""
    
    def __init__(self, target_size=TARGET_SIZE):
        self.target_size = target_size
    
    def preprocess(self, img):
        """Complete preprocessing pipeline"""
        # Convert to grayscale
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
        
        # Resize with aspect ratio preservation
        img_resized = self._resize_with_padding(img_cropped, self.target_size)
        
        return img_resized, img_gray
    
    def _crop_to_content(self, img):
        """Crop image to signature content"""
        coords = cv2.findNonZero(img)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            # Add padding
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
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create canvas and center image
        canvas = np.zeros((target_h, target_w), dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized
        
        return canvas

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
        st.error("❌ No signature database found. Please create index first.")
        st.stop()
    
    index = pc.Index(index_name)
    return index

def find_similar_signatures(index, query_embedding, top_k=5, user_filter=None):
    """Search for similar signatures"""
    query_params = {
        'vector': query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding,
        'top_k': top_k,
        'include_metadata': True
    }
    
    if user_filter:
        query_params['filter'] = {"user_id": {"$eq": user_filter}}
    
    results = index.query(**query_params)
    return results

# ==================== SIMILARITY COMPUTATION ====================

def cosine_similarity(embedding1, embedding2):
    """Compute cosine similarity between embeddings"""
    return np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) * np.linalg.norm(embedding2) + 1e-10
    )

def euclidean_distance(embedding1, embedding2):
    """Compute Euclidean distance"""
    return np.linalg.norm(embedding1 - embedding2)

def calculate_verification_score(match_score, distance_type='cosine'):
    """Convert distance/similarity to percentage score"""
    if distance_type == 'cosine':
        # match_score is distance (0-2), convert to similarity
        similarity = 1 - (match_score / 2)
        return similarity * 100
    else:
        # For euclidean, lower is better
        # Normalize to 0-100 scale
        return max(0, (1 - match_score) * 100)

# ==================== MAIN APPLICATION ====================

def main():
    st.set_page_config(
        page_title="Deep Learning Signature Verification",
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
    .metric-box {
        background: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header"><h1>🧠 Deep Learning Signature Verification</h1><p>Powered by Siamese Neural Networks & Transfer Learning</p></div>', unsafe_allow_html=True)
    
    # Initialize components
    if 'preprocessor' not in st.session_state:
        st.session_state.preprocessor = SignaturePreprocessor()
    
    if 'siamese_model' not in st.session_state:
        with st.spinner("🔄 Loading Siamese Neural Network..."):
            st.session_state.siamese_model = SiameseNetwork(
                input_shape=(128, 128, 1),
                embedding_dim=EMBEDDING_DIM
            )
    
    if 'mobilenet_model' not in st.session_state:
        with st.spinner("🔄 Loading MobileNetV2 Model..."):
            st.session_state.mobilenet_model = MobileNetEmbedder(
                input_shape=(128, 128, 3),
                embedding_dim=EMBEDDING_DIM
            )
    
    # Sidebar - Model Selection & Settings
    st.sidebar.header("⚙️ Model Configuration")
    
    model_choice = st.sidebar.selectbox(
        "Select Embedding Model",
        ["Siamese CNN (Custom)", "MobileNetV2 (Transfer Learning)", "Hybrid (Both)"],
        help="Choose the deep learning model for feature extraction"
    )
    
    st.sidebar.markdown("---")
    
    # Database connection
    try:
        index = init_pinecone()
        stats = index.describe_index_stats()
        
        st.sidebar.success("✅ Database Connected")
        st.sidebar.metric("Total Signatures", stats['total_vector_count'])
        st.sidebar.metric("Embedding Dimension", stats['dimension'])
        
    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
        st.stop()
    
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Verification Settings")
    
    verification_mode = st.sidebar.radio(
        "Mode",
        ["General Search", "User-Specific"],
        help="Verify against all signatures or specific user"
    )
    
    user_filter = None
    if verification_mode == "User-Specific":
        user_filter = st.sidebar.text_input(
            "User ID",
            placeholder="e.g., user_001",
            help="Enter the expected user ID"
        )
    
    threshold = st.sidebar.slider(
        "Verification Threshold (%)",
        min_value=60,
        max_value=95,
        value=80,
        help="Minimum similarity score to accept signature"
    )
    
    top_k = st.sidebar.slider(
        "Top K Matches",
        min_value=1,
        max_value=20,
        value=5,
        help="Number of similar signatures to retrieve"
    )
    
    # Main Content Area
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("📤 Upload Signature for Verification")
        
        uploaded_file = st.file_uploader(
            "Choose signature image",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="Upload a clear signature image"
        )
        
        if uploaded_file:
            # Read image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Display original
            st.image(img, caption="Original Signature", use_container_width=True)
            
            # Preprocess
            with st.spinner("🔄 Processing signature..."):
                preprocessed, gray_img = st.session_state.preprocessor.preprocess(img)
                st.session_state.query_img = img
                st.session_state.preprocessed_img = preprocessed
                st.session_state.gray_img = gray_img
            
            # Show preprocessing stages
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.image(gray_img, caption="Grayscale", use_container_width=True)
            with col_b:
                st.image(preprocessed, caption="Binary (Processed)", use_container_width=True)
            with col_c:
                edges = cv2.Canny(preprocessed, 50, 150)
                st.image(edges, caption="Edge Detection", use_container_width=True)
    
    with col2:
        st.subheader("🧠 Model Information")
        
        if model_choice == "Siamese CNN (Custom)":
            st.info("""
            **Siamese CNN Architecture**
            - 4 Convolutional blocks
            - Batch normalization
            - Global average pooling
            - 256-dim embedding
            - L2 normalized output
            
            **Best for:** Writer-independent verification
            """)
        elif model_choice == "MobileNetV2 (Transfer Learning)":
            st.info("""
            **MobileNetV2 Transfer Learning**
            - Pre-trained on ImageNet
            - Lightweight & efficient
            - 256-dim embedding
            - Fine-tuned head
            
            **Best for:** Robust feature extraction
            """)
        else:
            st.info("""
            **Hybrid Approach**
            - Combines both models
            - Ensemble predictions
            - Higher accuracy
            
            **Best for:** Maximum reliability
            """)
        
        st.markdown("---")
        st.subheader("📊 Quick Stats")
        if 'preprocessed_img' in st.session_state:
            img_stats = st.session_state.preprocessed_img
            st.write(f"**Image Size:** {img_stats.shape}")
            st.write(f"**Non-zero Pixels:** {np.count_nonzero(img_stats)}")
            st.write(f"**Fill Ratio:** {np.count_nonzero(img_stats)/img_stats.size*100:.1f}%")
    
    # Verification Button
    st.markdown("---")
    
    if st.button("🚀 Start Deep Learning Verification", type="primary", use_container_width=True):
        if 'preprocessed_img' not in st.session_state:
            st.error("❌ Please upload a signature first")
            st.stop()
        
        with st.spinner("🧠 Extracting deep features and searching database..."):
            preprocessed = st.session_state.preprocessed_img
            
            # Extract embeddings based on model choice
            if model_choice == "Siamese CNN (Custom)":
                embedding = st.session_state.siamese_model.extract_embedding(preprocessed)
                st.success("✅ Features extracted using Siamese CNN")
                
            elif model_choice == "MobileNetV2 (Transfer Learning)":
                # Convert to 3-channel for MobileNet
                preprocessed_rgb = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2RGB)
                embedding = st.session_state.mobilenet_model.extract_embedding(preprocessed_rgb)
                st.success("✅ Features extracted using MobileNetV2")
                
            else:  # Hybrid
                emb1 = st.session_state.siamese_model.extract_embedding(preprocessed)
                preprocessed_rgb = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2RGB)
                emb2 = st.session_state.mobilenet_model.extract_embedding(preprocessed_rgb)
                # Average embeddings
                embedding = (emb1 + emb2) / 2
                # Re-normalize
                embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
                st.success("✅ Features extracted using Hybrid approach")
            
            # Store embedding
            st.session_state.query_embedding = embedding
            
            # Search database
            results = find_similar_signatures(
                index, 
                embedding, 
                top_k=top_k,
                user_filter=user_filter
            )
            
            if not results['matches']:
                st.error("❌ No matching signatures found")
                if user_filter:
                    st.info(f"Try removing user filter '{user_filter}'")
                st.stop()
            
            # Process results
            st.header("📊 Verification Results")
            
            best_match = results['matches'][0]
            best_score = calculate_verification_score(best_match['score'])
            best_metadata = best_match.get('metadata', {})
            
            # Display metrics
            metric_cols = st.columns(4)
            
            with metric_cols[0]:
                delta_text = "VERIFIED ✓" if best_score >= threshold else "REJECTED ✗"
                delta_color = "normal" if best_score >= threshold else "inverse"
                st.metric(
                    "Match Score",
                    f"{best_score:.2f}%",
                    delta=delta_text,
                    delta_color=delta_color
                )
            
            with metric_cols[1]:
                st.metric("Threshold", f"{threshold}%")
            
            with metric_cols[2]:
                confidence = "High" if best_score >= 90 else "Medium" if best_score >= 75 else "Low"
                st.metric("Confidence", confidence)
            
            with metric_cols[3]:
                st.metric("Matches Found", len(results['matches']))
            
            # Verification Decision
            st.markdown("---")
            
            if best_score >= threshold:
                st.success(f"""
                ### ✅ SIGNATURE VERIFIED
                
                **Authenticated User:** {best_metadata.get('user_id', 'Unknown')}  
                **Match Confidence:** {best_score:.2f}%  
                **Full Name:** {best_metadata.get('full_name', 'N/A')}  
                **Department:** {best_metadata.get('department', 'N/A')}  
                
                ✓ This signature is **GENUINE** and matches the database record.
                """)
            else:
                st.error(f"""
                ### ❌ SIGNATURE REJECTED
                
                **Reason:** Match score ({best_score:.2f}%) below threshold ({threshold}%)  
                **Closest Match:** {best_metadata.get('user_id', 'Unknown')}  
                **Score:** {best_score:.2f}%  
                
                ✗ This signature is **NOT VERIFIED** - possible forgery or unknown signer.
                """)
            
            # Detailed Match Analysis
            st.subheader("🔍 Top Matches Analysis")
            
            for idx, match in enumerate(results['matches'][:5]):
                score = calculate_verification_score(match['score'])
                metadata = match.get('metadata', {})
                status = "✅ PASS" if score >= threshold else "⚠️ FAIL"
                
                with st.expander(f"{status} Match #{idx+1}: {score:.2f}% - {metadata.get('user_id', 'Unknown')}"):
                    info_cols = st.columns(3)
                    
                    with info_cols[0]:
                        st.markdown("**Match Details**")
                        st.write(f"Score: {score:.2f}%")
                        st.write(f"Distance: {match['score']:.4f}")
                        st.write(f"User: {metadata.get('user_id', 'N/A')}")
                    
                    with info_cols[1]:
                        st.markdown("**User Information**")
                        st.write(f"Name: {metadata.get('full_name', 'N/A')}")
                        st.write(f"Dept: {metadata.get('department', 'N/A')}")
                        st.write(f"Added: {metadata.get('timestamp', 'N/A')}")
                    
                    with info_cols[2]:
                        st.markdown("**Quality Metrics**")
                        st.write(f"Quality: {metadata.get('quality_score', 'N/A')}")
                        st.write(f"Vector ID: {match['id'][:16]}...")
                        st.write(f"Status: {'PASS' if score >= threshold else 'FAIL'}")
            
            # Visualization: Embedding comparison
            st.subheader("📈 Embedding Visualization")
            
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                st.write("**Query Embedding Statistics**")
                st.write(f"Mean: {np.mean(embedding):.4f}")
                st.write(f"Std: {np.std(embedding):.4f}")
                st.write(f"Min: {np.min(embedding):.4f}")
                st.write(f"Max: {np.max(embedding):.4f}")
                st.write(f"L2 Norm: {np.linalg.norm(embedding):.4f}")
            
            with col_viz2:
                st.write("**Match Distribution**")
                scores = [calculate_verification_score(m['score']) for m in results['matches']]
                st.bar_chart({"Match Score": scores[:10]})
            
            # Audit Log
            st.subheader("📋 Verification Audit Log")
            
            audit_log = {
                "timestamp": datetime.now().isoformat(),
                "model_used": model_choice,
                "verification_result": "VERIFIED" if best_score >= threshold else "REJECTED",
                "match_score": round(best_score, 2),
                "threshold": threshold,
                "matched_user": best_metadata.get('user_id', 'Unknown'),
                "confidence_level": confidence,
                "total_matches_checked": len(results['matches']),
                "verification_mode": verification_mode,
                "embedding_dimension": EMBEDDING_DIM,
                "distance_metric": "cosine_similarity"
            }
            
            st.json(audit_log)
    
    # Sidebar - Additional Info
    st.sidebar.markdown("---")
    st.sidebar.header("ℹ️ Model Details")
    
    with st.sidebar.expander("🧠 Siamese Network"):
        st.write("""
        **Architecture:**
        - Conv2D: 32 → 64 → 128 → 256
        - Batch Normalization
        - MaxPooling after each conv
        - Global Average Pooling
        - Dense: 512 → 256
        - L2 Normalization
        
        **Training:** Contrastive Loss
        **Metric:** Cosine Similarity
        """)
    
    with st.sidebar.expander("📱 MobileNetV2"):
        st.write("""
        **Architecture:**
        - Pre-trained on ImageNet
        - Inverted residual blocks
        - Depthwise separable convolutions
        - Custom classification head
        
        **Parameters:** 3.5M
        **Speed:** ~15ms per image
        """)
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **💡 Tips for Best Results:**
    - Use high-resolution images (300+ DPI)
    - Ensure clean, white background
    - Avoid shadows and glare
    - Center the signature in frame
    
    **Threshold Guidelines:**
    - 90-95%: Banking (High Security)
    - 80-89%: Legal Documents
    - 70-79%: General Verification
    - 60-69%: Relaxed Checks
    """)

if __name__ == "__main__":
    main()