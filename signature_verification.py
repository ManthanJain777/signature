"""
Professional Signature Verification System
FIXED: Better signature recognition and preprocessing
"""

import cv2
import numpy as np
import streamlit as st
from pinecone import Pinecone
from datetime import datetime

# --------- TensorFlow / MobileNet imports ----------
try:
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
    TF_AVAILABLE = True
except Exception as e:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = e

# Parameters
TARGET_SIZE = (400, 150)
EMBEDDING_DIM = 512

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
        st.error("❌ No signature database found. Please add signatures first.")
        st.stop()
    
    return pc.Index(index_name)

def preprocess_signature(img):
    """Improved preprocessing to better preserve signatures"""
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    
    # Method 1: Try adaptive thresholding first
    img_thresh = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2)  # Reduced block size for better detail
    
    # Check if we have enough signature content
    signature_pixels = np.sum(img_thresh > 0)
    total_pixels = img_thresh.size
    
    # If signature is too faint, try alternative methods
    if signature_pixels < total_pixels * 0.01:  # Less than 1% signature pixels
        # Method 2: Try Otsu's thresholding
        _, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        signature_pixels = np.sum(img_thresh > 0)
        if signature_pixels < total_pixels * 0.01:
            # Method 3: Try contrast enhancement + thresholding
            img_enhanced = cv2.equalizeHist(img_gray)
            _, img_thresh = cv2.threshold(img_enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Noise removal (gentle)
    kernel = np.ones((3,3), np.uint8)
    img_clean = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
    
    # Resize to target size
    resized = cv2.resize(img_clean, TARGET_SIZE)
    
    return resized

def check_signature_visibility(preprocessed_img):
    """Check if signature is visible after preprocessing"""
    signature_pixels = np.sum(preprocessed_img > 0)
    total_pixels = preprocessed_img.size
    visibility_ratio = signature_pixels / total_pixels
    
    return visibility_ratio

@st.cache_resource(show_spinner=False)
def load_feature_model():
    """Load MobileNetV2"""
    if not TF_AVAILABLE:
        raise RuntimeError(f"TensorFlow not available: {TF_IMPORT_ERROR}")

    base_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
    return base_model

def _cnn_embedding(preprocessed_img: np.ndarray) -> np.ndarray:
    """CNN embedding"""
    model = load_feature_model()
    img_rgb = cv2.cvtColor(preprocessed_img, cv2.COLOR_GRAY2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224)).astype("float32")
    
    x = np.expand_dims(img_resized, axis=0)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    emb = model.predict(x, verbose=0)[0]
    
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm

    if emb.shape[0] >= EMBEDDING_DIM:
        emb = emb[:EMBEDDING_DIM]
    else:
        emb = np.pad(emb, (0, EMBEDDING_DIM - emb.shape[0]))

    return emb.astype("float32")

def _fallback_embedding(preprocessed_img: np.ndarray) -> np.ndarray:
    """Fallback embedding"""
    features = []
    features.extend(preprocessed_img.flatten())
    
    hist = cv2.calcHist([preprocessed_img], [0], None, [64], [0, 256])
    features.extend(hist.flatten())
    
    features.extend([
        float(np.mean(preprocessed_img)),
        float(np.std(preprocessed_img)),
        float(np.median(preprocessed_img)),
        float(np.min(preprocessed_img)),
        float(np.max(preprocessed_img)),
    ])
    
    gx = cv2.Sobel(preprocessed_img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(preprocessed_img, cv2.CV_32F, 0, 1)
    magnitude, _ = cv2.cartToPolar(gx, gy)
    features.extend([float(np.mean(magnitude)), float(np.std(magnitude))])
    
    arr = np.array(features, dtype=np.float32)
    
    if len(arr) >= EMBEDDING_DIM:
        emb = arr[:EMBEDDING_DIM]
    else:
        emb = np.pad(arr, (0, EMBEDDING_DIM - len(arr)))
    
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm

    return emb.astype("float32")

def generate_signature_embedding(preprocessed_img):
    """Generate embedding"""
    if TF_AVAILABLE:
        try:
            emb = _cnn_embedding(preprocessed_img)
        except Exception as e:
            st.warning(f"⚠️ CNN embedding failed ({e}). Falling back.")
            emb = _fallback_embedding(preprocessed_img)
    else:
        emb = _fallback_embedding(preprocessed_img)
    
    return emb.tolist()

def find_similar_signatures(index, query_embedding, top_k=5, user_filter=None):
    """Find similar signatures in database"""
    try:
        if user_filter:
            results = index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter={"user_id": {"$eq": user_filter}}
            )
        else:
            results = index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
        return results
    except Exception as e:
        st.error(f"❌ Search failed: {e}")
        return {'matches': []}

def main():
    st.set_page_config(page_title="Signature Verification Pro", layout="wide")
    st.title("🔐 Professional Signature Verification")
    st.write("**Verify real signatures against your secure database**")
    
    # Show embedding method status
    if TF_AVAILABLE:
        st.sidebar.success("✅ CNN embeddings enabled")
    else:
        st.sidebar.warning("⚠️ Using fallback embeddings")
    
    # Initialize Pinecone
    try:
        index = init_pinecone()
        st.sidebar.success("✅ Connected to Signature Database")
        
        # Show database statistics
        stats = index.describe_index_stats()
        st.sidebar.header("📊 Database Info")
        st.sidebar.metric("Total Signatures", stats['total_vector_count'])
        st.sidebar.metric("Dimension", stats['dimension'])
        
    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
        return

    # Main verification interface
    st.header("🔍 Signature Verification")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Signature upload section
        st.subheader("1. Upload Signature")
        query_file = st.file_uploader(
            "Choose signature image to verify", 
            type=['png','jpg','jpeg'],
            help="Upload a clear image of the signature you want to verify"
        )
        
        if query_file:
            # Display the uploaded image
            st.image(query_file, caption="Original Signature", use_column_width=True)
            
            # Convert for processing
            file_bytes = np.asarray(bytearray(query_file.read()), dtype=np.uint8)
            query_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Store for later use
            st.session_state.query_img = query_img
            
            # Show processed version with visibility check
            preprocessed = preprocess_signature(query_img)
            visibility_ratio = check_signature_visibility(preprocessed)
            
            processed_display = (preprocessed * 255).astype(np.uint8)
            st.image(processed_display, caption="Processed Signature", use_column_width=True)
            
            # Show visibility warning if needed
            if visibility_ratio < 0.01:
                st.warning("⚠️ Signature appears very faint after processing. Try uploading a clearer image with better contrast.")
            elif visibility_ratio < 0.05:
                st.info("ℹ️ Signature visibility is low. Consider using a darker signature or better lighting.")
    
    with col2:
        # Verification settings
        st.subheader("2. Verification Settings")
        
        verification_mode = st.radio(
            "Verification Mode",
            ["General Search", "User-Specific Verification"],
            help="Search all signatures or verify against a specific user"
        )
        
        if verification_mode == "User-Specific Verification":
            user_filter = st.text_input(
                "Expected User ID",
                placeholder="e.g., john_doe_001",
                help="Enter the User ID you expect this signature to match"
            )
        else:
            user_filter = None
        
        st.subheader("3. Security Settings")
        min_similarity = st.slider(
            "Minimum Similarity Threshold (%)", 
            min_value=50, 
            max_value=95, 
            value=75,
            help="Signatures below this similarity score will be rejected"
        )
        
        top_k = st.slider(
            "Number of Matches to Check",
            min_value=1,
            max_value=10,
            value=5,
            help="How many similar signatures to compare against"
        )
    
    # Verification action
    if st.button("🚀 Start Verification", type="primary", use_container_width=True):
        if 'query_img' not in st.session_state:
            st.error("❌ Please upload a signature image first")
            return
            
        with st.spinner("🔍 Analyzing signature and searching database..."):
            # Preprocess and generate embedding
            query_img = st.session_state.query_img
            preprocessed = preprocess_signature(query_img)
            
            # Check signature visibility again
            visibility_ratio = check_signature_visibility(preprocessed)
            if visibility_ratio < 0.005:  # Less than 0.5% signature pixels
                st.error("❌ Signature not detected in image. Please upload a clearer signature.")
                return
            
            query_embedding = generate_signature_embedding(preprocessed)
            
            # Search in database
            results = find_similar_signatures(index, query_embedding, top_k=top_k, user_filter=user_filter)
            
            # Display results
            st.header("📊 Verification Results")
            
            if not results['matches']:
                st.error("❌ No matching signatures found in the database")
                if visibility_ratio < 0.01:
                    st.info("💡 The signature appears very faint. Try uploading a darker signature or adjusting image contrast.")
                if user_filter:
                    st.info(f"Try removing the user filter or check if user '{user_filter}' exists")
                return
            
            # Best match analysis
            best_match = results['matches'][0]
            best_similarity = (1 - best_match['score']) * 100
            best_metadata = best_match.get('metadata', {})
            
            # Result summary
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Similarity Score",
                    f"{best_similarity:.1f}%",
                    delta="PASS" if best_similarity >= min_similarity else "FAIL",
                    delta_color="normal" if best_similarity >= min_similarity else "inverse"
                )
            
            with col2:
                st.metric("Threshold", f"{min_similarity}%")
            
            with col3:
                status = "✅ VERIFIED" if best_similarity >= min_similarity else "❌ REJECTED"
                st.metric("Status", status)
            
            with col4:
                confidence = "High" if best_similarity >= 85 else "Medium" if best_similarity >= 70 else "Low"
                st.metric("Confidence", confidence)
            
            # Decision card
            if best_similarity >= min_similarity:
                st.success(f"""
                **✅ SIGNATURE VERIFIED**
                
                **Match Details:**
                - **User:** {best_metadata.get('user_id', 'Unknown')}
                - **Similarity:** {best_similarity:.1f}%
                - **Full Name:** {best_metadata.get('full_name', 'N/A')}
                - **Department:** {best_metadata.get('department', 'N/A')}
                """)
            else:
                st.error(f"""
                **❌ SIGNATURE REJECTED**
                
                **Reason:** Similarity score ({best_similarity:.1f}%) below required threshold ({min_similarity}%)
                
                **Best Match:** {best_metadata.get('user_id', 'Unknown')} ({best_similarity:.1f}%)
                """)
            
            # Detailed matches
            st.subheader("🔍 Detailed Match Analysis")
            
            for i, match in enumerate(results['matches']):
                similarity_percent = (1 - match['score']) * 100
                metadata = match.get('metadata', {})
                
                with st.expander(f"Match #{i+1}: {similarity_percent:.1f}% - {metadata.get('user_id', 'Unknown')}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Match Details:**")
                        st.write(f"• **Similarity:** {similarity_percent:.1f}%")
                        st.write(f"• **User ID:** {metadata.get('user_id', 'N/A')}")
                        st.write(f"• **Full Name:** {metadata.get('full_name', 'N/A')}")
                        st.write(f"• **Signature ID:** {match['id'][:20]}...")
                    
                    with col2:
                        st.write("**User Information:**")
                        st.write(f"• **Department:** {metadata.get('department', 'N/A')}")
                        st.write(f"• **Stored:** {metadata.get('timestamp', 'N/A')}")
                        st.write(f"• **Quality Score:** {metadata.get('quality_score', 'N/A')}")
            
            # Audit log
            st.subheader("📋 Audit Log")
            audit_data = {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Verification Result": "VERIFIED" if best_similarity >= min_similarity else "REJECTED",
                "Similarity Score": f"{best_similarity:.1f}%",
                "Matched User": best_metadata.get('user_id', 'Unknown'),
                "Threshold Used": f"{min_similarity}%",
                "Database Matches": len(results['matches'])
            }
            
            st.json(audit_data)

    # Quick actions in sidebar
    st.sidebar.header("⚡ Quick Actions")
    
    if st.sidebar.button("🔄 Refresh", use_container_width=True):
        st.rerun()
    
    st.sidebar.header("ℹ️ Tips for Better Recognition")
    st.sidebar.info("""
    **For best results:**
    - Use **dark signatures** on white background
    - Ensure **good lighting** when capturing
    - Avoid **blurry images**
    - Use **high contrast** between signature and background
    - **File types:** PNG, JPG, JPEG
    """)

if __name__ == "__main__":
    main()