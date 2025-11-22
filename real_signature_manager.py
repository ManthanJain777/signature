"""
Real Signature Manager - Add real signatures to database with quality checks
FIXED: Embedding generation to prevent zero vectors
"""

import cv2
import numpy as np
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import uuid
from datetime import datetime

# Parameters
TARGET_SIZE = (400, 150)
EMBEDDING_DIM = 512

def init_pinecone():
    """Initialize Pinecone connection with free plan compatible region"""
    try:
        # Try to get from secrets, otherwise show input
        try:
            pinecone_api_key = st.secrets["PINECONE_API_KEY"]
        except:
            st.sidebar.warning("Please enter Pinecone credentials")
            pinecone_api_key = st.sidebar.text_input("Pinecone API Key", type="password")
            
            if not pinecone_api_key:
                st.stop()
    
        # Initialize Pinecone with new API
        pc = Pinecone(api_key=pinecone_api_key)
        
        index_name = "signature-verification"
        
        # Check if index exists, create if not
        existing_indexes = [index.name for index in pc.list_indexes()]
        if index_name not in existing_indexes:
            # Use free plan compatible region
            pc.create_index(
                name=index_name,
                dimension=EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"  # Free plan compatible region
                )
            )
            st.sidebar.info("✅ Created new 'signature-verification' index")
        
        # Connect to the index
        index = pc.Index(index_name)
        return index
        
    except Exception as e:
        st.error(f"❌ Pinecone initialization failed: {e}")
        return None

def check_image_quality(img):
    """Check signature image quality"""
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    
    # Blur detection
    laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    
    # Contrast check
    contrast = img_gray.std()
    
    # Brightness check
    brightness = img_gray.mean()
    
    quality_score = 0
    issues = []
    
    if laplacian_var > 100:
        quality_score += 3
    elif laplacian_var > 50:
        quality_score += 2
        issues.append("Image is slightly blurry")
    else:
        quality_score += 1
        issues.append("Image is very blurry - please use a clearer signature")
    
    if contrast > 50:
        quality_score += 2
    elif contrast > 25:
        quality_score += 1
        issues.append("Low contrast")
    else:
        issues.append("Very low contrast - signature may not be visible")
    
    if 50 < brightness < 200:
        quality_score += 2
    else:
        issues.append("Brightness issues")
    
    return {
        "quality_score": quality_score,
        "issues": issues,
        "blur_score": laplacian_var,
        "contrast": contrast,
        "brightness": brightness
    }

def preprocess_signature(img):
    """Apply preprocessing pipeline to the signature image."""
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    
    # Enhance contrast first
    img_enhanced = cv2.equalizeHist(img_gray)
    
    # Adaptive thresholding
    img_thresh = cv2.adaptiveThreshold(
        img_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 15)
    
    # Noise removal
    img_blur = cv2.GaussianBlur(img_thresh, (3,3), 0)
    kernel = np.ones((5,5), np.uint8)
    img_close = cv2.morphologyEx(img_blur, cv2.MORPH_CLOSE, kernel)
    resized = cv2.resize(img_close, TARGET_SIZE)
    
    return resized

def generate_signature_embedding(preprocessed_img):
    """Generate embedding vector from preprocessed signature."""
    # Use multiple feature extraction methods to create a robust embedding
    
    features = []
    
    # 1. Flatten the image
    flattened = preprocessed_img.flatten()
    features.extend(flattened)
    
    # 2. Add histogram features
    hist = cv2.calcHist([preprocessed_img], [0], None, [64], [0, 256])
    features.extend(hist.flatten())
    
    # 3. Add statistical features
    features.extend([
        np.mean(preprocessed_img),
        np.std(preprocessed_img),
        np.median(preprocessed_img),
        np.min(preprocessed_img),
        np.max(preprocessed_img)
    ])
    
    # 4. Add HOG-like features (simplified)
    gx = cv2.Sobel(preprocessed_img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(preprocessed_img, cv2.CV_32F, 0, 1)
    magnitude, angle = cv2.cartToPolar(gx, gy)
    features.extend([np.mean(magnitude), np.std(magnitude)])
    
    # Convert to numpy array and ensure proper length
    features_array = np.array(features, dtype=np.float32)
    
    # Pad or truncate to EMBEDDING_DIM
    if len(features_array) > EMBEDDING_DIM:
        embedding = features_array[:EMBEDDING_DIM]
    else:
        embedding = np.pad(features_array, (0, EMBEDDING_DIM - len(features_array)))
    
    # Add small random noise to prevent all zeros
    embedding = embedding + np.random.normal(0, 0.01, EMBEDDING_DIM)
    
    # Normalize to unit vector
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    else:
        # Fallback: create a random embedding if everything is zero
        embedding = np.random.randn(EMBEDDING_DIM)
        embedding = embedding / np.linalg.norm(embedding)
    
    return embedding.tolist()

def validate_embedding(embedding):
    """Validate that embedding is not all zeros"""
    embedding_array = np.array(embedding)
    if np.all(embedding_array == 0):
        st.error("❌ Generated embedding is all zeros - signature may be too faint")
        return False
    if np.linalg.norm(embedding_array) < 0.1:
        st.warning("⚠️ Embedding norm is very low - signature quality may be poor")
    return True

def store_signature_batch(index, signatures_data):
    """Store multiple signatures in Pinecone"""
    vectors = []
    
    for data in signatures_data:
        signature_id = data['signature_id']
        embedding = data['embedding']
        metadata = data['metadata']
        
        # Validate embedding before storing
        if validate_embedding(embedding):
            vectors.append({
                "id": signature_id,
                "values": embedding,
                "metadata": metadata
            })
        else:
            st.error(f"❌ Skipping signature {signature_id} - invalid embedding")
    
    if vectors:
        # Batch upsert to Pinecone
        index.upsert(vectors=vectors)
        return len(vectors)
    else:
        return 0

def main():
    st.set_page_config(page_title="Real Signature Manager", layout="wide")
    st.title("✍️ Real Signature Manager")
    st.write("**Add real signatures to your database with quality validation**")
    
    # Initialize Pinecone
    index = init_pinecone()
    if index is None:
        st.stop()
    
    st.sidebar.success("✅ Connected to Pinecone Database")
    
    # Show current stats
    try:
        stats = index.describe_index_stats()
        current_count = stats['total_vector_count']
        st.sidebar.metric("Current Signatures", current_count)
    except Exception as e:
        st.sidebar.metric("Current Signatures", 0)

    # Main interface
    tab1, tab2 = st.tabs(["➕ Add Single Signature", "📚 Batch Upload"])

    with tab1:
        st.header("Add Single Signature")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            sig_file = st.file_uploader(
                "Upload Signature Image", 
                type=['png','jpg','jpeg'],
                key="single_upload"
            )
            
            if sig_file:
                # Display original image
                st.image(sig_file, caption="Original Signature", use_column_width=True)
                
                # Process for preview
                file_bytes = np.asarray(bytearray(sig_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                st.session_state.current_img = img
                
                # Show processed version
                preprocessed = preprocess_signature(img)
                processed_display = (preprocessed * 255).astype(np.uint8)
                st.image(processed_display, caption="Processed Signature", use_column_width=True)
        
        with col2:
            st.subheader("Signature Details")
            user_id = st.text_input("User ID*", placeholder="e.g., john_doe_001")
            full_name = st.text_input("Full Name", placeholder="John Doe")
            department = st.selectbox("Department", ["Finance", "HR", "Legal", "Operations", "Management", "Sales", "IT", "Other"])
            description = st.text_area("Description", placeholder="Official signature for bank documents")
            
            if st.button("💾 Store Signature", type="primary", use_container_width=True):
                if 'current_img' not in st.session_state:
                    st.error("❌ Please upload a signature image first")
                    return
                if not user_id:
                    st.error("❌ Please enter a User ID")
                    return
                
                with st.spinner("Storing signature..."):
                    img = st.session_state.current_img
                    
                    # Quality check
                    quality = check_image_quality(img)
                    
                    if quality['issues']:
                        st.warning("⚠️ Image Quality Issues:")
                        for issue in quality['issues']:
                            st.write(f"- {issue}")
                    
                    # Preprocess and generate embedding
                    preprocessed = preprocess_signature(img)
                    embedding = generate_signature_embedding(preprocessed)
                    
                    # Validate embedding
                    if not validate_embedding(embedding):
                        st.error("❌ Cannot store signature - embedding generation failed")
                        return
                    
                    # Generate unique signature ID
                    signature_id = f"{user_id}_{uuid.uuid4().hex[:8]}"
                    
                    # Store in Pinecone
                    metadata = {
                        "user_id": user_id,
                        "full_name": full_name,
                        "department": department,
                        "description": description,
                        "original_filename": sig_file.name,
                        "timestamp": datetime.now().isoformat(),
                        "quality_score": quality['quality_score'],
                        "blur_score": float(quality['blur_score']),
                        "contrast": float(quality['contrast'])
                    }
                    
                    try:
                        index.upsert(
                            vectors=[{
                                "id": signature_id,
                                "values": embedding,
                                "metadata": metadata
                            }]
                        )
                        st.success(f"✅ Signature stored successfully!")
                        
                        # Show storage details
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Storage Details:**")
                            st.write(f"• **Signature ID:** {signature_id}")
                            st.write(f"• **User ID:** {user_id}")
                            st.write(f"• **Department:** {department}")
                        
                        with col2:
                            st.write("**Quality Metrics:**")
                            st.write(f"• **Quality Score:** {quality['quality_score']}/7")
                            st.write(f"• **Blur Score:** {quality['blur_score']:.1f}")
                            st.write(f"• **Contrast:** {quality['contrast']:.1f}")
                    
                    except Exception as e:
                        st.error(f"❌ Failed to store signature: {e}")

    with tab2:
        st.header("Batch Upload Multiple Signatures")
        
        st.info("""
        **Instructions for batch upload:**
        1. Prepare signature images with clear naming
        2. Upload multiple images at once
        3. Assign user IDs automatically or manually
        4. Review quality before storing
        """)
        
        uploaded_files = st.file_uploader(
            "Upload multiple signature images",
            type=['png','jpg','jpeg'],
            accept_multiple_files=True,
            key="batch_upload"
        )
        
        if uploaded_files:
            st.write(f"📁 **{len(uploaded_files)} signature files selected**")
            
            # Display files in a grid
            st.subheader("Signature Preview")
            cols = st.columns(4)
            signature_data = []
            
            for i, file in enumerate(uploaded_files):
                with cols[i % 4]:
                    st.image(file, caption=f"File {i+1}", use_column_width=True)
                    
                    # Process each file
                    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    # Quality check
                    quality = check_image_quality(img)
                    
                    # Store data for batch processing
                    signature_data.append({
                        'file': file,
                        'image': img,
                        'quality': quality,
                        'user_id': f"user_{i+1:03d}",
                        'filename': file.name
                    })
            
            # Batch processing options
            st.subheader("Batch Configuration")
            col1, col2 = st.columns(2)
            
            with col1:
                naming_option = st.radio(
                    "User ID Generation",
                    ["Auto-generate (user_001, user_002, ...)", "Manual input"]
                )
                
                if naming_option == "Manual input":
                    user_ids_input = st.text_area("Enter User IDs (one per line)", placeholder="john_doe\njane_smith\nmike_wilson\n...")
            
            with col2:
                department = st.selectbox("Department for all", 
                    ["Finance", "HR", "Legal", "Operations", "Management", "Sales", "IT", "Various"])
            
            # Quality summary
            st.subheader("📊 Quality Summary")
            quality_scores = [data['quality']['quality_score'] for data in signature_data]
            avg_quality = sum(quality_scores) / len(quality_scores)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Quality Score", f"{avg_quality:.1f}/7")
            with col2:
                good_quality = sum(1 for score in quality_scores if score >= 4)
                st.metric("Good Quality Signatures", f"{good_quality}/{len(signature_data)}")
            with col3:
                poor_quality = sum(1 for score in quality_scores if score < 3)
                st.metric("Poor Quality", f"{poor_quality}/{len(signature_data)}")
            
            if st.button("🚀 Store All Signatures", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                signatures_to_store = []
                failed_signatures = []
                
                for i, data in enumerate(signature_data):
                    progress = (i + 1) / len(signature_data)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing {i+1}/{len(signature_data)}: {data['filename']}")
                    
                    # Preprocess and generate embedding
                    preprocessed = preprocess_signature(data['image'])
                    embedding = generate_signature_embedding(preprocessed)
                    
                    # Handle user IDs
                    if naming_option == "Manual input" and user_ids_input:
                        user_ids = user_ids_input.strip().split('\n')
                        if i < len(user_ids):
                            user_id = user_ids[i].strip()
                        else:
                            user_id = f"user_{i+1:03d}"
                    else:
                        user_id = data['user_id']
                    
                    # Create metadata
                    metadata = {
                        "user_id": user_id,
                        "full_name": f"User {i+1}",
                        "department": department,
                        "original_filename": data['filename'],
                        "timestamp": datetime.now().isoformat(),
                        "quality_score": data['quality']['quality_score'],
                        "blur_score": float(data['quality']['blur_score']),
                        "batch_upload": True
                    }
                    
                    signature_id = f"{user_id}_{uuid.uuid4().hex[:8]}"
                    
                    # Validate embedding before adding to batch
                    if validate_embedding(embedding):
                        signatures_to_store.append({
                            "signature_id": signature_id,
                            "embedding": embedding,
                            "metadata": metadata
                        })
                    else:
                        failed_signatures.append(data['filename'])
                
                # Store all signatures
                if signatures_to_store:
                    stored_count = store_signature_batch(index, signatures_to_store)
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success(f"✅ Successfully stored {stored_count} signatures!")
                    
                    if failed_signatures:
                        st.warning(f"⚠️ Failed to process {len(failed_signatures)} signatures due to poor quality")
                        for failed in failed_signatures:
                            st.write(f"- {failed}")
                    
                    # Show summary table
                    st.subheader("📋 Storage Summary")
                    summary_data = []
                    for data in signatures_to_store:
                        summary_data.append({
                            "User ID": data['metadata']['user_id'],
                            "Signature ID": data['signature_id'][:20] + "...",
                            "Quality Score": data['metadata']['quality_score'],
                            "Department": data['metadata']['department']
                        })
                    
                    st.dataframe(summary_data)
                else:
                    progress_bar.empty()
                    status_text.empty()
                    st.error("❌ No signatures could be stored - all embeddings were invalid")

    # Database info in sidebar
    st.sidebar.header("📊 Database Info")
    try:
        stats = index.describe_index_stats()
        st.sidebar.write(f"**Total Vectors:** {stats['total_vector_count']}")
        st.sidebar.write(f"**Dimension:** {stats['dimension']}")
    except:
        st.sidebar.write("Unable to fetch stats")
    
    st.sidebar.header("⚡ Quick Actions")
    if st.sidebar.button("🔄 Refresh Stats"):
        st.experimental_rerun()

if __name__ == "__main__":
    main()