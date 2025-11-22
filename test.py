try:
    import pinecone
    print("✅ Pinecone imported successfully")
    
    import cv2
    print("✅ OpenCV imported successfully")
    
    import streamlit
    print("✅ Streamlit imported successfully")
    
    import numpy as np
    print("✅ NumPy imported successfully")
    
    print("\n🎉 All imports successful! You're ready to go.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")