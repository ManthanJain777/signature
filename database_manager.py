"""
Database Manager - Delete and recreate Pinecone database
"""

import streamlit as st
from pinecone import Pinecone, ServerlessSpec

def delete_and_recreate_database():
    """Delete existing database and create a new one"""
    try:
        # Get API key from secrets
        pinecone_api_key = st.secrets["PINECONE_API_KEY"]
        
        # Initialize Pinecone
        pc = Pinecone(api_key=pinecone_api_key)
        
        index_name = "signature-verification"
        
        # Check if index exists and delete it
        existing_indexes = [index.name for index in pc.list_indexes()]
        
        if index_name in existing_indexes:
            # Delete the existing index
            pc.delete_index(index_name)
            st.success(f"✅ Deleted existing index: {index_name}")
        else:
            st.info(f"ℹ️ No existing index found: {index_name}")
        
        # Create new index
        pc.create_index(
            name=index_name,
            dimension=512,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        st.success(f"✅ Created new index: {index_name}")
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return False

def main():
    st.set_page_config(page_title="Database Manager", layout="centered")
    st.title("🗃️ Database Manager")
    st.write("Delete and recreate your signature database")
    
    st.warning("""
    ⚠️ **Warning: This will delete ALL signatures from your database!**
    
    - All stored signatures will be permanently deleted
    - You will need to re-add all signatures
    - This action cannot be undone
    """)
    
    if st.button("🚨 Delete & Recreate Database", type="secondary"):
        with st.spinner("Deleting and recreating database..."):
            success = delete_and_recreate_database()
            
            if success:
                st.balloons()
                st.success("🎉 Database successfully reset! You can now add new signatures.")
            else:
                st.error("Failed to reset database")

if __name__ == "__main__":
    main()