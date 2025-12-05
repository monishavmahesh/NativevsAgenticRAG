import streamlit as st
import os
import shutil
from langchain_ollama import OllamaLLM
from agents.base_agent import BaseAgent
from agents.meta_agent import MetaAgent
from core.memory import load_persistent_memory
from core.loader import load_documents
from core.vectorstore import build_vector_store, load_vector_store
from config import MODEL_NAME

st.set_page_config(page_title="Agentic RAG School Assistant", layout="wide")
st.title("🎓 Agentic RAG School Assistant")

# Use Ollama LLM (low temperature for factual accuracy)
qa_pipeline = OllamaLLM(
    model=MODEL_NAME,
    temperature=0.1
)

# Initialize session state
if 'agents' not in st.session_state:
    st.session_state.agents = {}
if 'textbooks_uploaded' not in st.session_state:
    st.session_state.textbooks_uploaded = {"Math": False, "English": False, "EVS": False}
if 'uploaded_file_hashes' not in st.session_state:
    st.session_state.uploaded_file_hashes = {"Math": None, "English": None, "EVS": None}

# Sidebar for PDF uploads
st.sidebar.header("📚 Upload Subject PDFs")
st.sidebar.write("Upload PDFs for each subject to enable the assistant")

subjects = ["Math", "English", "EVS"]

for subject in subjects:
    uploaded_file = st.sidebar.file_uploader(
        f"Upload {subject} PDF", 
        type=['pdf'], 
        key=f"upload_{subject}"
    )
    
    if uploaded_file is not None:
        # Get file hash to detect new uploads
        file_hash = hash(uploaded_file.getvalue())
        
        # Check if this is a new file (different from previous upload)
        is_new_file = st.session_state.uploaded_file_hashes[subject] != file_hash
        
        if is_new_file:
            st.sidebar.info(f"🔄 Processing new {subject} PDF...")
            
            # Save uploaded file
            os.makedirs("./textbooks", exist_ok=True)
            file_path = f"./textbooks/{subject.lower()}.pdf"
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Delete old DB if exists
            db_path = f"./db/{subject}"
            if os.path.exists(db_path):
                shutil.rmtree(db_path)
                st.sidebar.info(f"🗑️ Removed old {subject} database")
            
            # Remove old agent from session
            if subject in st.session_state.agents:
                del st.session_state.agents[subject]
            
            with st.spinner(f"Creating new embeddings for {subject}..."):
                try:
                    # Load documents
                    docs = load_documents(file_path)
                    
                    if not docs:
                        st.sidebar.error(f"❌ No content found in {subject} PDF")
                        continue
                    
                    # Build vector store (creates new DB)
                    vectordb = build_vector_store(subject, docs)
                    
                    # Load memory
                    subject_memory = load_persistent_memory(subject)
                    
                    # Create new agent
                    agent = BaseAgent(subject, file_path, qa_pipeline, subject_memory)
                    st.session_state.agents[subject] = agent
                    st.session_state.textbooks_uploaded[subject] = True
                    st.session_state.uploaded_file_hashes[subject] = file_hash
                    
                    st.sidebar.success(f"✅ {subject} textbook loaded with new embeddings!")
                    
                except Exception as e:
                    st.sidebar.error(f"❌ Error loading {subject}: {str(e)}")
                    st.session_state.textbooks_uploaded[subject] = False
        else:
            # Same file already loaded
            if subject in st.session_state.agents:
                st.sidebar.success(f"✅ {subject} textbook already loaded")
    else:
        # No file uploaded - try to load existing DB if available
        if subject not in st.session_state.agents:
            db_path = f"./db/{subject}"
            file_path = f"./textbooks/{subject.lower()}.pdf"
            
            if os.path.exists(db_path) and os.path.exists(file_path):
                try:
                    with st.spinner(f"Loading existing {subject} database..."):
                        # Load existing vector store
                        vectordb = load_vector_store(subject)
                        
                        if vectordb:
                            # Load memory
                            subject_memory = load_persistent_memory(subject)
                            
                            # Create agent with existing DB
                            agent = BaseAgent(subject, file_path, qa_pipeline, subject_memory)
                            st.session_state.agents[subject] = agent
                            st.session_state.textbooks_uploaded[subject] = True
                            
                            st.sidebar.info(f"ℹ️ Loaded existing {subject} database")
                except Exception as e:
                    st.sidebar.warning(f"⚠️ Could not load existing {subject} DB: {str(e)}")

# Check if at least one subject is uploaded
if not any(st.session_state.textbooks_uploaded.values()):
    st.warning("⚠️ Please upload at least one subject PDF from the sidebar to get started")
    st.stop()

# Create meta agent if we have agents
if st.session_state.agents:
    meta_agent = MetaAgent(st.session_state.agents)

# Streamlit UI Tabs
available_subjects = [s for s in subjects if st.session_state.textbooks_uploaded[s]]
tabs_list = [f"{'🧮' if s=='Math' else '📘' if s=='English' else '🌱'} {s}" for s in available_subjects]
tabs_list.append("🤖 Auto (Meta-Agent)")

tabs = st.tabs(tabs_list)

# Individual subject tabs
for idx, subject in enumerate(available_subjects):
    with tabs[idx]:
        q = st.text_input(f"Ask a {subject} question:", key=f"input_{subject}")
        if st.button(f"Ask {subject}", key=f"btn_{subject}"):
            if q.strip():
                with st.spinner(f"Thinking about {subject}..."):
                    answer = st.session_state.agents[subject].query(q)
                    st.write("**Answer:**", answer)
            else:
                st.warning("Please enter a question")

# Meta-agent tab (last tab)
with tabs[-1]:
    q = st.text_input("Ask any question:", key="meta")
    if st.button("Ask Auto-Agent", key="btn_meta"):
        if q.strip():
            with st.spinner("Auto-routing your question..."):
                answer = meta_agent.route(q)
                st.write("**Answer:**", answer)
        else:
            st.warning("Please enter a question")

# Show upload status
st.sidebar.markdown("---")
st.sidebar.header("📊 Subject Status")
for subject in subjects:
    if st.session_state.textbooks_uploaded[subject]:
        st.sidebar.write(f"**{subject}:** ✅ Active")
        # Add button to clear subject
        if st.sidebar.button(f"Clear {subject}", key=f"clear_{subject}"):
            # Remove agent
            if subject in st.session_state.agents:
                del st.session_state.agents[subject]
            
            # Remove from state
            st.session_state.textbooks_uploaded[subject] = False
            st.session_state.uploaded_file_hashes[subject] = None
            
            # Delete DB
            db_path = f"./db/{subject}"
            if os.path.exists(db_path):
                shutil.rmtree(db_path)
            
            # Delete PDF
            file_path = f"./textbooks/{subject.lower()}.pdf"
            if os.path.exists(file_path):
                os.remove(file_path)
            
            st.sidebar.success(f"Cleared {subject}!")
            st.rerun()
    else:
        st.sidebar.write(f"**{subject}:** ❌ Not uploaded")
