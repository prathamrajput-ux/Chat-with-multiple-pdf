import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Chat with Multiple PDFs 📚",
    page_icon="📚",
    layout="wide"
)

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF documents"""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            st.error(f"Error reading PDF {pdf.name}: {str(e)}")
    return text

def get_text_chunks(text):
    """Split text into smaller chunks for processing"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

@st.cache_resource
def get_embeddings_model():
    """Load and cache the embeddings model"""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'}
    )
    return embeddings

def get_vectorstore(text_chunks):
    """Create vector store from text chunks"""
    embeddings = get_embeddings_model()
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process chunks in batches to show progress
    batch_size = 10
    
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]
        
        # Update progress
        progress = min((i + batch_size) / len(text_chunks), 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Processing chunks: {min(i + batch_size, len(text_chunks))}/{len(text_chunks)}")
    
    # Create vector store
    status_text.text("Creating vector database...")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    
    progress_bar.empty()
    status_text.empty()
    
    return vectorstore

@st.cache_resource
def get_llm():
    """Load and cache the language model"""
    model_name = "google/flan-t5-base"
    
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Create pipeline
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            temperature=0.7,
            do_sample=True
        )
        
        llm = HuggingFacePipeline(pipeline=pipe)
        return llm
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def get_conversation_chain(vectorstore):
    """Create conversation chain with memory"""
    llm = get_llm()
    
    if llm is None:
        return None
    
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True
    )
    
    return conversation_chain

def handle_user_input(user_question):
    """Handle user input and generate response"""
    if st.session_state.conversation is None:
        st.warning("Please upload and process PDFs first!")
        return
    
    # Add user message to chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    st.session_state.messages.append({"role": "user", "content": user_question})
    
    with st.spinner("Thinking..."):
        try:
            response = st.session_state.conversation({'question': user_question})
            answer = response['answer']
            
            # Add assistant response
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            # Store source documents
            if 'source_documents' in response:
                st.session_state.last_sources = response['source_documents']
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": "Sorry, I encountered an error processing your question."})

def display_chat_history():
    """Display chat messages"""
    if "messages" in st.session_state:
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # Show sources for the last assistant message only
                if (message["role"] == "assistant" and 
                    i == len(st.session_state.messages) - 1 and 
                    hasattr(st.session_state, 'last_sources')):
                    with st.expander("📑 Source Documents", expanded=False):
                        for j, doc in enumerate(st.session_state.last_sources[:2]):
                            st.write(f"**Source {j+1}:**")
                            st.write(doc.page_content[:300] + "...")
                            st.divider()

def main():
    """Main function"""
    st.title("Chat with Multiple PDFs 📚")
    st.markdown("""
    Upload your PDF documents and ask questions about their content. 
    This app uses **completely FREE** Hugging Face models for processing!
    """)
    
    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar for PDF upload
    with st.sidebar:
        st.subheader("📄 Upload Documents")
        
        pdf_docs = st.file_uploader(
            "Choose PDF files",
            accept_multiple_files=True,
            type="pdf",
            help="Upload one or more PDF files to chat with"
        )
        
        if st.button("🔄 Process Documents", type="primary"):
            if pdf_docs:
                with st.spinner("Processing documents..."):
                    try:
                        # Extract text
                        st.info("📖 Extracting text from PDFs...")
                        raw_text = get_pdf_text(pdf_docs)
                        
                        if not raw_text.strip():
                            st.error("❌ No text found in the uploaded PDFs!")
                            return
                        
                        # Create chunks
                        st.info("✂️ Creating text chunks...")
                        text_chunks = get_text_chunks(raw_text)
                        st.success(f"✅ Created {len(text_chunks)} text chunks")
                        
                        # Create vector store
                        st.info("🔍 Creating vector database...")
                        vectorstore = get_vectorstore(text_chunks)
                        
                        # Create conversation chain
                        st.info("🤖 Setting up conversation...")
                        conversation_chain = get_conversation_chain(vectorstore)
                        
                        if conversation_chain is not None:
                            st.session_state.conversation = conversation_chain
                            # Clear previous messages
                            st.session_state.messages = []
                            st.success("🎉 Documents processed successfully!")
                        else:
                            st.error("❌ Failed to setup conversation chain!")
                            
                    except Exception as e:
                        st.error(f"❌ Error processing documents: {str(e)}")
            else:
                st.warning("⚠️ Please upload at least one PDF file!")
        
        # Model information
        st.divider()
        st.subheader("🤖 Model Information")
        st.info("""
        **📊 Embeddings:** sentence-transformers/all-MiniLM-L6-v2  
        **🧠 LLM:** google/flan-t5-base  
        **💾 Vector DB:** FAISS  
        **💰 Cost:** Completely Free! 🎉
        """)
        
        # Instructions
        st.subheader("📋 How to Use")
        st.markdown("""
        1. 📁 Upload one or more PDF files
        2. 🔄 Click "Process Documents"  
        3. ⏳ Wait for processing to complete
        4. 💬 Ask questions in the chat box
        5. 🎯 Get AI-powered answers!
        """)
        
        # Usage stats
        if st.session_state.conversation:
            st.divider()
            st.metric("💬 Messages", len(st.session_state.messages))
    
    # Main chat interface
    if st.session_state.conversation is not None:
        st.subheader("💬 Chat with your documents")
        
        # Display chat history
        display_chat_history()
        
        # Chat input
        user_question = st.chat_input("Ask a question about your documents...")
        
        if user_question:
            handle_user_input(user_question)
            st.rerun()
    else:
        # Welcome message
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info("👈 Please upload PDF documents from the sidebar to get started!")
            
            st.subheader("🎯 Example Questions You Can Ask:")
            st.markdown("""
            - **"What is the main topic of these documents?"**
            - **"Summarize the key points from the PDFs"**
            - **"What are the conclusions mentioned?"**
            - **"Can you explain [specific topic] from the documents?"**
            - **"What recommendations are provided?"**
            - **"List the important findings"**
            """)
            
        with col2:
            st.subheader("⚡ Features")
            st.markdown("""
            ✅ **Multiple PDFs**  
            ✅ **100% Free**  
            ✅ **Local Processing**  
            ✅ **Source References**  
            ✅ **Chat History**  
            ✅ **No API Keys**
            """)

if __name__ == "__main__":
    main()
