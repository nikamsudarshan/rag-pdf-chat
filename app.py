import streamlit as st
import os
import asyncio
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

# --- Page Configuration ---
st.set_page_config(
    page_title="PDF Chat AI",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="auto",
)

# --- Load Environment Variables ---
# Create a .env file in your project directory and add your API key:
# GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
load_dotenv()

# --- Custom CSS for Aesthetics ---
st.markdown("""
<style>
    /* General body styling */
    body {
        font-family: 'Inter', sans-serif;
    }

    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
        border-radius: 1rem;
    }

    /* Title styling */
    h1 {
        font-weight: 700;
        color: #2E3B4E;
        text-align: center;
    }

    /* Subheader/instruction styling */
    .st-emotion-cache-16idsys p {
        text-align: center;
        color: #555;
    }

    /* File uploader custom styling */
    .stFileUploader > label {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2E3B4E;
    }

    .stFileUploader > div > div {
        border: 2px dashed #B0B0B0;
        background-color: #f9f9f9;
        border-radius: 0.5rem;
        padding: 1.5rem;
        text-align: center;
    }

    .stFileUploader > div > div > p {
        font-weight: 500;
        color: #555;
    }

    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 0.5rem;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        color: #FFFFFF;
        background-color: #4A90E2;
        border: none;
        transition: background-color 0.2s;
    }

    .stButton > button:hover {
        background-color: #357ABD;
    }

    .stButton > button:focus {
        outline: none;
        box-shadow: 0 0 0 2px #4A90E2, 0 0 0 4px rgba(74, 144, 226, 0.3);
    }

    /* Text input styling */
    .stTextInput > div > div > input {
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
        border: 1px solid #B0B0B0;
    }

    .stTextInput > div > div > input:focus {
        border-color: #4A90E2;
        box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.3);
    }

    /* Chat message styling */
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
    }

    .chat-message.user {
        background-color: #E9F5FF;
    }

    .chat-message.bot {
        background-color: #F0F2F6;
    }

    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin-right: 1rem;
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 1.5rem;
    }

    .chat-message.user .avatar {
        background-color: #4A90E2;
        color: white;
    }

    .chat-message.bot .avatar {
        background-color: #6C757D;
        color: white;
    }

    .chat-message .message {
        flex: 1;
        padding-top: 0.25rem;
    }

</style>
""", unsafe_allow_html=True)

# --- Backend Functions ---

def get_pdf_text(pdf_docs):
    """Extracts text from a list of uploaded PDF documents."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_text_chunks(text):
    """Splits text into manageable chunks for processing."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    """Creates and returns a FAISS vector store from text chunks."""
    if not text_chunks:
        return None
    try:
        # FIX: Set up the asyncio event loop for the current thread
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None


def get_conversation_chain(vectorstore):
    """Creates a conversational retrieval chain."""
    # UPDATED: Changed model to gemini-2.5-flash
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# --- Chat UI Templates ---
bot_template = """
<div class="chat-message bot">
    <div class="avatar">🤖</div>
    <div class="message">{{MSG}}</div>
</div>
"""

user_template = """
<div class="chat-message user">
    <div class="avatar">👤</div>
    <div class="message">{{MSG}}</div>
</div>
"""

# --- Main Application ---
def main():
    st.title("Chat with your PDF 📄")
    st.write("Upload PDF documents and ask questions. The AI will provide answers based on the content.")

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # --- Sidebar for PDF Upload ---
    with st.sidebar:
        st.header("Upload Your PDFs")
        pdf_docs = st.file_uploader(
            "Drag and drop your PDFs here or click to upload",
            type="pdf",
            accept_multiple_files=True
        )

        if st.button("Process Documents"):
            if pdf_docs:
                with st.spinner("Processing documents..."):
                    # 1. Get PDF text
                    raw_text = get_pdf_text(pdf_docs)

                    # 2. Get text chunks
                    text_chunks = get_text_chunks(raw_text)

                    if not text_chunks:
                         st.warning("Could not extract text from the PDF(s). Please try another document.")
                    else:
                        # 3. Create vector store
                        vectorstore = get_vectorstore(text_chunks)

                        if vectorstore:
                            # 4. Create conversation chain and store in session state
                            st.session_state.conversation = get_conversation_chain(vectorstore)
                            st.success("Documents processed! You can now ask questions.")
            else:
                st.warning("Please upload at least one PDF file.")

    # --- Main Chat Interface ---
    if st.session_state.conversation:
        # Display chat history
        st.write("### Chat History")
        if st.session_state.chat_history:
            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0: # User message
                    st.markdown(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
                else: # Bot message
                    st.markdown(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

        st.markdown("---")

        # User prompt section
        user_question = st.text_input("Ask a question about your documents:", key="prompt_input")

        if st.button("Submit Question", key="submit_button"):
            if user_question:
                response = st.session_state.conversation.invoke({'question': user_question})
                st.session_state.chat_history = response['chat_history']
                st.rerun()
            else:
                st.warning("Please enter a question.")
    else:
        st.info("Please upload and process your PDF document(s) in the sidebar to begin.")


if __name__ == "__main__":
    main()
