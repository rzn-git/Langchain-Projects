import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Check for OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Set page configuration
st.set_page_config(
    page_title="PDF Question Answering System",
    page_icon="📚",
    layout="wide"
)

# Load the PDF file
@st.cache_resource
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

# Split the documents into chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

# Create vector store
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store, embeddings

# Save vector store to disk
def save_vector_store(vector_store, directory="vector_store"):
    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save the vector store
    vector_store.save_local(directory)

# Load vector store from disk
@st.cache_resource
def load_vector_store(directory="vector_store"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists(directory):
        vector_store = FAISS.load_local(directory, embeddings, allow_dangerous_deserialization=True)
        return vector_store
    else:
        return None

# Create QA chain - Note the underscore prefix in _vector_store to prevent hashing
@st.cache_resource
def create_qa_chain(_vector_store):
    # Initialize OpenAI model (using GPT-4 for more advanced capabilities)
    llm = ChatOpenAI(
        model="gpt-4-turbo",  # Using the more advanced GPT-4 Turbo model
        temperature=0.1,  # Lower temperature for more factual responses
        api_key=openai_api_key,  # Explicitly pass the API key
    )
    
    # Create a custom prompt template
    template = """
    You are an AI assistant that answers questions based on the provided context.
    
    Context: {context}
    
    Question: {question}
    
    Please provide a detailed and accurate answer based only on the information in the context.
    If the information is not in the context, say "I don't have enough information to answer this question."
    
    Answer:
    """
    
    PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    chain_type_kwargs = {"prompt": PROMPT}
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=_vector_store.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True
    )
    
    return qa_chain

def initialize_system():
    # Path to your PDF file
    pdf_path = "dataset/read.pdf"
    vector_store_dir = "vector_store"
    
    # Try to load existing vector store first
    vector_store = load_vector_store(vector_store_dir)
    
    # If no vector store exists, create a new one
    if vector_store is None:
        with st.spinner("Processing PDF for the first time... This may take a moment."):
            # Load and process the PDF
            documents = load_pdf(pdf_path)
            chunks = split_documents(documents)
            vector_store, _ = create_vector_store(chunks)
            save_vector_store(vector_store, vector_store_dir)
    
    # Create QA chain
    qa_chain = create_qa_chain(vector_store)
    
    return qa_chain

def main():
    st.title("📚 PDF Question Answering System")
    st.markdown("Ask questions about the content in the PDF document.")
    
    # Initialize the QA system
    qa_chain = initialize_system()
    
    # Create a text input for the user's question
    user_question = st.text_input("Ask a question about the PDF:", placeholder="e.g., How many movies has Salman Shah done?")
    
    # Add a submit button
    submit_button = st.button("Get Answer")
    
    # Display chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # When the user submits a question
    if submit_button and user_question:
        with st.spinner("Thinking..."):
            # Get the answer using the recommended invoke method
            result = qa_chain.invoke({"query": user_question})
            answer = result["result"]
            
            # Add to chat history
            st.session_state.chat_history.append({"question": user_question, "answer": answer})
    
    # Display chat history
    for i, exchange in enumerate(st.session_state.chat_history):
        st.markdown(f"### Question {i+1}:")
        st.markdown(f"**Q:** {exchange['question']}")
        st.markdown(f"**A:** {exchange['answer']}")
        st.divider()
    
    # Add a sidebar with information
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This application uses:
        - LangChain for document processing
        - OpenAI's GPT-4 Turbo model
        - FAISS for vector storage
        - Streamlit for the user interface
        
        You can ask questions about the content in the PDF document.
        """)
        
        # Add a button to clear chat history
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.experimental_rerun()

if __name__ == "__main__":
    main() 