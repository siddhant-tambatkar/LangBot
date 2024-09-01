# import streamlit as st
# import os
# from langchain_groq import ChatGroq
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_google_genai import GoogleGenerativeAIEmbeddings

# # Load the GROQ and Google API keys directly in the file
# os.environ["GOOGLE_API_KEY"] = google_api_key

# st.title("Gemma Model Document Q&A")

# llm = ChatGroq(groq_api_key=groq_api_key,
#                model_name="Llama3-8b-8192")

# prompt = ChatPromptTemplate.from_template(
# """
# Answer the questions based on the provided context only.
# Please provide the most accurate response based on the question.
# <context>
# {context}
# <context>
# Questions: {input}
# """
# )

# def vector_embedding():
#     if "vectors" not in st.session_state:
#         st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#         st.session_state.loader = PyPDFDirectoryLoader("D:\\VIT academics\\Sem 6\\EDI\\pdfs")  # Data Ingestion
#         st.session_state.docs = st.session_state.loader.load()  # Document Loading
#         st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
#         st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])  # Splitting
#         st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector OpenAI embeddings

# prompt1 = st.text_input("Enter Your Question From Documents")

# if st.button("Documents Embedding"):
#     vector_embedding()
#     st.write("Vector Store DB Is Ready")

# import time

# if prompt1:
#     if "vectors" not in st.session_state:
#         vector_embedding()
#     document_chain = create_stuff_documents_chain(llm, prompt)
#     retriever = st.session_state.vectors.as_retriever()
#     retrieval_chain = create_retrieval_chain(retriever, document_chain)
#     start = time.process_time()
#     response = retrieval_chain.invoke({'input': prompt1})
#     st.write("Response time:", time.process_time() - start)
#     st.write(response['answer'])

#     # With a Streamlit expander
#     with st.expander("Document Similarity Search"):
#         # Find the relevant chunks
#         for i, doc in enumerate(response["context"]):
#             st.write(doc.page_content)
#             st.write("--------------------------------")

# # Load the GROQ and Google API keys directly in the file

import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import speech_recognition as sr
from gtts import gTTS
import pyttsx3
import time

# Custom CSS for centering and coloring the title
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

    body {
        font-family: 'Roboto', sans-serif;
        background-color: white; /* Background color */
    }

    body .container {
    font-size: 40px;
    text-align: center;
    color: grey;
    max-width: 800px;
    margin: auto;
    font-weight: 300;
    margin-bottom: 40px;
}

    .container1 {

        text-align: center;
        max-width: 500px;
        margin: auto;
        padding: 5px;
    }

    .header {
        text-align: center;
        margin-bottom: 30px;
    }

    .footer {
        text-align: center;
        margin-top: 30px;
        color: #777;
    }

    .title {
        color: #4CAF50;
        font-size: 3em;
        font-weight: 700;
        margin-bottom: 20px;
    }

    .button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 1.1em;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }

    .button:hover {
        background-color: #45a049;
    }

    .toggle-label {
        font-size: 1.2em;
        margin-right: 10px;
    }

    .toggle-switch input[type="checkbox"] {
        height: 0;
        width: 0;
        visibility: hidden;
    }

    .toggle-switch input[type="checkbox"] + label {
        display: inline-block;
        width: 60px;
        height: 30px;
        background-color: #ccc;
        border-radius: 15px;
        transition: background-color 0.3s ease;
        position: relative;
        cursor: pointer;
    }

    .toggle-switch input[type="checkbox"] + label:before {
        content: '';
        position: absolute;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        background-color: white;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.25);
        transition: transform 0.3s ease;
        top: 0;
        left: 0;
    }

    .toggle-switch input[type="checkbox"]:checked + label {
        background-color: #4CAF50;
    }

    .toggle-switch input[type="checkbox"]:checked + label:before {
        transform: translateX(30px);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown('<div class="header"><h1 class="title">Intelli-Doc Navigator - For Visually Impaired</h1></div>', unsafe_allow_html=True)

# Introduction Section
st.markdown(
    """
    <div class="container">
    <p>Welcome to Intelli-Doc Navigator! This app helps visually impaired users navigate documents using text or voice input.</p>
    </div>
    <div1 class="container1">
    <p1>Please choose your preferred input method below:</p1>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar for inputting keys and PDF directory
st.sidebar.title("Configuration")
groq_api_key = st.sidebar.text_input("Enter GROQ API Key", type="password")
google_api_key = st.sidebar.text_input("Enter Google API Key", type="password")
pdf_directory = st.sidebar.text_input("Enter PDF Directory Path", value="D:\\VIT academics\\Sem 6\\EDI\\pdfs")

if st.sidebar.button("Set Keys and Directory"):
    if groq_api_key and google_api_key and pdf_directory:
        os.environ["GROQ_API_KEY"] = groq_api_key
        os.environ["GOOGLE_API_KEY"] = google_api_key
        st.session_state.groq_api_key = groq_api_key
        st.session_state.google_api_key = google_api_key
        st.session_state.pdf_directory = pdf_directory
        st.sidebar.success("Keys and directory set.")
    else:
        st.sidebar.error("Please fill in all fields.")

# Ensure keys are set before proceeding
if "groq_api_key" in st.session_state and "google_api_key" in st.session_state:
    try:
        # Load the GROQ model
        llm = ChatGroq(groq_api_key=st.session_state.groq_api_key, model_name="Llama3-8b-8192")

        # Define the chat prompt template
        prompt = ChatPromptTemplate.from_template(
            """
            Answer the questions based on the provided context only.
            Please provide the most accurate response based on the question.
            <context>
            {context}
            <context>
            Questions: {input}
            """
        )

        # Function to handle document embedding
        def vector_embedding():
            if "vectors" not in st.session_state:
                st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                st.session_state.loader = PyPDFDirectoryLoader(st.session_state.pdf_directory)  # Data Ingestion
                st.session_state.docs = st.session_state.loader.load()  # Document Loading
                st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
                st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])  # Splitting
                st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector OpenAI embeddings

        # Function to recognize speech
        def recognize_speech():
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                st.write("Speak Now...")
                audio = recognizer.listen(source)

            try:
                text = recognizer.recognize_google(audio)
                st.write("You said:", text)
                return text
            except sr.UnknownValueError:
                st.write("Sorry, could not understand the audio")
                return None
            except sr.RequestError as e:
                st.write(f"Could not request results from Google Speech Recognition service; {e}")
                return None

        # Function to convert text to speech
        def text_to_speech(text):
            try:
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                st.write("An error occurred:", e)

        # Function to handle Q&A
        def handle_qa(input_text):
            try:
                if "vectors" not in st.session_state:
                    vector_embedding()
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vectors.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                start = time.process_time()
                response = retrieval_chain.invoke({'input': input_text})
                st.write("Response time:", time.process_time() - start)
                st.write(response['answer'])
                text_to_speech(response['answer'])
            except Exception as e:
                st.error(f"An error occurred during Q&A processing: {e}")

        # Inline contents inside the app
        st.markdown('<div class="container">', unsafe_allow_html=True)

        # Interactive Elements: Toggle switch for input method
        input_option = st.checkbox("Voice Input", key="voice_input")

        # User input options
        if input_option:  # Voice input
            if st.button("Start Voice Recognition"):
                recognized_text = recognize_speech()
                if recognized_text:
                    handle_qa(recognized_text)
        else:  # Text input
            input_text = st.text_input("Enter Your Question From Documents", key="text_input", max_chars=200, value="", type="default", help="Type your question here")
            if st.button("Submit Question"):
                if input_text:
                    handle_qa(input_text)

        # Button for embedding documents
        if st.button("Embed Documents"):
            try:
                vector_embedding()
                st.write("Document embedding completed.")
            except Exception as e:
                st.error(f"An error occurred during document embedding: {e}")

        # Close container
        st.markdown('</div>', unsafe_allow_html=True)

        # Footer
        st.markdown('<div class="footer"><p>Intelli-Doc Navigator - Developed by Siddhant :)</p></div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred during model initialization: {e}")
else:
    st.warning("Please set the API keys and PDF directory in the sidebar.")
