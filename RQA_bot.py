import streamlit as st  # Importing Streamlit for web app development
from streamlit_chat import message  # Importing message function from streamlit_chat for chat UI
import os  # Importing os module for interacting with the operating system
import random  # Importing random for generating random values
import string  # Importing string for string operations
from dotenv import load_dotenv  # Importing load_dotenv from dotenv to load environment variables
from langchain.document_loaders import PyPDFLoader  # Importing PyPDFLoader for reading PDF files
from langchain.document_loaders import Docx2txtLoader  # Importing Docx2txtLoader for reading DOCX files
from langchain.chat_models import ChatOpenAI  # Importing ChatOpenAI for OpenAI chat model integration
from langchain.text_splitter import CharacterTextSplitter  # Importing CharacterTextSplitter for text splitting
from langchain.embeddings import HuggingFaceEmbeddings  # Importing HuggingFaceEmbeddings for embeddings
from langchain.vectorstores import Qdrant  # Importing Qdrant for vector storage
from langchain.chains import RetrievalQA  # Importing RetrievalQA for question-answering
from langchain.docstore.document import Document  # Importing Document class

# Load environment variables and set up API keys
openai_key = st.secrets['OPENAI_API_KEY']
url = st.secrets['QDRANT_URL']
qdrant_api_key = st.secrets['QDRANT_API_KEY']
pinecone_url = st.secrets['PINECONE_URL']
pinecone_api_key = st.secrets['PINECONE_API_KEY']
embeddings = HuggingFaceEmbeddings(model_name='intfloat/e5-large-v2')
embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')

# Initialize session state variables
if 'conversation' not in st.session_state:
    st.session_state.conversation = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processcomplete' not in st.session_state:
    st.session_state.processcomplete = None

def get_pdf_text(pdf):
    # Create an instance of PyPDFLoader with the provided PDF file
    pdf_reader = PyPDFLoader(pdf)

    # Initialize an empty string to hold the extracted text
    text = ''

    # Iterate through each page in the PDF
    for page in pdf_reader.pages:
        # Extract text from the current page and add it to the text variable
        text += page.extract_text()

    # Return the concatenated text from all pages
    return text

def get_docx_text(file):
    # Create an instance of Docx2txtLoader with the provided DOCX file
    doc = Docx2txtLoader(file)

    # Initialize an empty list to hold the text from each paragraph
    allText = []

    # Iterate through each paragraph in the DOCX document
    for docpara in doc.paragraphs:
        # Append the text of the current paragraph to the allText list
        allText.append(docpara.text)

    # Join all elements of the allText list into a single string, separated by spaces
    text = ' '.join(allText)

    # Return the concatenated text from all paragraphs
    return text

def get_file_text(file_upload):
    # Initialize an empty string to hold the extracted text
    text = ''

    # Split the file name to extract the file extension
    spliter = os.path.splitext(file_upload.name)
    file_extract = spliter[1]

    # Check if the file is a PDF
    if file_extract == 'pdf':
        # If it's a PDF, extract the text using the get_pdf_text function
        text += get_pdf_text(file_upload)

    # Check if the file is a DOCX
    elif file_extract == ".docx":
        # If it's a DOCX, extract the text using the get_docx_text function
        text += get_docx_text(file_upload)

    # If the file is neither PDF nor DOCX, do nothing
    else:
        pass

    # Return the extracted text
    return text

def get_chunk_file(file_name, file_text):
    # Initialize a CharacterTextSplitter with specified parameters for splitting text
    chunks = CharacterTextSplitter(
        separator='\n',          # Use newline character as the separator
        chunk_size=250,          # Each chunk will be roughly 250 characters long
        chunk_overlap=50,        # Adjacent chunks will overlap by 50 characters
        length_function=len,     # Function used to calculate the length of text
    )

    # Split the provided text into chunks based on the defined splitter settings
    chunk = chunks.split_text(file_text)

    # Initialize an empty list to store the chunked documents
    chnnk_list = []

    # Iterate over each chunk of text
    for cha in chunk:
        # Create metadata for each chunk with the file name as a source reference
        metadata = {'source': file_name}

        # Create a Document object with the chunk text and its metadata
        doc_string = Document(page_content=cha, metadata=metadata)

        # Add the Document object to the list
        chnnk_list.append(doc_string)

    # Return the list of Document objects
    return chnnk_list

def get_vectorestore(COLLECTION, chunk_list):
    # Try-catch block to handle potential exceptions
    try:
        # Initialize a Qdrant vector store from a list of documents (chunks)
        base = Qdrant.from_documents(
            documents=chunk_list,       # The list of document chunks to be stored
            embedding=embeddings,       # The embedding model to use for document representation
            api_key=qdrant_api_key,     # API key for Qdrant service
            url=url,                    # URL of the Qdrant service
            prefer_grpc=True,           # Prefer gRPC protocol for communication
            collection_name=COLLECTION, # The name of the collection in the Qdrant service
        )

    # Catch any exceptions that occur during the process
    except Exception as e:
        # Write the error message to the Streamlit interface
        st.write(f'Error: {e}')

    # Return the initialized Qdrant vector store
    return base

def qa_chain(num_chunk, vectorestore):
    # Initialize a language model using ChatOpenAI with a specified GPT model
    llm = ChatOpenAI(model='gpt-3.5-turbo')

    # Create a RetrievalQA object for question-answering using specific configurations
    qa = RetrievalQA.from_chain_type(
        llm=llm,  # The language model (in this case, GPT-3.5-turbo) to be used for generating responses
        chain_type='stuff',  # The chain type, here indicated as 'stuff' (needs context for specific meaning)
        retriever=vectorestore.as_retriever(
            search_type='similarity',  # The type of search to perform in the vector store
            search_k={'k': num_chunk}  # The number of similar chunks to retrieve for answering
        ),
        return_source_documents=True  # Flag to indicate whether to return source documents along with answers
    )

    # Return the initialized RetrievalQA object
    return qa

def main():
    # Load environment variables from a .env file
    load_dotenv()

    # Set Streamlit page configuration with a title
    st.set_page_config(page_title='Asif')
    st.title('GPT_ASIF')  # Set the title of the Streamlit app

    # Create a sidebar for file uploads and process initiation
    with st.sidebar:
        # File uploader widget in the sidebar for PDF and DOCX files
        upload_files = st.file_uploader('please upload_files', type=['pdf', 'docx'], accept_multiple_files=True)
        # Button in the sidebar to start the processing
        process = st.button('process')

    # Check if the process button has been clicked
    if process:
        # Check if the OpenAI API key is available
        if not openai_key:
            st.info('please enter the api')  # Show a message if the API key is missing
            st.stop()  # Stop further execution

        # Initialize an empty list to store document chunks
        chunk_list = []

        # Loop through each uploaded file
        for file_upload in upload_files:
            file_name = file_upload.name  # Get the name of the file
            file_text = get_file_text(file_upload)  # Extract text from the file
            chunk_file = get_chunk_file(file_name, file_text)  # Split the text into chunks
            chunk_list.extend(chunk_file)  # Add the chunks to the chunk list

            # Generate a random collection name for the vector store
            COLLECTION = ''.join(random.choices(string.ascii_letters, k=4))
            vectorestore = get_vectorestore(COLLECTION, chunk_list)  # Create a vector store with the chunks
            st.write('vectorestore is created')  # Notify that the vector store is created

            num_chunk = 4  # Set the number of chunks to be used in the QA chain
            # Initialize the QA chain with the vector store and number of chunks
            st.session_state.conversation = qa_chain(num_chunk, vectorestore)
            st.session_state.processcomplete = True  # Indicate that the process is complete

        # Check if the processing is complete
        if st.session_state.processcomplete == True:
            # Input field for the user to enter a question
            user_question = st.text_input('please enter your question')
            # If a question is entered, process it
            if user_question:
                inputchat(user_question)  # Call the function to handle the chat input

def inputchat(user_question):
    # Display a spinner in the UI while generating a response
    with st.spinner('generating response'):
        # Call the conversation function with the user's question
        result = st.session_state.conversation({'question': user_question})
        response = result['result']  # Extract the response from the result
        # Extract the source of the response. Note: There's a typo in the original code.
        # It should be 'metadata' instead of 'medata', and the list should be result['source_documents'].
        source = result['source_documents'][0]['metadata']['source']

    # Append the user's question to the chat history
    st.session_state.chat_history.append(user_question)
    # Append the response and its source to the chat history
    st.session_state.chat_history.append(f'{response}\n source documents {source}')

    # Create a container in Streamlit for displaying responses
    response_container = st.container()
    with response_container:
        # Iterate over the chat history and display each message
        for i, messages in enumerate(st.session_state.chat_history):
            # Check if the message is from the user or the response
            if i % 2 == 0:
                # Display the user's message
                message(messages, key=str(i), is_user=True)
            else:
                # Display the response message
                message(messages, key=str(i))

# This line checks if the script is being run as the main program
if __name__ == '__main__':
    # If it is, the main() function is called
    main()
