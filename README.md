README.md for GPT_ASIF Chatbot
This README provides instructions on how to run the GPT_ASIF chatbot locally. The chatbot is designed to process documents (PDF and DOCX) and answer questions based on the content of these documents.
Prerequisites
Before running the chatbot, ensure you have the following installed:
Python 3.8 or higher
pip (Python package installer)
Installation
Clone the Repository:
Clone this repository to your local machine using Git.
Install Required Packages:
Navigate to the cloned directory and install the required packages using pip:
bashCopy code
pip install -r requirements.txt 
This will install all necessary libraries, including Streamlit, LangChain, and other dependencies.
Environment Variables:
Set up the following environment variables in a .env file in the root directory of the project:
OPENAI_API_KEY: Your OpenAI API key.
QDRANT_URL: The URL for Qdrant service.
QDRANT_API_KEY: Your Qdrant API key.
PINECONE_URL: The URL for Pinecone service.
PINECONE_API_KEY: Your Pinecone API key.
Example .env file:
makefileCopy code
OPENAI_API_KEY=your_openai_api_key QDRANT_URL=your_qdrant_url QDRANT_API_KEY=your_qdrant_api_key PINECONE_URL=your_pinecone_url PINECONE_API_KEY=your_pinecone_api_key 
Running the Application:
Start the application by running the following command in the terminal:
bashCopy code
streamlit run your_script_name.py 
Replace your_script_name.py with the name of the Python script for the chatbot.
Usage
Upload Documents:
Use the sidebar to upload PDF or DOCX documents. You can upload multiple files at once.
Process Documents:
Click the 'Process' button to process the uploaded documents. This will extract text and prepare the data for the chatbot.
Ask Questions:
Once the processing is complete, you can ask questions related to the content of the uploaded documents.
View Responses:
The chatbot will respond to your questions, sourcing information directly from the processed documents.
Note
Ensure you have an active internet connection as the chatbot relies on external APIs.
The accuracy of responses depends on the quality and content of the uploaded documents.
