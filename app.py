# pip install langchain openai streamlit tiktoken chromadb pypdf pycryptodome

# Import os to set API key

import os

# Import OpenAI as main LLM service
from langchain.llms import OpenAI

# Bring in streamlit for UI/app interface
import streamlit as st

# Import PDF document loaders
from langchain.document_loaders import PyPDFLoader

# Import chroma as the vector store
from langchain.vectorstores import Chroma

# Import vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo,
)
# Set APIkey for OpneAI Service
# Can sub this out for other LLM providers
os.environ['OPENAI_API_KEY'] = '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'

# Create instance of OpenAI LLM
llm = OpenAI(temprature=0.9)

# Create and load PDF Loader
loader = PyPDFLoader('annualreport.pdf')

# Split pages from pdf
pages = loader.load_and_split()

# Load document into vector database aka ChromaDB
store = Chroma.from_documents(pages, collection_name='annualreport')

# Create vectorstore info object - metadata repo?
vectorstore_info = VectorStoreInfo(
    name='annual_report',
    description='a banking annual report as a pdf',
    vectorstore=store
)

# Convert the document store into a langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

# Add the toolkit to and end to end LC
agent_executer = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
)
# Create a text input box for the user
prompt = st.text_input('Input your prompt here')

# If the user hits enter
if prompt:
    # Then pass the prompt to LLM
    # response = llm(prompt)

    # Swap out the raw llm for document agent
    response = agent_executer.run(prompt)
    
    # and write it out to the screen
    st.write(response)

    # With a streamlit expander
    with st.expander('Document Similarity Search'):
        # Fine the relevant pages
        search = store.similarity_search_with_score(prompt)

        # Write out the first 
        st.write(search[0][0].page_content)