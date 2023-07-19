import os
import re
import tempfile
import pandas as pd
import streamlit as st
import time
import openai

from streamlit_option_menu import option_menu
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks import get_openai_callback
from queries import query_dict

def is_valid_openai_key(api_key):
    """
    Checks if the OpenAI API key is valid.
    """
    # Set up a temporary API client with the provided key
    openai.api_key = api_key
    
    # Try to fetch a list of available models
    # (or any other lightweight request to the API)
    try:
        openai.Model.list()
        return True
    except openai.error.AuthenticationError:
        # Handle specific authentication error
        return False
    except openai.error.OpenAIError:
        # Handle other types of OpenAI related errors
        return False
    except Exception as e:
        # Handle any other unexpected exceptions (optional, based on your needs)
        return False

def get_unique_values(data):
    """Returns a list of unique dictionaries based on their metadata."""
    
    # Set to track unique metadata values
    unique_values = set()
    
    # List to store unique dictionaries
    unique_dicts = []

    # Iterate through each item in data
    for item in data:
        # Convert dictionary metadata to a tuple to make it hashable
        item_tuple = tuple(item.metadata.items())
        
        # If tuple is already encountered, skip to the next iteration
        if item_tuple in unique_values:
            continue
        
        # Add the tuple to the set and the original dictionary to the list
        unique_values.add(item_tuple)
        unique_dicts.append(item)

    return unique_dicts

def clean_source_name(temp_filename):
    """Returns source file name without directory information or file extension."""
    
    # Extract the filename without directory information
    filename = temp_filename.split("/")[-1]
    
    # Remove the file extension
    base_name = filename.split(".")[0]

    return base_name

def format_sources(source_docs_list):
    output_dict = {}
    
    for source_docs in source_docs_list:
        if 'source' in source_docs.metadata and 'page' in source_docs.metadata:
            source = clean_source_name(source_docs.metadata['source'])
            page_num = str(source_docs.metadata['page'])
            
            if source in output_dict:
                output_dict[source].append(page_num)
            else:
                output_dict[source] = [page_num]

    output_list = []
    for source, pages in output_dict.items():
        pages.sort(key=lambda x: int(re.search(r'\d+', x).group()))
        page_str = ", ".join(pages)
        output_list.append("• " + source + " - Pg. " + page_str)

    output_string = "<br>".join(output_list)

    return output_string

def docs_to_dataframe(vectorstore, doc_names):
    llm = OpenAI(
        temperature=0, 
        openai_api_key=st.session_state.api_key, 
        model='text-davinci-003'
    )
    
    chain = load_qa_chain(llm, chain_type="stuff")

    results = {key: [] for key in query_dict}
    results['File Name'] = []

    # Loop through files in the directory
    for source_doc in doc_names:
        results['File Name'].append(clean_source_name(source_doc))
        
        for key, value in query_dict.items():
            chunks = vectorstore.similarity_search(value, filter={"source": source_doc})
            
            with get_openai_callback() as cb:
                results[key].append(chain.run(input_documents=chunks, question=key))
                st.session_state.callbacks += cb.total_cost

    dataframe = pd.DataFrame(results)
    return dataframe

def csv_excel_to_dictionary(uploaded_file):
    data_dict = {}
    
    if uploaded_file is None:
        return data_dict

    file_extension = uploaded_file.name.split(".")[-1]

    if file_extension not in ["csv", "xlsx"]:
        st.error("Invalid file format. Please upload a CSV or Excel file.")
        return data_dict

    df = pd.read_csv(uploaded_file) if file_extension == "csv" else pd.read_excel(uploaded_file)

    for column in df.columns:
        data_dict[column] = df[column].tolist()

    return data_dict

def get_pdf_text(pdf_docs):
    docs = []
    doc_names = []

    for file in pdf_docs:
        with tempfile.NamedTemporaryFile(prefix=file.name, delete=False) as temp_file:
            temp_file.write(file.read())
            file_path = temp_file.name
            doc_names.append(file_path)  # Replaced temp_file.name with the already defined file_path

        loader = PyPDFLoader(file_path)
        docs.append(loader.load())
        os.remove(file_path)

    return docs, doc_names

def get_text_chunks(docs):
    chunk_list = []
    metadata_list = []

    for doc in docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        
        chunks = text_splitter.split_documents(doc)
        
        for chunk in chunks:
            chunk_list.append(chunk)
            metadata_list.append({
                'source': chunk.metadata['source'], 
                'page': chunk.metadata['page']
            })

    return chunk_list, metadata_list

def get_vectorstore(text_chunks, metadata):
    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.api_key)
    
    vectorstore = FAISS.from_texts(
        [t.page_content for t in text_chunks], 
        embeddings, 
        metadatas=metadata
    )
    
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(openai_api_key=st.session_state.api_key)
    # streaming=True,
    # callbacks=[StreamingStdOutCallbackHandler()])

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True
    )

    return conversation_chain

def format_output(message, source):
    output = message + '<br><br>' + '<u>Sources:</u><br>' + '<i>' + source + '</i>'
    return output

def initialize_states():
    """Initializes session state variables if they don't exist."""
    
    default_values = {
        "messages": [],
        "conversation": None,
        "chat_history": None,
        "callbacks": 0,
        "api_key": None
    }
    
    for key, default in default_values.items():
        if key not in st.session_state:
            setattr(st.session_state, key, default)

def handle_chat_input(prompt, pdf_docs):
    if st.session_state.conversation is not None:
        # Record user's message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user's message in chat
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process conversation
        result = st.session_state.conversation({"question": prompt})
        
        # Update chat history and get the latest response
        st.session_state.chat_history = result['chat_history']
        response = st.session_state.chat_history[-1]

        # Format sources related to the response
        sources = result['source_documents']
        unique_sources = get_unique_values(sources)
        formatted_sources = format_sources(unique_sources)

        # Display the assistant's response in chat
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Modified typing effect for the response to handle newlines
            for chunk in re.split(r'([\s\n])', response.content):  # Split by space and newline
                if chunk == '\n':  # Preserve newline character as it is
                    full_response += chunk
                else:
                    full_response += chunk + " "
                
                time.sleep(0.04)
                message_placeholder.markdown(full_response + "▌")

            # Format and display the full response with sources
            message_placeholder.markdown(
                format_output(full_response, formatted_sources),
                unsafe_allow_html=True
            )

        # Record the assistant's response
        st.session_state.messages.append({
            "role": "assistant",
            "content": response.content,
            'sources': formatted_sources
        })

        # Rerun the session
        st.experimental_rerun()

    elif pdf_docs is not None:
        st.error("Click \"Process for PDF chat\"")

    else:
        st.error("Please upload PDF documents")

def display_chat_history():
    """Displays the chat history based on the stored session messages."""
    
    for i, message in enumerate(st.session_state.messages):
        
        # Open a chat message for the role (user or assistant)
        with st.chat_message(message["role"]):
            
            # If it's a user message (even index), display content directly
            if i % 2 == 0:
                st.markdown(message["content"])
            
            # If it's an assistant message (odd index), format the content with sources
            else:
                st.markdown(
                    format_output(message['content'], message['sources']),
                    unsafe_allow_html=True
                )

def chat_processing(pdf_docs):
    if st.button("Process for PDF chat", use_container_width=True):
        
        # Check if documents are present
        if len(pdf_docs) == 0:
            st.error("Please upload PDF documents")
            return
        
        # Validate API Key
        if st.session_state.api_key is None or not is_valid_openai_key(st.session_state.api_key):
            st.error("Please enter a valid OpenAI API key before processing.")
            return
        
        # Process the PDF
        with st.spinner("Processing"):
            text_obj = get_pdf_text(pdf_docs)[0]
            text_chunks, metadata = get_text_chunks(text_obj)
            vectorstore = get_vectorstore(text_chunks, metadata)
            st.session_state.conversation = get_conversation_chain(vectorstore)
        
        st.success("Processing Complete")

    # Download chat history if present
    if len(st.session_state.messages) > 0:
        st.download_button(
            'Download Chat History',
            pd.DataFrame(st.session_state.messages).to_csv(index=False),
            file_name='Chat_History.csv',
            use_container_width=True
        )

def csv_processing(pdf_docs):
    if st.button("Retrieve CSV", use_container_width=True):
        
        # Check if documents are present
        if len(pdf_docs) == 0:
            st.error("Please upload PDF documents")
            return
        
        # Process the PDF
        with st.spinner("Processing"):
            text_obj, doc_names = get_pdf_text(pdf_docs)
            text_chunks, metadata = get_text_chunks(text_obj)
            vectorstore = get_vectorstore(text_chunks, metadata)
            dataframe = docs_to_dataframe(vectorstore, doc_names)
            csv_file = dataframe.to_csv(index=False)

        # Offer CSV download button
        st.download_button(
            'Download CSV',
            csv_file,
            file_name='Data.csv',
            use_container_width=True
        )

        # Display total cost
        st.write("Total Cost: ${:.2f}".format(st.session_state.callbacks))

def main():
    # Initialization and Loading
    load_dotenv()
    initialize_states()

    st.title("PDF Chat and Query Bot")

    # Get and validate the API Key
    st.session_state.api_key = st.text_input(
        "Input OpenAI API key",
        type='password',
        help="https://platform.openai.com/account/api-keys",
        placeholder="Enter your OpenAI API key here"
    )

    if st.session_state.api_key and not is_valid_openai_key(st.session_state.api_key):
        st.error('Invalid API key. Please try again.')

    # Sidebar content and PDF Uploader
    with st.sidebar:
        selected = option_menu(
            "Function Menu",
            ["PDF Chat", "PDF to CSV"],
            icons=['chat-fill', 'filetype-csv'],
            menu_icon="stack",
            default_index=0
        )

        st.subheader("Your Documents", help="Try and upload all your files at the same time.")
        
        pdf_docs = st.file_uploader(
            'Upload your PDFs here and click **"Process"**',
            accept_multiple_files=True,
            type=["pdf"]
        )

        # Handle Sidebar Selection
        if selected == "PDF Chat":
            chat_processing(pdf_docs)
        elif selected == "PDF to CSV":
            csv_processing(pdf_docs)

    # Main content based on the sidebar selection
    if selected == "PDF Chat":
        display_chat_history()
        prompt = st.chat_input("What's up?")
        if prompt:
            handle_chat_input(prompt, pdf_docs)
    elif selected == "PDF to CSV":
        st.subheader("CSV Columns and Associated Retrieval Queries")
        query_df = pd.DataFrame.from_dict(query_dict, orient='index', columns=['Queries'])
        st.write(query_df)

if __name__ == '__main__':
    main()