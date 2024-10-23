import streamlit as st
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain import hub
from streamlit_chat import message
import os 
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from docx import Document

hide_streamlit_style = """
            <style>
            [data-testid="stAppToolbar"] {visibility: hidden !important;}
            footer {visibility: hidden !important;}
            
            </style>
            """

# Set environment variables
load_dotenv()

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['USER_AGENT'] = 'myagent'
# Initialize the model and necessary components
model = ChatOpenAI(model="gpt-4o-mini")
loader = Docx2txtLoader("Kalambot_Info.docx")

documents = loader.load()

def extract_headers_and_content(doc_path):
    document = Document(doc_path)
    content_dict = {}
    current_header = None

    for para in document.paragraphs:
        # Check if the paragraph is a header (you can adjust the conditions)
        if para.style.name.startswith('Heading'):
            current_header = para.text.strip()
            content_dict[current_header] = ''
        elif current_header:
            # Append content to the current header's content
            content_dict[current_header] += para.text.strip() + "\n"

    return content_dict

headers_content = extract_headers_and_content("/content/Kalambot_Info.docx")

# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=200, add_start_index=True
)
all_splits = []
for doc in documents:
    splits = text_splitter.split_documents([doc])  # Use the Document object directly
    all_splits.extend(splits)

vectorstore = FAISS.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

# Perform similarity search (retriever based on FAISS)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})


# Load the RAG prompt template
template = """Use the following pieces of context to answer the question at the end.
You are a chatbot and answer questions. Keep the sentence concise. Keep the answer concise.

{context}

Question: {question}

Helpful Answer:"""

prompt = hub.pull("rlm/rag-prompt")
base_prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": lambda x: x}
    | prompt | model
    | StrOutputParser()
)

# Base chain
base_chain = base_prompt | model | StrOutputParser()

check_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an intelligent assistant tasked with evaluating whether the current question ('{text}') is a follow-up to the previous answer ('{previous_answer}'). 1. Analyze the previous answer for its entity. 2. Examine the current question to determine if it seeks further information or elaboration about the same entity or concept discussed in the previous answer. 3. Ensure both texts refer to the same entity (such as a company, individual, or event) and that the current question logically follows from the previous answer. Respond with one of the following: - 'Match' if the current question is a clear follow-up and refers to the same entity or concept as the previous answer. - 'Related' if the current question touches on the same entity or concept but is not a direct follow-up. - 'No Match' if the current question is unrelated to the previous answer."),
        ("user", "Follow-up question: {text}. Previous answer: {previous_answer}.")
    ]
)

check_prompt_chain = check_prompt_template | model | StrOutputParser()

prompt_template = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful assistant. Please answer the following question whether the user is talking about fine tuning a dataset in either 'yes' or 'no'"), ("user", "{text}")]
)
fine_tune_chain = prompt_template | model | StrOutputParser()

# Initialize session state for messages and context
if "messages" not in st.session_state:
    st.session_state.messages = []
if "previous_answer" not in st.session_state:
    st.session_state.previous_answer = ""
if "previous_chain_type" not in st.session_state:
    st.session_state.previous_chain_type = None

# Streamlit UI

st.set_page_config(page_title="Chatbot Interface", page_icon="💬")
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

user_input = st.chat_input("You:", key="input", on_submit=lambda: st.session_state.update({"enter_pressed": True}))

# Initialize session state for 'enter_pressed'
if "enter_pressed" not in st.session_state:
    st.session_state.enter_pressed = False

# Check if Send button or Enter was pressed
if (st.session_state.enter_pressed) and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    is_finetune = fine_tune_chain.invoke(user_input).strip().lower() == "yes"
    # Define the logic to use the correct chain based on previous context
    if st.session_state.previous_answer:
        
        match_result = check_prompt_chain.invoke({"previous_answer": st.session_state.previous_answer, "text": user_input})
        if match_result == "Match":
            combined_input = f"{st.session_state.previous_answer} {user_input}"
            if st.session_state.previous_chain_type == "base":
                output = base_chain.invoke({"context": st.session_state.previous_answer, "question": user_input})
            else:
                output = rag_chain.invoke(combined_input)
        else:
            if st.session_state.previous_chain_type == "base":
                output = rag_chain.invoke(user_input)
                st.session_state.previous_chain_type = "rag"
            else:
                output = base_chain.invoke({"context": "", "question": user_input})
                st.session_state.previous_chain_type = "base"
    else:
        if is_finetune:
            output = base_chain.invoke({"context": "", "question": user_input})
            st.session_state.previous_chain_type = "base"
        else:
            output = rag_chain.invoke(user_input)
            st.session_state.previous_chain_type = "rag"
    st.session_state.previous_answer = output
    # Add bot's response to the session state
    st.session_state.messages.append({"role": "bot", "content": output})

with st.container():
    for idx, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("ai", avatar="logo.png"):
                st.write(msg["content"])
# test


