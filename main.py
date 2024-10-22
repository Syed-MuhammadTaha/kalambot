import streamlit as st
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain import hub
from streamlit_chat import message
import os 
from langchain.vectorstores import FAISS
from dotenv import load_dotenv


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
loader = PyPDFLoader("Kalambot_Info.pdf")
documents = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200, add_start_index=True)
all_splits = text_splitter.split_documents(documents)

vectorstore = FAISS.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

# Perform similarity search (retriever based on FAISS)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})


# Load the RAG prompt template
template = """Use the following pieces of context to answer the question at the end.
You are a chatbot and answer questions. Keep the sentence concise. Keep the answer concise.

{context}

Question: {question}

Helpful Answer:"""

base_prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": lambda x: x}
    | base_prompt | model
    | StrOutputParser()
)

# Base chain
base_chain = base_prompt | model | StrOutputParser()

# Checking the relation between answers and questions
check_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an intelligent assistant tasked with evaluating the relationship between two pieces of text. 1. Analyze the previous answer ('{previous_answer}') for its key concepts, themes, and specific details. 2. Examine the current prompt ('{text}') to identify its main focus and intent. 3. Consider whether both texts discuss the same entity and if the current prompt seeks further elaboration, specific details, or information about the entity's structure or personnel based on the previous context. Respond with one of the following: - 'Match' if the previous answer provides relevant context or details that can be extended to inform or enhance the current prompt. - 'Related' if the previous answer does not directly address the current prompt but still pertains to the same entity in a broader sense. - 'No Match' if the previous answer does not offer applicable ideas, context, or further details about the same entity for the current prompt."),
        ("user", "Follow-up question: {text}. Previous answer: {previous_answer}.")
    ]
)

check_prompt_chain = check_prompt_template | model | StrOutputParser()

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
        # Handle first input
        output = rag_chain.invoke(user_input)
        st.session_state.previous_chain_type = "rag"
    st.session_state.previous_answer = output
    # Add bot's response to the session state
    st.session_state.messages.append({"role": "bot", "content": output})
    
with st.container():
    for idx, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            with st.chat_message("user", avatar=":material/thumb_up:"):
                st.write(msg["content"])
        else:
            with st.chat_message("ai"):
                st.write(msg["content"])

