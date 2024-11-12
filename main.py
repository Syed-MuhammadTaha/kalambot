import streamlit as st
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
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

headers_content = extract_headers_and_content("Kalambot_Info.docx")

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


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_template = """
You are Kalambot, chatbot for the company Kalambot. Answer only questions related to the company, its values, operations, policies, and other relevant information about Kalambot. If the question is unrelated to the company (e.g., questions about fine-tuning, personal advice, or unrelated topics), return an empty string as the response. Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Response:
"""



rag_prompt = ChatPromptTemplate.from_template(rag_template)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt | model
    | StrOutputParser()
)

base_memory = ConversationBufferWindowMemory(input_key="question", memory_key="context", k=2)

generic_template = """Use the following pieces of context to answer the question at the end.
You are a chatbot and answer questions. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

{context}

Question: {question}

Helpful Answer:"""

generic_prompt = ChatPromptTemplate.from_template(generic_template)

base_chain = LLMChain(
    prompt=generic_prompt,
    llm=model,
    memory=base_memory,
    output_parser=StrOutputParser()
)

check_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are KalamBot, an intelligent assistant tasked with determining if the user's current question is a follow-up to the same subject of previous response. Follow these steps to decide:\n\n"
         "1. **Identify the Main Subject**: First, identify the main subject or entity in the previous response ('{previous_answer}'). For example, if the previous response is 'happy to help you,' the subject is likely the chatbot itself.\n\n"
         "2. **Analyze the Current Question**: Check if the current question ('{text}') addresses the same subject. Look for indications that the question refers back to the same entity, such as references to 'you,' 'the company,' or other specific terms linked to the identified subject.\n\n"
         "3. **Determine Match or No Match**: If the current question logically follows from the previous response and both refer to the same subject or entity, categorize as 'match.' Otherwise, categorize as 'No Match'.\n\n"
         "Respond with one of the following:\n"
         "- 'match' if the question clearly follows from and refers to the same entity or concept as the previous response.\n"
         "- 'No Match' if the question is unrelated to the previous response."
        ),
        ("user", "Follow-up question: {text}. Previous answer: {previous_answer}.")
    ]
)



check_prompt_chain = check_prompt_template | model | StrOutputParser()

# Initialize session state for messages and context
if "messages" not in st.session_state:
    st.session_state.messages = []
if "previous_answer" not in st.session_state:
    st.session_state.previous_answer = ""
if "chain_type" not in st.session_state:
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

    combined_input = f"{st.session_state.previous_answer} {user_input}"
    is_match = check_prompt_chain.invoke({"previous_answer": st.session_state.previous_answer, "text": user_input})
    # Define the logic to use the correct chain based on previous context
    if is_match == "match":
        combined_input = f"{st.session_state.previous_answer} {user_input}"

    # Use the appropriate chain based on previous chain type
        if st.session_state.chain_type == "base":
            response = base_chain.invoke(combined_input)
            response = response["text"]
        else:
            response = rag_chain.invoke(combined_input)

    # Handle non-matching case
    else:
        response = rag_chain.invoke(user_input)

        # If rag_chain yields no response, switch to base_chain
        if not response:
            response = base_chain.invoke(user_input)["text"]
            st.session_state.chain_type = "base"
        else:
            st.session_state.chain_type = "rag"

    # Update the previous answer in session state for the next interaction
    st.session_state.previous_answer = response
    # Add bot's response to the session state
    st.session_state.messages.append({"role": "bot", "content": response})

with st.container():
    for idx, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("ai", avatar="logo.png"):
                st.write(msg["content"])
# test


