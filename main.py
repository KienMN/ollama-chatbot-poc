import streamlit as st
import PyPDF2

# from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain

# from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.schema import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from PIL import Image
import fitz
import os

st.set_page_config(page_title="Ollama RAG Chatbot", page_icon="🤖", layout="wide")

with st.sidebar:
    st.title("PDF Upload")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        st.success("File uploaded successfully!")

        # PDF preview
        st.subheader("PDF Preview")
        pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        num_pages = pdf_document.page_count
        page_num = st.number_input("Page", min_value=1, max_value=num_pages, value=1)

        page = pdf_document.load_page(page_num - 1)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        st.image(img, caption=f"Page {page_num}", use_container_width=True)

st.title("Ollama RAG Chatbot")

if "chain" not in st.session_state:
    st.session_state.chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "user_question" not in st.session_state:
    st.session_state.user_question = ""


def process_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


if uploaded_file is not None:
    if st.session_state.chain is None:
        with st.spinner("Processing PDF..."):
            pdf_text = process_pdf(uploaded_file)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=100
            )
            texts = text_splitter.split_text(pdf_text)

            metadatas = [{"source": f"chunk_{i}"} for i in range(len(texts))]

            embeddings = OllamaEmbeddings(model="nomic-embed-text")

            if os.path.exists("./db") and os.path.isdir("./db"):
                print("Read from db")
                docsearch = Chroma(
                    persist_directory="./db", embedding_function=embeddings
                )

            else:
                print("Create a new db")
                docsearch = Chroma.from_texts(
                    texts, embeddings, metadatas=metadatas, persist_directory="./db"
                )

            message_history = ChatMessageHistory()
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                output_key="answer",
                chat_memory=message_history,
                return_messages=True,
            )

            st.session_state.chain = ConversationalRetrievalChain.from_llm(
                ChatOllama(model="llama3.1", temperature=0.7),
                chain_type="stuff",
                retriever=docsearch.as_retriever(search_kwargs={"k": 1}),
                memory=memory,
                return_source_documents=True,
            )

            st.success("PDF processed successfully!")


def submit_question():
    st.session_state.user_question = st.session_state.question_input
    st.session_state.question_input = ""


st.subheader("Chat with your PDF")
st.text_input(
    "Ask a question about the document", key="question_input", on_change=submit_question
)

user_input = st.session_state.user_question
st.session_state.user_question = ""
print("userinput", user_input, len(user_input))

if user_input:
    print("user input condition", user_input)
    if st.session_state.chain is None:
        st.warning("Please upload a PDF file first!")
    else:
        with st.spinner("Thinking..."):
            response = st.session_state.chain.invoke({"question": user_input})
            answer = response["answer"]
            source_documents = response["source_documents"]

            st.session_state.chat_history.append(HumanMessage(user_input))
            st.session_state.chat_history.append(AIMessage(answer))
    # st.session_state.question_input = ""
    user_input = ""

# chat_container = st.container()
# with chat_container:
#     for message in reversed(st.session_state.chat_history):
#         if isinstance(message, HumanMessage):
#             st.write(f"**You:** {message.text}")
#         else:
#             st.write(f"**Ollama:** {message.text}")

#         if isinstance(message, AIMessage):
#             with st.expander("View Sources"):
#                 for idx, doc in enumerate(source_documents):
#                     st.write(f"**Source {idx + 1}:** {doc.page_content[:150]}...")

chat_container = st.container()
with chat_container:
    print(len(st.session_state.chat_history))
    for message in reversed(st.session_state.chat_history):
        print(message)
        if isinstance(message, HumanMessage):
            st.markdown(f"👤 {message.content}")
        elif isinstance(message, AIMessage):
            st.markdown(f"🤖 {message.content}")

        if isinstance(message, AIMessage):
            # if source_documents is not None:
            with st.expander("View Sources"):
                try:
                    for idx, doc in enumerate(source_documents):
                        st.write(f"Source {idx + 1}:", doc.page_content[:150] + "...")
                except:
                    print("No source found")
