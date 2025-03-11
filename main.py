import os
import shutil

import fitz
import PyPDF2
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama, OllamaEmbeddings
from loguru import logger
from PIL import Image

db_directory = "./db"
embedding_model = "nomic-embed-text"
chat_model = "llama3.1"

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. " \

{context}
"""

st.set_page_config(
    page_title="Documents Reader Assistant", page_icon="🤖", layout="wide"
)

with st.sidebar:
    st.title("PDF Document Upload")
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

st.title("Documents Reader Assistant")

if "chain" not in st.session_state:
    st.session_state.chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "user_question" not in st.session_state:
    st.session_state.user_question = ""


def process_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = []
    for i, page in enumerate(pdf_reader.pages):
        text.append({"content": page.extract_text(), "page_number": i + 1})
    return text


if uploaded_file is not None:
    if st.session_state.chain is None:
        with st.spinner("Processing PDF..."):
            pdf_text = process_pdf(uploaded_file)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=100
            )

            texts = []
            metadatas = []

            for page in pdf_text:
                chunks = text_splitter.split_text(page.get("content", ""))
                texts.extend(chunks)
                metadatas.extend(
                    [
                        {
                            "source": f"chunk_{i}",
                            "page_number": page.get("page_number", None),
                        }
                        for i in range(len(chunks))
                    ]
                )

            embeddings = OllamaEmbeddings(model=embedding_model)

            # Delete db folder if it exists
            if os.path.exists(db_directory) and os.path.isdir(db_directory):
                shutil.rmtree(db_directory)

            docsearch = Chroma.from_texts(
                texts, embeddings, metadatas=metadatas, persist_directory=db_directory
            )

            llm = ChatOllama(model=chat_model, temperature=0.7)
            retriever = docsearch.as_retriever(search_kwargs={"k": 3})

            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

            history_aware_retriever = create_history_aware_retriever(
                llm, retriever, contextualize_q_prompt
            )

            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", qa_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

            rag_chain = create_retrieval_chain(
                history_aware_retriever, question_answer_chain
            )
            st.session_state.chain = rag_chain

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
logger.info("userinput: {}, {}", user_input, len(user_input))

if user_input:
    logger.info("user input condition {}", user_input)
    if st.session_state.chain is None:
        st.warning("Please upload a PDF file first!")
    else:
        with st.spinner("Thinking..."):
            response = st.session_state.chain.invoke(
                {"input": user_input, "chat_history": st.session_state.chat_history}
            )
            answer = response["answer"]
            source_documents = response.get("context", None)
            # logger.info(source_documents)

            st.session_state.chat_history.append(("human", user_input))
            st.session_state.chat_history.append(("ai", answer))
            # logger.info("chat history: {}", st.session_state.chat_history)

    user_input = ""

chat_container = st.container()
with chat_container:
    for message in reversed(st.session_state.chat_history):
        if message[0] == "human":
            st.markdown(f"👤 {message[1]}")
        elif message[0] == "ai":
            st.markdown(f"🤖 {message[1]}")

        if message[0] == "ai":
            if source_documents is not None:
                with st.expander("View Sources"):
                    try:
                        for doc in source_documents:
                            st.write(
                                f"Page {doc.metadata.get('page_number')}:",
                                doc.page_content[:150] + "...",
                            )
                    except:
                        logger.info("No source found")
