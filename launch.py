import requests
import streamlit as st 
from qdrant_client import QdrantClient
from generationthis import prompt_template_generation, create_query_engine
from pypdf import PdfReader
from io import BytesIO
from llama_index.core.schema import Document
from doc_processingthis import Custom_transformation

@st.cache_data
def delete_existing():
    client = QdrantClient(url="http://localhost:6333")
    client.delete_collection("new_rag_docs")

delete_existing()  

st.set_page_config(page_title="Hybrid RAG with Qdrant BM42 & Mistral", layout="wide")
st.title("Hybrid RAG with Qdrant BM42 & Mistral 8x7B")

uploaded_pdfs = st.file_uploader(
    "upload a pdf",
    type=["pdf"],
    accept_multiple_files=True
)

@st.cache_data(show_spinner="Extracting PDF text...")
def extract_pdf_text_cached(pdf_bytes: bytes, pdf_name: str):
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append(
                {
                    "page": i + 1,
                    "text": text,
                    "source": pdf_name,
                }
            )
    return pages

if "indexed" not in st.session_state:
    st.session_state.indexed = False

if uploaded_pdfs and not st.session_state.indexed:
    st.success("pdf uploaded succesfully")
    all_pdfs = []
    for pdf in uploaded_pdfs:
        pdf_bytes = pdf.getvalue() 
        pdf = extract_pdf_text_cached(pdf_bytes, pdf.name)
        all_pdfs.extend(pdf)

    documents = [
                Document(
                    text = page["text"],
                    metadata = {
                        "filename" : page["source"],
                        "source" : "streamlit"
                    },
                ) for page in all_pdfs
            ]
    if st.button("Index documents"):
        st.session_state.indexed = True
        Custom_transformation(documents)
        st.success('INDEXING IS DONE')

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Ask me anything about different RAG frameworks!")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        query_str = st.session_state.messages[-1]["content"]
        prompt_gen = prompt_template_generation()
        prompt = prompt_gen.prompt_generation(query=query_str)
        response = create_query_engine(prompt)
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
