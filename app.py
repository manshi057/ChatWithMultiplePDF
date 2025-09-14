import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from transformers import pipeline
from langchain.llms import HuggingFacePipeline


# ---- Extract text from PDFs ----
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text


# ---- Split text into chunks ----
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(text)


# ---- Create vectorstore ----
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(text_chunks, embeddings)


# ---- Build LLM with HuggingFace pipeline ----
def get_conversation_chain(vectorstore):
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        max_new_tokens=512,
        temperature=0.5
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )


# ---- Handle user input ----
def handle_userinput(user_question):
    response = st.session_state.conversation.invoke({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for msg in st.session_state.chat_history:
        if msg.type == "human":
            st.markdown(f"**You:** {msg.content}")
        else:
            st.markdown(f"**Bot:** {msg.content}")


# ---- Streamlit App ----
def main():
    st.set_page_config(page_title="Chat with PDFs", page_icon="📄")
    st.header("📄 Chat with your PDFs")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input("Ask a question about your documents:")
    if user_question and st.session_state.conversation:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Upload Documents")
        pdf_docs = st.file_uploader("Upload your PDFs", accept_multiple_files=True)
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.success("Done! Now ask questions on the left 👈")
            else:
                st.warning("Please upload at least one PDF.")


if __name__ == "__main__":
    main()
