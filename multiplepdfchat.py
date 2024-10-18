import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from key import GOOGLE_API_KEY
import os
os.environ["GOOGLE_API_KEY"]=GOOGLE_API_KEY


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    try:
        embedding = GoogleGenerativeAIEmbeddings(model='models/embedding-001')  # Ensure this is the correct model
        vector_store = FAISS.from_texts(text_chunks, embedding=embedding)
        vector_store.save_local("faiss_index")
    except Exception as e:
        st.error(f"Error embedding content: {e}")
def get_conversational_chain():
    prompt_template = """
    You are an expert assistant with access to detailed information from PDFs provided as context. 
    Answer the question comprehensively, providing as much relevant and in-depth detail as possible. Be factual, and make sure to cover all aspects of the question using the information available in the context.

    If the answer is not explicitly mentioned in the provided context, respond with: 
    "Answer is not available in the provided context."

    Ensure that the answer is thorough, well-explained, and directly related to the question. Include examples or references from the context if applicable.

    Context:
    {context}

    Question:
    {question}

    Detailed Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)  # Ensure this is the correct model
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    try:
        embedding = GoogleGenerativeAIEmbeddings(model='models/embedding-001')  # Ensure this is the correct model
        new_db = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()

        # Check if docs is not empty
        if not docs:
            st.error("No relevant documents found.")
            return

        response = chain({
            "input_documents": docs,  # Ensure the correct key is used here
            "question": user_question
        }, return_only_outputs=True)

        st.write("Reply:", response.get("output_text", "No response generated."))

    except Exception as e:
        st.error(f"Error processing user input: {e}")


def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")
    st.button("Ask")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",
                                    accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")


if __name__ == "__main__":
    main()