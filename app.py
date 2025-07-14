import streamlit as st
from retriever import Retriever
from generator import Generator

st.title("RAG Q&A Chatbot - Loan Approval Dataset")

query = st.text_input("Ask a question about the loan data:")
retriever = Retriever("data/Training_Dataset.csv")
generator = Generator()

if query:
    docs = retriever.retrieve(query)
    context = " ".join(docs.astype(str).apply(lambda x: ' '.join(x), axis=1).tolist())
    answer = generator.generate_answer(query, context)
    st.write("**Answer:**", answer)
