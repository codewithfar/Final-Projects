import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Page configuration
st.set_page_config(
    page_title="RAG Q&A Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 RAG Q&A Chatbot")
st.write("Ask questions from the knowledge base.")

# Load embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Load language model
@st.cache_resource
def load_language_model():
    model_name = "google/flan-t5-small"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model

# Connect to Chroma
@st.cache_resource
def load_database():
    client = chromadb.PersistentClient(
        path="data/chroma_db"
    )

    collection = client.get_collection(
        name="rag_documents"
    )

    return collection

embedding_model = load_embedding_model()
tokenizer, llm = load_language_model()
collection = load_database()

# User question
query = st.text_input(
    "Enter your question:"
)

if st.button("Get Answer"):

    if query.strip():

        # Create query embedding
        query_embedding = embedding_model.encode(
            [query]
        ).tolist()

        # Retrieve relevant documents
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=3
        )

        retrieved_chunks = results["documents"][0]

        # Create context
        context = "\n".join(retrieved_chunks)

        # Create prompt
        prompt = f"""
        Answer the question using only the context below.

        Context:
        {context}

        Question:
        {query}

        Give a short and clear answer.
        """

        # Generate answer
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )

        outputs = llm.generate(
            **inputs,
            max_new_tokens=100
        )

        answer = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        # Display answer
        st.subheader("Answer")
        st.write(answer)

        # Display sources
        st.subheader("Retrieved Sources")

        for i, chunk in enumerate(retrieved_chunks):
            st.write(f"**Source {i + 1}:**")
            st.write(chunk)

    else:
        st.warning("Please enter a question.")