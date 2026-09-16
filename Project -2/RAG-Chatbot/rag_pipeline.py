import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --------------------------------
# 1. Load Embedding Model
# --------------------------------
print("Loading embedding model...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------------
# 2. Connect to Chroma Database
# --------------------------------
client = chromadb.PersistentClient(
    path="data/chroma_db"
)

collection = client.get_collection(
    name="rag_documents"
)

print("Chroma database connected!")
print("Documents in database:", collection.count())

# --------------------------------
# 3. Load FLAN-T5 Language Model
# --------------------------------
print("\nLoading language model...")

model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)

llm = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print("Language model loaded successfully!")

# --------------------------------
# 4. Ask Question
# --------------------------------
query = input("\nAsk your question: ")

# --------------------------------
# 5. Convert Question to Embedding
# --------------------------------
query_embedding = embedding_model.encode(
    [query]
).tolist()

# --------------------------------
# 6. Retrieve Relevant Chunks
# --------------------------------
results = collection.query(
    query_embeddings=query_embedding,
    n_results=3
)

retrieved_chunks = results["documents"][0]

print("\n--- RETRIEVED INFORMATION ---")

for i, chunk in enumerate(retrieved_chunks):
    print(f"\nSource {i + 1}:")
    print(chunk)

# --------------------------------
# 7. Create Context
# --------------------------------
context = "\n".join(retrieved_chunks)

# --------------------------------
# 8. Create Prompt
# --------------------------------
prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{query}

Give a short and clear answer.
"""

# --------------------------------
# 9. Tokenize Prompt
# --------------------------------
inputs = tokenizer(
    prompt,
    return_tensors="pt",
    max_length=512,
    truncation=True
)

# --------------------------------
# 10. Generate Answer
# --------------------------------
outputs = llm.generate(
    **inputs,
    max_new_tokens=100
)

answer = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

# --------------------------------
# 11. Display Final Answer
# --------------------------------
print("\n--- FINAL ANSWER ---")
print(answer)