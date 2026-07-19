import os
import hashlib
import chromadb

from sentence_transformers import SentenceTransformer
from streamlit import context

# ==========================================
# CONFIGURATION
# ==========================================

KNOWLEDGE_DIR = "knowledge_base"
VECTOR_DB_DIR = "vector_store"

MODEL_NAME = "all-MiniLM-L6-v2"

embedding_model = SentenceTransformer(MODEL_NAME)

client = chromadb.PersistentClient(path=VECTOR_DB_DIR)

COLLECTION_NAME = "studentbuddy"

collection = client.get_or_create_collection(COLLECTION_NAME)

# ==========================================
# FILE HASH
# ==========================================

def calculate_hash():

    md5 = hashlib.md5()

    for filename in sorted(os.listdir(KNOWLEDGE_DIR)):

        path = os.path.join(KNOWLEDGE_DIR, filename)

        if not filename.endswith(".txt"):
            continue

        with open(path, "rb") as f:
            md5.update(f.read())

    return md5.hexdigest()


# ==========================================
# CHUNKING
# ==========================================

def chunk_text(text, chunk_size=400):

    paragraphs = text.split("\n")

    chunks = []

    current = ""

    for para in paragraphs:

        para = para.strip()

        if not para:
            continue

        if len(current) + len(para) < chunk_size:

            current += "\n" + para

        else:

            chunks.append(current.strip())

            current = para

    if current:

        chunks.append(current.strip())

    return chunks


# ==========================================
# BUILD DATABASE
# ==========================================

def build_vector_store():

    current_hash = calculate_hash()

    hash_file = os.path.join(VECTOR_DB_DIR, "hash.txt")

    if os.path.exists(hash_file):

        old_hash = open(hash_file).read().strip()

        if old_hash == current_hash:

            return

    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass

    global collection

    collection = client.get_or_create_collection(COLLECTION_NAME)

    doc_id = 0

    for filename in os.listdir(KNOWLEDGE_DIR):

        if not filename.endswith(".txt"):
            continue

        path = os.path.join(KNOWLEDGE_DIR, filename)

        with open(path, encoding="utf-8") as f:

            text = f.read()

        chunks = chunk_text(text)

        for chunk in chunks:

            embedding = embedding_model.encode(chunk).tolist()

            collection.add(

                ids=[str(doc_id)],

                documents=[chunk],

                embeddings=[embedding],

                metadatas=[

                    {

                        "source": filename

                    }

                ]

            )

            doc_id += 1

    with open(hash_file, "w") as f:

        f.write(current_hash)


# ==========================================
# SEARCH
# ==========================================

def retrieve_context(question):

    embedding = embedding_model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )

    docs = results["documents"][0]
    sources = results["metadatas"][0]
    distances = results["distances"][0]

    context = []

    SIMILARITY_THRESHOLD = 0.65

    for doc, source, distance in zip(docs, sources, distances):

        print(
            f"{source['source']} | Distance: {distance:.3f}"
        )

        if distance <= SIMILARITY_THRESHOLD:

            context.append(
                f"""Source: {source['source']}

{doc}
"""
            )

    if len(context) == 0:
        return None

    return "\n\n".join(context)


build_vector_store()