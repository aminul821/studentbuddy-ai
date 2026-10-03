import os
import hashlib
import logging
import chromadb

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ==========================================
# CONFIGURATION
# ==========================================

# Resolve paths relative to this file so the app works from any working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge_base")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_store")

MODEL_NAME = "all-MiniLM-L6-v2"

COLLECTION_NAME = "studentbuddy"

# Maximum distance for a chunk to count as relevant (lower = stricter)
SIMILARITY_THRESHOLD = 0.65

N_RESULTS = 5

# Loaded once per process: Python caches imported modules across Streamlit reruns
embedding_model = SentenceTransformer(MODEL_NAME)

client = chromadb.PersistentClient(path=VECTOR_DB_DIR)

collection = client.get_or_create_collection(COLLECTION_NAME)

# ==========================================
# FILE HELPERS
# ==========================================

def list_knowledge_files():

    return sorted(
        f for f in os.listdir(KNOWLEDGE_DIR)
        if f.endswith(".txt")
    )


def calculate_hash():

    md5 = hashlib.md5()

    # Changing the embedding model must also trigger a rebuild
    md5.update(MODEL_NAME.encode("utf-8"))

    for filename in list_knowledge_files():

        # Include the filename so renames also trigger a rebuild
        md5.update(filename.encode("utf-8"))

        with open(os.path.join(KNOWLEDGE_DIR, filename), "rb") as f:
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

            if current.strip():
                chunks.append(current.strip())

            current = para

    if current.strip():

        chunks.append(current.strip())

    return chunks


# ==========================================
# BUILD DATABASE
# ==========================================

def build_vector_store():

    global collection

    current_hash = calculate_hash()

    hash_file = os.path.join(VECTOR_DB_DIR, "hash.txt")

    if os.path.exists(hash_file) and collection.count() > 0:

        with open(hash_file) as f:
            old_hash = f.read().strip()

        if old_hash == current_hash:

            return

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        # Collection may not exist yet
        pass

    collection = client.get_or_create_collection(COLLECTION_NAME)

    documents = []
    metadatas = []

    for filename in list_knowledge_files():

        with open(os.path.join(KNOWLEDGE_DIR, filename), encoding="utf-8") as f:

            text = f.read()

        for chunk in chunk_text(text):

            documents.append(chunk)

            metadatas.append({"source": filename})

    if documents:

        # Encode and insert all chunks in one batch
        embeddings = embedding_model.encode(documents).tolist()

        collection.add(
            ids=[str(i) for i in range(len(documents))],
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    os.makedirs(VECTOR_DB_DIR, exist_ok=True)

    with open(hash_file, "w") as f:

        f.write(current_hash)


# ==========================================
# SEARCH
# ==========================================

def retrieve_context(question):

    count = collection.count()

    if count == 0:
        return None

    embedding = embedding_model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(N_RESULTS, count),
        include=["documents", "metadatas", "distances"]
    )

    docs = results["documents"][0]
    sources = results["metadatas"][0]
    distances = results["distances"][0]

    context = []

    for doc, source, distance in zip(docs, sources, distances):

        logger.debug("%s | Distance: %.3f", source["source"], distance)

        if distance <= SIMILARITY_THRESHOLD:

            context.append(
                f"""Source: {source['source']}

{doc}
"""
            )

    if not context:
        return None

    return "\n\n".join(context)


build_vector_store()
