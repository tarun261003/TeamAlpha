import json
import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

# Qdrant cloud configuration
QDRANT_URL = "https://503671a8-0c76-4019-ab6a-3a238095f135.eu-west-2-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6M2QwN2RkYTEtN2Y0MS00YTkxLTgwZTAtZmQ4MWM5ODRjOWRkIn0.qXCI9iSC-mbzUjFDXBbQnNCZQLbvVU3Asvf5798VWJY"
COLLECTION_NAME = "truth-seeker"  # This is what rag_retriever.py defaults to
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset", "tasks.jsonl")

def chunk_text(text, max_chars=1000):
    """Split text into rough chunks."""
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) > max_chars and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            current_chunk += "\n\n" + para
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def main():
    print(f"Reading dataset from {DATASET_PATH}...")
    
    docs_to_embed = []
    with open(DATASET_PATH, 'r') as f:
        for line in f:
            if not line.strip(): continue
            task = json.loads(line)
            # Extract internal_docs for easy and medium tasks
            if "internal_docs" in task:
                for doc_id, doc_text in task["internal_docs"].items():
                    # Chunk to make retrieval similar to what we'd get natively
                    chunks = chunk_text(doc_text)
                    for i, chunk in enumerate(chunks):
                        docs_to_embed.append({
                            "source": doc_id,
                            "text": chunk,
                            "task_id": task["task_id"]
                        })

    print(f"Extracted {len(docs_to_embed)} document chunks from tasks.jsonl.")

    print(f"Loading embedding model '{EMBEDDING_MODEL}'...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    print("Connecting to Qdrant Cloud...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)
    
    vector_size = model.get_sentence_embedding_dimension()
    
    collections = client.get_collections()
    collection_names = [c.name for c in collections.collections]
    
    # Ensure collection matches the one rag_retriever uses
    if COLLECTION_NAME not in collection_names:
        print(f"Collection '{COLLECTION_NAME}' does not exist. Creating it with dimension {vector_size}...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
        )
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")

    print("Embedding chunks and preparing points for upload...")
    points = []
    texts = [d["text"] for d in docs_to_embed]
    embeddings = model.encode(texts, show_progress_bar=True)
    
    for i, (doc, embedding) in enumerate(zip(docs_to_embed, embeddings)):
        point_id = str(uuid.uuid4())
        points.append(
            models.PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    "text": doc["text"], 
                    "source": doc["source"],
                    "task_id": doc["task_id"]
                }
            )
        )
        
    print(f"Upserting {len(points)} points into collection '{COLLECTION_NAME}'...")
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )
    
    print("Upload completed successfully! You can now run inference.")

if __name__ == "__main__":
    main()
