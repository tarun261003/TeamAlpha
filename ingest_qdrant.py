import os
import uuid
import docx
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

# Qdrant cloud configuration
QDRANT_URL = "https://503671a8-0c76-4019-ab6a-3a238095f135.eu-west-2-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6M2QwN2RkYTEtN2Y0MS00YTkxLTgwZTAtZmQ4MWM5ODRjOWRkIn0.qXCI9iSC-mbzUjFDXBbQnNCZQLbvVU3Asvf5798VWJY"
COLLECTION_NAME = "Docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DOC_PATH = r"c:\Users\H PATEL\Downloads\OpenEnv\OpenEnv Hackathon Solution Proposal.docx"

def read_docx(path):
    print(f"Reading document from {path}...")
    doc = docx.Document(path)
    fullText = []
    for para in doc.paragraphs:
        if para.text.strip():
            fullText.append(para.text)
    return fullText

def chunk_text(paragraphs, chunk_size=3):
    """Group every `chunk_size` paragraphs into a chunk."""
    chunks = []
    for i in range(0, len(paragraphs), chunk_size):
        chunk = "\n\n".join(paragraphs[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def main():
    paragraphs = read_docx(DOC_PATH)
    chunks = chunk_text(paragraphs, chunk_size=3)
    print(f"Extracted {len(chunks)} chunks from the document.")

    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    print("Connecting to Qdrant Cloud...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)
    
    # Ensure collection exists and matches the embedding dimension
    vector_size = model.get_sentence_embedding_dimension()
    
    collections = client.get_collections()
    collection_names = [c.name for c in collections.collections]
    
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
    embeddings = model.encode(chunks, show_progress_bar=True)
    
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point_id = str(uuid.uuid4())
        points.append(
            models.PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={"text": chunk, "source": os.path.basename(DOC_PATH)}
            )
        )
        
    print(f"Upserting {len(points)} points into collection '{COLLECTION_NAME}'...")
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )
    
    print("Upload completed successfully!")

if __name__ == "__main__":
    main()
