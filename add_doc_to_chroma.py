import time
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
import time
import os

CHROMA_PATH = "chroma_db_test_21"  # Chemin vers le répertoire de la base de données Chroma
DATA_PATH = "D:\\Generative IA\\rag_pdfs\\pdf_data"

PROMPT_TEMPLATE = """
Answer the question based on the following context and your knowledge:

{context}

---

Answer the question based on the above context and your knowledge: {question} """
def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    for doc in document_loader.load():
        yield doc  # Utilisation d'un générateur pour éviter de charger tous les documents en mémoire

def split_documents(doc):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents([doc])  # Retourne les chunks pour un seul document

def add_to_chroma_parallel(num_workers=None, batch_size=10):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

    existing_items = db.get(include=[])  # Vérification des documents existants
    existing_ids = set(existing_items["ids"])
    print(f"Nombre de documents existants dans la base de données: {len(existing_ids)}")

    # Boucle pour charger les documents et les ajouter en parallèle
    for document in load_documents():
            chunks = split_documents(document)
            chunks_with_ids = calculate_chunk_ids(chunks)

            # Filtrer les nouveaux chunks
            new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]
            if not new_chunks:
                print("Aucun nouveau document à ajouter pour ce document.")
                continue

            print(f"Ajout de {len(new_chunks)} nouveaux documents...")
            chunk_batches = list(chunkify(new_chunks, batch_size))

            for chunk_batch in chunk_batches:
                add_documents_batch(db, chunk_batch)
                time.sleep(0.1)

            del new_chunks# Optionnel : pause pour éviter la surcharge

def chunkify(lst, n):
    """Divise la liste `lst` en `n` sous-listes à peu près égales."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def add_documents_batch(db, chunk_batch):
    """Ajoute un lot de documents à la base de données Chroma."""
    new_chunk_ids = [chunk.metadata["id"] for chunk in chunk_batch]
    db.add_documents(chunk_batch, ids=new_chunk_ids)
    print(f"Ajouté {len(new_chunk_ids)} documents.")

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks

# Appel de la fonction

if __name__ == "__main__":
    add_to_chroma_parallel(batch_size=10)
