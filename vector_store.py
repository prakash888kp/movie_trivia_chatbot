from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings

def create_vector_store(chunks, collection_name="movie_trivia"):
    embeddings = OllamaEmbeddings(model="granite-embedding:latest")
    return Chroma.from_documents(chunks, embedding=embeddings, collection_name=collection_name)
