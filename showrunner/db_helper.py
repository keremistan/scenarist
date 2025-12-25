from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from pprint import pprint

embedding_model = OllamaEmbeddings(model='nomic-embed-text', temperature=0)
screenplays_vector_store_collection = Chroma(
    collection_name="screenplays", 
    embedding_function=embedding_model, 
    persist_directory='./chroma')


def delete_collection():
    screenplays_vector_store_collection.reset_collection()
    
    
if __name__ == "__main__":
    # delete_collection()
    pass