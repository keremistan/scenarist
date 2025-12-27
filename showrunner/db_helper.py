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
    
    
def delete_docs_by_id(ids: list[str]):
    screenplays_vector_store_collection.delete(ids)
    
def search_docs():
    res = screenplays_vector_store_collection.get(where_document={"$contains": "0"}, include=['documents'])
    print(res)


if __name__ == "__main__":
    # delete_collection() # comment out to delete db mistakenly
    # search_docs()
    
    # ids = []
    # delete_docs_by_id(ids)
    pass