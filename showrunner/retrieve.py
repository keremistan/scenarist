from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from pprint import pprint

embedding_model = OllamaEmbeddings(model='nomic-embed-text', temperature=0)
screenplays_vector_store_collection = Chroma(
    collection_name="screenplays", 
    embedding_function=embedding_model, 
    persist_directory='./chroma')

query = "how can I create a suspense and tension?"
# res_sim = screenplays_vector_store_collection.search(query, 'similarity')
# res_mmr = screenplays_vector_store_collection.search(query, 'mmr')
# sim_res = screenplays_vector_store_collection.similarity_search(query, k=2)
mmr_res = screenplays_vector_store_collection.max_marginal_relevance_search(query, k=2, fetch_k=20)

# print(res_sim)
# print(res_mmr)
# print(sim_res)
# print(mmr_res)

print("\n\nmmr_res:\n")
for res in mmr_res:
    pprint(res.model_dump_json())    

# print("doc count: {}".format(screenplays_vector_store_collection._collection.count()))
# print("first doc: {}".format(screenplays_vector_store_collection._collection.peek(1)))

