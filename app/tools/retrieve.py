from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
from langfuse import observe

from app.tools.traced_embeddings import TracedEmbeddings

has_anything_loaded = load_dotenv()

if not has_anything_loaded:
    raise ValueError("No .env file found")

class SceneRetriever:

    def __init__(self) -> None:
        self.embedding_model = TracedEmbeddings()
        self.screenplays_vector_store_collection = Chroma(
            collection_name="screenplays", 
            embedding_function=self.embedding_model, 
            persist_directory='app/chroma')

    @observe(name="querying_in_scene_retriever", as_type="retriever")
    def query(self, query_text: str, k: int = 5, fetch_k: int = 20) -> list[Document]:
        # query = "how can I create a suspense and tension?"
        print("query_text from llm: {}\n".format(query_text))
        
        mmr_res = self.screenplays_vector_store_collection.max_marginal_relevance_search(query_text, k=k, fetch_k=fetch_k)

        # put the original text inside the page_content 
        for res in mmr_res:
            original_text = res.metadata.get('original_text', '')
            res.page_content = original_text

            if original_text == '':
                mmr_res.remove(res)
            
        return mmr_res

        # print("doc count: {}".format(screenplays_vector_store_collection._collection.count()))
        # print("first doc: {}".format(screenplays_vector_store_collection._collection.peek(1)))

