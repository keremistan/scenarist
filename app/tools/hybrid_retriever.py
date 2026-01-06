from collections import defaultdict

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import CrossEncoder

class HybridRetriever:
    def __init__(self, chroma_client: Chroma, k: int = 5, fetch_k: int = 20):
        self.k = k
        self.fetch_k = fetch_k

        self.chroma_client = chroma_client
        self.all_documents_with_metadata = self.chroma_client.get()

        # use original scene text instead of the subtext/analysis of it when doing bm25 search
        self.all_original_scene_text = [metadata['original_text'] for metadata in self.all_documents_with_metadata['metadatas']]

        # create an index on them within BM25
        self.tokenized_documents = []
        for og_scene in self.all_original_scene_text:
            doc_tokenized = og_scene.split(" ")
            self.tokenized_documents.append(doc_tokenized)

        self.bm25_okapi = BM25Okapi(self.tokenized_documents)

    def forward(self, query: str):
        # query the chroma client
        vector_res = self.chroma_client.max_marginal_relevance_search(query)

        # query the bm25
        bm25_scores = np.flip(np.argsort(self.bm25_okapi.get_scores(query.split(" "))))[:self.k]
        bm25_vector_docs_ids = [self.all_documents_with_metadata['ids'][bm25_current_index] for bm25_current_index in bm25_scores]
        bm25_vector_docs = self.chroma_client.get_by_ids(bm25_vector_docs_ids)

        # merge results from both using rrf
        both_lists: list[Document] = [*bm25_vector_docs, *vector_res] #unpack both lists into a single one.
        scores = defaultdict(float)
        for ind, current_doc in enumerate(both_lists, 1):
            scores[current_doc.id] += 1 / (ind + 60)
        sorted_scores = sorted(scores, key=lambda x: x[1] ,reverse=True)

        # filter out the docs according to their ranking
        ranked_docs: list[Document] = []
        for current_score in sorted_scores:
            for current_doc in both_lists:
                if current_score == current_doc.id:
                    ranked_docs.append(current_doc)

        # re-rank using cross encoder
        rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
        rerank_scores = rerank_model.predict([(query, doc.page_content) for doc in ranked_docs])
        rerank_scores_indexes = np.argsort(rerank_scores)[::-1]
        reranked_docs = [ranked_docs[rerank_score_index] for rerank_score_index in rerank_scores_indexes]

        # return only as much as required
        return reranked_docs[:self.k]
