from collections import defaultdict

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:
    def __init__(self, chroma_client: Chroma, k: int = 5, fetch_k: int = 20):
        self.k = k
        self.fetch_k = fetch_k

        self.chroma_client = chroma_client
        self.all_documents_with_metadata = self.chroma_client.get()
        self.all_documents = self.all_documents_with_metadata['documents']

        # print "all_documents: {}".format(self.all_documents))
        # print "all_documents_with_metadata: {}".format(self.all_documents_with_metadata))

        # get all documents and create an index on them
        self.tokenized_documents = []
        for doc in self.all_documents:
            # print("current doc: {}".format(doc))
            doc_tokenized = doc.split(" ")
            # print("doc_tokenized: {}".format(doc_tokenized))

            self.tokenized_documents.append(doc_tokenized)

        # self.tokenized_documents = [doc.metadata.original_text for doc in all_documents]
        self.bm25_okapi = BM25Okapi(self.tokenized_documents)

    def forward(self, query: str):
        # query the chroma client
        vector_res = self.chroma_client.max_marginal_relevance_search(query)
        # print("vector_res: {}".format(vector_res))

        # query the bm25
        bm25_scores = np.flip(np.argsort(self.bm25_okapi.get_scores(query.split(" "))))[:self.k]
        bm25_vector_docs_ids = [self.all_documents_with_metadata['ids'][bm25_current_index] for bm25_current_index in bm25_scores]
        bm25_vector_docs = self.chroma_client.get_by_ids(bm25_vector_docs_ids)
        # print("bm25_res: {}".format(bm25_vector_docs))

        # merge results from both using rrf
        scores = defaultdict(float)
        both_lists: list[Document] = [*bm25_vector_docs, *vector_res] #unpack both lists into a single one.
        # print("both_lists: {}".format(both_lists))

        for ind, current_doc in enumerate(both_lists, 1):
            scores[current_doc.id] += 1 / (ind + 60)

        # print("scores: {}".format(scores))
        sorted_scores = sorted(scores, key=lambda x: x[1] ,reverse=True)
        # print("sorted_scores: {}".format(sorted_scores))

        ranked_docs = []
        for current_score in sorted_scores:
            for current_doc in both_lists:
                if current_score == current_doc.id:
                    ranked_docs.append(current_doc)


        # print("ranked_docs: {}".format(ranked_docs))

        return ranked_docs[:self.k]
