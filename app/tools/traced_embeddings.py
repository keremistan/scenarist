from typing import Union

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
import os
import tiktoken
import langfuse

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

class TracedEmbeddings(Embeddings):
    def __init__(self):
        super().__init__()
        self.ollama_embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, temperature=0, base_url="http://{}:11434".format(os.getenv("OLLAMA_DOCKER_SERVICE")))
        self.encoding = tiktoken.get_encoding("o200k_base")
        self.langfuse_client = langfuse.get_client()

    def _count_tokens(self, input_text: Union[str, list[str]]) -> int:
        return sum([len(self.encoding.encode(text)) for text in input_text])

    def embed_query(self, text: str) -> list[float]:
        token_count = self._count_tokens(text)
        with self.langfuse_client.start_as_current_observation(
            name="embed_query",
            as_type="embedding",
            model=EMBEDDING_MODEL,
            input={"input_data": text},
            usage_details={
                "input":token_count,
                "total":token_count
            }
        ) as obs:
            embeddings = self.ollama_embeddings.embed_query(text)

        return embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        token_count = self._count_tokens(texts)
        with self.langfuse_client.start_as_current_observation(
            name="embed_documents",
            as_type="embedding",
            model=EMBEDDING_MODEL,
            input={"input_data": texts},
            usage_details={
                "input":token_count,
                "total":token_count
            }
        ) as obs:
            embeddings = self.ollama_embeddings.embed_documents(texts)

        return embeddings

