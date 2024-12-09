import numpy as np
import yaml
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Union


class Retriever:
    def __init__(
        self, 
        yaml_config_path: str = "config/en_config.yaml"
    ):
        with open(yaml_config_path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self.retrieval_top_k = self.config["Retriever"]["parameters"]["retrieval_top_k"]

        print("Retriever: Load documents")
        loader = JSONLoader(
            file_path=self.config["Retriever"]["input_file_path"],
            jq_schema=".text",
            text_content=False,
            json_lines=True,
        )

        print("Retriever: Split documents")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["Retriever"]["parameters"]["retrieval_chunk_size"],
            chunk_overlap=self.config["Retriever"]["parameters"]["retrieval_chunk_overlap"],
            length_function=len,
            is_separator_regex=False,
        )

        print("Retriever: Loading and splitting documents...")
        split_docs = loader.load_and_split(text_splitter)

        self.langchain_retriever = BM25Retriever.from_documents(split_docs, k=self.retrieval_top_k)
        print("Retriever: Initialized")


    def retrieve(
        self, 
        query: str, 
        return_type: str = "list"
    ) -> Union[List[str], str]:
        list_of_document = self.langchain_retriever.invoke(query)
        if return_type.lower() == "list":
            return [doc.page_content for doc in list_of_document]
        elif return_type.lower() in ["text", "str", "string"]:
            return "\n".join([doc.page_content for doc in list_of_document])


class HybridRetriever(Retriever):
    def __init__(
        self, 
        yaml_config_path: str = "config/en_config.yaml"
    ):
        with open(yaml_config_path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        print("HybridRetriever: Create a HuggingFaceEmbeddings object")
        self.hf_embeddings = HuggingFaceEmbeddings(
            model_name=self.config["HybridRetriever"]["embedding_model_path"],
            model_kwargs={"device": "cuda"},  # cuda, cpu
            encode_kwargs={"normalize_embeddings": True},
        )

        print("HybridRetriever: Load documents")
        loader = JSONLoader(
            file_path=self.config["HybridRetriever"]["input_file_path"],
            jq_schema=".text",
            text_content=False,
            json_lines=True,
        )

        print("HybridRetriever: Split documents")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["HybridRetriever"]["parameters"]["retrieval_chunk_size"],
            chunk_overlap=self.config["HybridRetriever"]["parameters"]["retrieval_chunk_overlap"],
            length_function=len,
            is_separator_regex=False,
        )

        print("HybridRetriever: Loading and splitting documents...")
        split_docs = loader.load_and_split(text_splitter)

        self.retrieval_top_k = self.config["HybridRetriever"]["parameters"]["retrieval_top_k"]
        self.reranking_top_k = self.config["HybridRetriever"]["parameters"]["reranking_top_k"]
        self.langchain_retriever = BM25Retriever.from_documents(split_docs, k=self.retrieval_top_k)
        print("HybridRetriever: Initialized")


    def _rerank_passages(
        self,
        query: str,
        passages_list: List[str],
        return_type: str = "list",
    ) -> List[str]:
        """
        Rerank passages for the given question text.

        Args:
        - question_text: str - question text
        - passages_list: List[str] - list of passages

        Returns:
        - reranked_passages_text or reranked_passages_list: Union[str, List[str]] - reranked passages text or reranked passages list
        """

        embedded_query = self.hf_embeddings.embed_query(query)
        embedded_documents = self.hf_embeddings.embed_documents(passages_list)

        embedded_query_np = np.array(embedded_query)
        embedded_query_reshaped = embedded_query_np.reshape(1, -1)
        cosine_similarities = cosine_similarity(
            embedded_documents, embedded_query_reshaped
        )

        top_k_indices = np.argsort(cosine_similarities, axis=0)[::-1][: self.reranking_top_k]
        reranked_passages_list = [
            passages_list[idx_array[0]] for idx_array in top_k_indices
        ]

        if return_type.lower() in ["text", "str", "string"]:
            reranked_passages_text = ""
            for passage in reranked_passages_list:
                reranked_passages_text += passage
                reranked_passages_text += "\n"
            reranked_passages_text = reranked_passages_text.strip()
            return reranked_passages_text
        elif return_type.lower() in ["list"]:
            return reranked_passages_list


    def retrieve(
        self, 
        query: str, 
        return_type: str = "list"
    ) -> Union[List[str], str]:
        assert return_type.lower() in ["list", "text", "str", "string"], "return_type should be either 'list' or 'text'"

        list_of_document = self.langchain_retriever.invoke(query)
        passages_list = [doc.page_content for doc in list_of_document]

        return self._rerank_passages(query, passages_list, return_type=return_type)