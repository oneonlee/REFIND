import argparse
from typing import List

import faiss
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from tqdm import tqdm


print("Setting the constants")
INPUT_FILE_PATH = "retriever/en_wiki_corpus.jsonl"
RETRIEVAL_TOP_K = 5
RETRIEVAL_CHUNK_SIZE = 600
RETRIEVAL_CHUNK_OVERLAP = 30
FOLDER_PATH = "retriever/faiss_db"
INDEX_NAME = "faiss_index"


def split_list(lst, n):
    """Split a list into n roughly equal parts.

    Args:
        lst (list): The list to be split.
        n (int): The number of parts to split the list into.

    Returns:
        list of lists: A list containing n sublists.
    """
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def retrieve(retriever, query):
    """
    Retrieve relevant documents based on the query.

    Args:
        retriever: The retriever object used to invoke the query.
        query (str): The query string to search for relevant documents.

    Returns:
        str: The concatenated content of the retrieved documents.
    """
    retrieved_str = ""
    retrieved_docs = retriever.invoke(query)
    for doc in retrieved_docs:
        retrieved_str += doc.page_content
        retrieved_str += "\n"
    return retrieved_str.strip()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--embedding_model_path", type=str, default="intfloat/multilingual-e5-large"
    )
    arg_parser.add_argument("--dimension_size", type=int, default=1024)
    arg_parser.add_argument("--batch_size", type=int, default=32)
    args = arg_parser.parse_args()

    print("Create a HuggingFaceEmbeddings object")
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=args.embedding_model_path,
        model_kwargs={"device": "cuda"},  # cuda, cpu
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Load and split documents")
    loader = JSONLoader(
        file_path=INPUT_FILE_PATH,
        jq_schema=".text",
        text_content=False,
        json_lines=True,
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=RETRIEVAL_CHUNK_SIZE,
        chunk_overlap=RETRIEVAL_CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )

    print("Loading and splitting documents...")
    split_docs = loader.load_and_split(text_splitter)

    print("Create new FAISS database")
    db = FAISS(
        embedding_function=hf_embeddings,
        index=faiss.IndexFlatL2(args.dimension_size),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    print("Adding documents to FAISS database")
    for i in tqdm(
        range(0, len(split_docs), args.batch_size), desc="Adding documents to database"
    ):
        temp_docs = split_docs[i : i + args.batch_size]
        db.add_documents(temp_docs)
    print("Documents added")
    torch.cuda.empty_cache()

    print("Saving the database")
    db.save_local(folder_path=FOLDER_PATH, index_name=INDEX_NAME)
    print("Database saved")

    # 테스트
    print("Testing retrieval...")
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=args.embedding_model_path,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    loaded_db = FAISS.load_local(
        folder_path=FOLDER_PATH,
        index_name=INDEX_NAME,
        embeddings=hf_embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = loaded_db.as_retriever(search_kwargs={"k": RETRIEVAL_TOP_K})
    query = "What is the capital of France?"
    print(f"Query: {query}")
    print("Retrieved documents:")
    print(retriever.invoke(query))
