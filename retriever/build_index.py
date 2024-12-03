print("Setting the constants")

import argparse
import os

import faiss
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm


print("Setting the constants")
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "--embedding_model_path", type=str, default="intfloat/multilingual-e5-large"
)
arg_parser.add_argument("--dimension_size", type=int, default=1024)
arg_parser.add_argument("--batch_size", type=int, default=262144)
args = arg_parser.parse_args()


print("Setting the constants")
INPUT_FILE_PATH = "retriever/en_wiki_corpus.jsonl"
RETRIEVAL_TOP_K = 5
RETRIEVAL_CHUNK_SIZE = 600
RETRIEVAL_CHUNK_OVERLAP = 30
FOLDER_PATH = "retriever/faiss_db"
TEMP_FOLDER_PATH = "retriever/faiss_db_temp"
INDEX_NAME = "faiss_index"


# def retrieve(retriever, query):
#     """
#     Retrieve relevant documents based on the query.

#     Args:
#         retriever: The retriever object used to invoke the query.
#         query (str): The query string to search for relevant documents.

#     Returns:
#         str: The concatenated content of the retrieved documents.
#     """
#     retrieved_str = ""
#     retrieved_docs = retriever.invoke(query)
#     for doc in retrieved_docs:
#         retrieved_str += doc.page_content
#         retrieved_str += "\n"
#     return retrieved_str.strip()


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


if os.path.exists(f"{TEMP_FOLDER_PATH}"):
    for temp_index in os.listdir(TEMP_FOLDER_PATH):
        if "_to_" in temp_index:
            temp_index_name = temp_index.split(".")[0]
            break
    temp_index_start_range = int(temp_index_name.split("_to_")[0])
    temp_index_end_range = int(temp_index_name.split("_to_")[1])

    print("Load existing FAISS database")
    db = FAISS.load_local(
        folder_path=TEMP_FOLDER_PATH, 
        index_name=temp_index_name,
        embeddings=hf_embeddings,
        allow_dangerous_deserialization=True
    )

    print("Adding documents to FAISS database")
    for_loop_start_range = temp_index_end_range
    for_loop_end_range = len(split_docs)
    for i in tqdm(
        range(for_loop_start_range, for_loop_end_range, args.batch_size),
        desc="Adding documents to database",
    ):
        temp_docs = split_docs[i : i + args.batch_size]
        db.add_documents(temp_docs)
        temp_index_name = f"{temp_index_start_range}_to_{i + args.batch_size}"
        db.save_local(folder_path=TEMP_FOLDER_PATH, index_name=temp_index_name)

        prev_index_name = f"{temp_index_start_range}_to_{i}"
        if os.path.exists(f"{TEMP_FOLDER_PATH}/{prev_index_name}.faiss"):
            os.remove(f"{TEMP_FOLDER_PATH}/{prev_index_name}.faiss")
            os.remove(f"{TEMP_FOLDER_PATH}/{prev_index_name}.pkl")


else:
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
        temp_index_name = f"0_to_{i + args.batch_size}"
        db.save_local(folder_path=TEMP_FOLDER_PATH, index_name=temp_index_name)

        prev_index_name = f"0_to_{i}"
        if os.path.exists(f"{TEMP_FOLDER_PATH}/{prev_index_name}.faiss"):
            os.remove(f"{TEMP_FOLDER_PATH}/{prev_index_name}.faiss")
            os.remove(f"{TEMP_FOLDER_PATH}/{prev_index_name}.pkl")

print("Documents added")
torch.cuda.empty_cache()

print("Saving the database")
db.save_local(folder_path=FOLDER_PATH, index_name=INDEX_NAME)
print("Database saved")
