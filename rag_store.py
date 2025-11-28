# # rag_store.py
# from typing import Dict, List
# from langchain_community.vectorstores import FAISS
# from langchain_core.documents import Document
# from langchain_openai import OpenAIEmbeddings

# from config import OPENAI_API_KEY

# # Per-session FAISS stores
# SESSION_RAG_STORES: Dict[str, FAISS] = {}

# # Shared embeddings model
# _embeddings = OpenAIEmbeddings(
#     model="text-embedding-3-small",
#     api_key=OPENAI_API_KEY,
# )


# def upsert_session_docs(session_id: str, docs: List[Document]) -> None:
#     """
#     Create or extend a FAISS vector store for this session with new docs.
#     """
#     if session_id in SESSION_RAG_STORES:
#         store = SESSION_RAG_STORES[session_id]
#         store.add_documents(docs)
#         SESSION_RAG_STORES[session_id] = store
#     else:
#         store = FAISS.from_documents(docs, embedding=_embeddings)
#         SESSION_RAG_STORES[session_id] = store


# def get_session_retrieved_snippets(session_id: str, query: str, k: int = 2) -> List[Document]:
#     """
#     Retrieve a few relevant chunks (if any) for a given session and query.
#     """
#     if session_id not in SESSION_RAG_STORES:
#         return []

#     store = SESSION_RAG_STORES[session_id]
#     retriever = store.as_retriever(search_kwargs={"k": k})
#     return retriever.invoke(query)
