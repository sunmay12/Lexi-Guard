from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

def build_vectordb(articles: list):
    """조문 리스트 -> 벡터DB 구축"""
    # 조문을 텍스트로 변환
    texts = [
        f"{a['조문번호']} {a['조문제목']}\n{a['조문내용']}"
        for a in articles
    ]
    
    # 청킹
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.create_documents(texts)
    
    # 벡터DB 저장
    db = Chroma.from_documents(
        chunks,
        OpenAIEmbeddings(),
        persist_directory="model/vectordb"
    )
    return db

def retrieve(db, query: str, k=3):
    """질문 -> 관련 조문 검색"""
    return db.similarity_search(query, k=k)