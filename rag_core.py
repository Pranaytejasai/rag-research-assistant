# Pranay Teja Chintakunta - 25079476
# MSc AI & ML - University of Limerick

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from dotenv import load_dotenv
import pypdf
import os

load_dotenv()

class RAGCore:

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_store = None
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        print("RAG system ready")

    def load_pdfs(self, pdf_paths):
        all_text = []
        for path in pdf_paths:
            try:
                reader = pypdf.PdfReader(path)
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        all_text.append({
                            "text": text,
                            "source": path,
                            "page": page_num + 1
                        })
                print(f"loaded {len(reader.pages)} pages from {path}")
            except Exception as e:
                print(f"could not load {path}: {e}")
        return all_text

    def build_index(self, pages):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = [
            Document(
                page_content=p["text"],
                metadata={"source": p["source"], "page": p["page"]}
            )
            for p in pages
        ]
        chunks = splitter.split_documents(docs)
        print(f"created {len(chunks)} chunks")
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="./my_database"
        )
        print("database saved successfully")

    def ask(self, question):
        if self.vector_store is None:
            return {"answer": "please upload documents first", "citations": []}

        prompt = ChatPromptTemplate.from_template("""
        Answer the question using only the context provided.
        If the answer is not in the context, say you dont know.
        Context: {context}
        Question: {question}
        Answer:
        """)

        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        answer = chain.invoke(question)
        source_docs = retriever.invoke(question)
        citations = []
        for i, doc in enumerate(source_docs):
            file_name = doc.metadata.get("source", "unknown")
            page_num = doc.metadata.get("page", "unknown")
            citations.append(f"[{i+1}] {file_name} - page {page_num}")
        return {"answer": answer, "citations": citations}


if __name__ == "__main__":
    rag = RAGCore()
    print("test passed!")