import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == "__main__":
    print("Hello from langchain-course!")

    print("Loading documents...")
    loader = TextLoader("/Users/purnendudas/Desktop/Learning and Development/Agentic AI/langchain-course/sample_text.txt")
    document = loader.load()

    print("Splitting documents...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"Split {len(texts)} texts")

    print("Embedding documents...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        output_dimensionality=1536
    )
    print(f"Embedded {len(texts)} texts")

    print("Creating vector store...")
    PineconeVectorStore.from_documents(index_name=os.getenv("INDEX_NAME"), embedding=embeddings, documents=texts)
    print(f"Created vector store")
    print ("Finished")