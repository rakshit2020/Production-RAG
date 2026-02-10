from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import os

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Connect to Milvus standalone (running on localhost:19530)
URI = "http://localhost:19530"

vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": URI},
    collection_name="test_collection",
    drop_old=True,  # Set True to reset collection
)

# Add some test documents
documents = [
    Document(page_content="Milvus is a vector database for AI applications.", metadata={"source": "doc1"}),
    Document(page_content="LangChain helps build LLM applications easily.", metadata={"source": "doc2"}),
    Document(page_content="RAG combines retrieval with generation.", metadata={"source": "doc3"}),
]

# Add documents to Milvus
vector_store.add_documents(documents=documents, ids=["1", "2", "3"])

# Test similarity search
results = vector_store.similarity_search("What is a vector database?", k=2)

print("Search Results:")
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")