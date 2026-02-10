"""
RAG Retrieval Pipeline with NVIDIA NIM (Embedding, Reranker, LLM) + Milvus GPU
Includes Chat History Support (up to 5 turns)
"""

import os
import logging
from typing import List
from collections import deque

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, NVIDIARerank, ChatNVIDIA
from langchain_milvus import Milvus

# ============== CONFIGURATION ==============
class Config:
    # Milvus Settings
    MILVUS_URI = "http://localhost:19530"
    COLLECTION_NAME = "rag_documents"
    
    # NVIDIA NIM Settings (API Catalog)
    NVIDIA_API_KEY = "your-nvidia-api-key"
    
    # Models
    EMBEDDING_MODEL = "nvidia/llama-3.2-nv-embedqa-1b-v2"
    RERANKER_MODEL = "nvidia/llama-3.2-nv-rerankqa-1b-v2"
    LLM_MODEL = "nvidia/llama-3.1-nemotron-70b-instruct"
    
    # Self-Hosted NIM URLs (uncomment if using local NIMs)
    # EMBEDDING_BASE_URL = "http://localhost:8080/v1"
    # RERANKER_BASE_URL = "http://localhost:2016/v1"
    # LLM_BASE_URL = "http://localhost:8000/v1"
    
    # Retrieval Settings
    TOP_K_RETRIEVAL = 10  # Initial retrieval count
    TOP_K_RERANK = 5      # After reranking
    
    # Chat History Settings
    MAX_HISTORY_TURNS = 5

        # HNSW Search Parameters
    SEARCH_PARAMS = {
        "ef": 256  # Search-time depth (64-512, higher = more accurate, slower)
    }
    
    # Alternative: For FLAT index (no search params needed)
    # SEARCH_PARAMS = {}


# ============== LOGGING ==============
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============== CHAT HISTORY MANAGER ==============
class ChatHistory:
    """Manages conversation history with a maximum of 5 turns."""
    
    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.history = deque(maxlen=max_turns)
    
    def add_turn(self, question: str, answer: str):
        """Add a Q&A turn to history."""
        self.history.append({"question": question, "answer": answer})
    
    def get_formatted_history(self) -> str:
        """Format history for prompt injection."""
        if not self.history:
            return "No previous conversation."
        
        formatted = []
        for turn in self.history:
            formatted.append(f"User: {turn['question']}")
            formatted.append(f"Assistant: {turn['answer']}")
        
        return "\n".join(formatted)
    
    def get_messages(self) -> list:
        """Get history as LangChain message objects."""
        messages = []
        for turn in self.history:
            messages.append(HumanMessage(content=turn["question"]))
            messages.append(AIMessage(content=turn["answer"]))
        return messages
    
    def clear(self):
        """Clear conversation history."""
        self.history.clear()


# ============== NVIDIA COMPONENTS ==============
def get_embeddings():
    """Initialize NVIDIA Embeddings."""
    # API Catalog
    return NVIDIAEmbeddings(
        model=Config.EMBEDDING_MODEL,
        truncate="END"
    )
    # Self-Hosted (uncomment to use)
    # return NVIDIAEmbeddings(base_url=Config.EMBEDDING_BASE_URL)


def get_reranker():
    """Initialize NVIDIA Reranker."""
    # API Catalog
    return NVIDIARerank(
        model=Config.RERANKER_MODEL,
        top_n=Config.TOP_K_RERANK,
        truncate="END"
    )
    # Self-Hosted (uncomment to use)
    # return NVIDIARerank(base_url=Config.RERANKER_BASE_URL, top_n=Config.TOP_K_RERANK)


def get_llm():
    """Initialize NVIDIA LLM."""
    # API Catalog
    return ChatNVIDIA(
        model=Config.LLM_MODEL,
        temperature=0.2,
        max_tokens=1024
    )
    # Self-Hosted (uncomment to use)
    # return ChatNVIDIA(base_url=Config.LLM_BASE_URL, model="meta/llama3-8b-instruct")


# ============== VECTOR STORE ==============
def get_vectorstore() -> Milvus:
    """Connect to existing Milvus collection with HNSW search params."""
    embeddings = get_embeddings()
    
    vectorstore = Milvus(
        embedding_function=embeddings,
        collection_name=Config.COLLECTION_NAME,
        connection_args={"uri": Config.MILVUS_URI},
    )
    return vectorstore


# ============== RETRIEVAL FUNCTIONS ==============
def retrieve_documents(query: str, vectorstore: Milvus) -> List[Document]:
    """Retrieve documents from Milvus with HNSW search params."""
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": Config.TOP_K_RETRIEVAL,
            "param": Config.SEARCH_PARAMS  # HNSW ef parameter
        }
    )
    docs = retriever.invoke(query)
    logger.info(f"Retrieved {len(docs)} documents")
    return docs


def rerank_documents(query: str, documents: List[Document]) -> List[Document]:
    """Rerank documents using NVIDIA Reranker."""
    if not documents:
        return []
    
    reranker = get_reranker()
    reranked_docs = reranker.compress_documents(query=query, documents=documents)
    logger.info(f"Reranked to top {len(reranked_docs)} documents")
    return list(reranked_docs)


def format_docs(docs: List[Document]) -> str:
    """Format documents for LLM context."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# ============== CONTEXTUALIZE QUESTION ==============
def contextualize_question(question: str, chat_history: ChatHistory) -> str:
    """Reformulate question using chat history for better retrieval."""
    if not chat_history.history:
        return question
    
    llm = get_llm()
    
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", """Given the chat history and a follow-up question, reformulate the question 
to be a standalone question that captures all necessary context.
If the question is already standalone, return it as-is.
Only return the reformulated question, nothing else."""),
        ("human", """Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:""")
    ])
    
    chain = contextualize_prompt | llm | StrOutputParser()
    
    standalone_question = chain.invoke({
        "chat_history": chat_history.get_formatted_history(),
        "question": question
    })
    
    logger.info(f"Contextualized question: {standalone_question}")
    return standalone_question.strip()


# ============== RAG CHAIN WITH HISTORY ==============
def create_rag_chain_with_history():
    """Create RAG chain that includes chat history."""
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions based on the provided context and chat history.
Use the context below to answer. If the answer is not in the context, say "I don't have enough information to answer this question."
Consider the chat history for continuity but prioritize the current context for facts.
Be concise and accurate."""),
        ("human", """Chat History:
{chat_history}

Context:
{context}

Question: {question}

Answer:""")
    ])
    
    chain = prompt | llm | StrOutputParser()
    return chain


# ============== MAIN QUERY FUNCTION ==============
def query_rag_with_history(question: str, chat_history: ChatHistory) -> str:
    """Full RAG pipeline with chat history: Contextualize -> Retrieve -> Rerank -> Generate."""
    logger.info(f"Original Query: {question}")
    
    # Step 1: Contextualize question using history
    standalone_question = contextualize_question(question, chat_history)
    
    # Step 2: Get vectorstore
    vectorstore = get_vectorstore()
    
    # Step 3: Retrieve using contextualized question
    retrieved_docs = retrieve_documents(standalone_question, vectorstore)
    
    # Step 4: Rerank
    reranked_docs = rerank_documents(standalone_question, retrieved_docs)
    
    # Step 5: Format context
    context = format_docs(reranked_docs)
    
    # Step 6: Generate answer with history
    chain = create_rag_chain_with_history()
    answer = chain.invoke({
        "chat_history": chat_history.get_formatted_history(),
        "context": context,
        "question": question  # Use original question for natural response
    })
    
    # Step 7: Update history
    chat_history.add_turn(question, answer)
    
    return answer


# ============== SINGLE QUERY WITHOUT HISTORY (OPTIONAL) ==============
def query_rag(question: str) -> str:
    """Single query RAG pipeline without history (for one-off questions)."""
    logger.info(f"Query: {question}")
    
    # Get vectorstore
    vectorstore = get_vectorstore()
    
    # Retrieve
    retrieved_docs = retrieve_documents(question, vectorstore)
    
    # Rerank
    reranked_docs = rerank_documents(question, retrieved_docs)
    
    # Format context
    context = format_docs(reranked_docs)
    
    # Generate answer
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions based on the provided context.
Use ONLY the context below to answer. If the answer is not in the context, say "I don't have enough information to answer this question."
Be concise and accurate."""),
        ("human", """Context:
{context}

Question: {question}

Answer:""")
    ])
    
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    
    return answer


# ============== MAIN ==============
def main():
    # Set API Key
    os.environ["NVIDIA_API_KEY"] = Config.NVIDIA_API_KEY
    
    # Initialize chat history (max 5 turns)
    chat_history = ChatHistory(max_turns=Config.MAX_HISTORY_TURNS)
    
    print("\n" + "="*60)
    print("RAG Chatbot with Chat History")
    print("="*60)
    print("Commands:")
    print("  - Type your question to chat")
    print("  - 'clear' - Reset chat history")
    print("  - 'history' - Show current chat history")
    print("  - 'quit' - Exit the chatbot")
    print("="*60)
    
    while True:
        try:
            question = input("\nYou: ").strip()
            
            if not question:
                continue
            
            # Handle commands
            if question.lower() == 'quit':
                print("Goodbye!")
                break
            
            if question.lower() == 'clear':
                chat_history.clear()
                print("Chat history cleared.")
                continue
            
            if question.lower() == 'history':
                print("\n--- Chat History ---")
                print(chat_history.get_formatted_history())
                print(f"--- [{len(chat_history.history)}/{chat_history.max_turns} turns] ---")
                continue
            
            # Process question with RAG
            answer = query_rag_with_history(question, chat_history)
            print(f"\nAssistant: {answer}")
            print(f"\n[History: {len(chat_history.history)}/{chat_history.max_turns} turns]")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"\nError occurred: {e}")
            print("Please try again.")


if __name__ == "__main__":
    main()