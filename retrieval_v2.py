"""
RAG Retrieval Pipeline V2 for Normalized OCR OUTPUT Chandra

Key features for our approach:
- Parent-child chunk retrieval (when child matches, can fetch parent for full context)
- Metadata filtering (year, month, report_type)
- Uses collection: normalized_chandra_v2
"""

import os
import logging
from typing import List, Dict, Any, Optional
from collections import deque
from langchain_core.documents import Document
import asyncio  
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, NVIDIARerank, ChatNVIDIA
from langchain_milvus import Milvus
from pymilvus import MilvusClient

# ============== CONFIGURATION ==============
class Config:
    # Milvus Settings - Point to our new collection
    MILVUS_URI = "http://localhost:19530"
    COLLECTION_NAME = "normalized_chandra_v2"

    # Models
    EMBEDDING_MODEL = "nvidia/llama-3.2-nv-embedqa-1b-v2"
    RERANKER_MODEL = "nvidia/llama-3.2-nv-rerankqa-1b-v2"
    LLM_MODEL = "mistralai/ministral-14b-instruct-2512"

    # API Key
    NIM_API_KEY = "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"

    # Retrieval Settings
    TOP_K_RETRIEVAL = 10  # Get more initially
    TOP_K_RERANK = 5      # Keep top 5 after reranking

    # Parent-Child Retrieval
    ENABLE_PARENT_RETRIEVAL = True  # Fetch parent context when needed

    # Chat History Settings
    MAX_HISTORY_TURNS = 5

    # Search Parameters
    SEARCH_PARAMS = {
        "ef": 512
    }


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============== CHAT HISTORY MANAGER ==============
class ChatHistory:
    """Manages conversation history for contextual queries"""

    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.history = deque(maxlen=max_turns)

    def add_turn(self, question: str, answer: str):
        self.history.append({"question": question, "answer": answer})

    def get_formatted_history(self) -> str:
        if not self.history:
            return "No previous conversation."
        formatted = []
        for turn in self.history:
            formatted.append(f"User: {turn['question']}")
            formatted.append(f"Assistant: {turn['answer']}")
        return "\n".join(formatted)

    def clear(self):
        self.history.clear()


# ============== COMPONENTS ==============
def get_embeddings():
    """Initialize NVIDIA Embeddings"""
    return NVIDIAEmbeddings(
        model=Config.EMBEDDING_MODEL,
        api_key=Config.NIM_API_KEY,
        truncate="END"
    )


def get_reranker():
    """Initialize NVIDIA Reranker"""
    return NVIDIARerank(
        top_n=Config.TOP_K_RERANK,
        model=Config.RERANKER_MODEL,
        api_key=Config.NIM_API_KEY,
        truncate="END"
    )


def get_llm():
    """Initialize LLM"""
    return ChatNVIDIA(
        model=Config.LLM_MODEL,
        temperature=0.2,
        max_completion_tokens=1024,
        api_key=Config.NIM_API_KEY
    )


# ============== VECTOR STORE ==============
async def get_vectorstore() -> Milvus:
    """Connect to Milvus collection"""
    embeddings = get_embeddings()
    vectorstore = Milvus(
        embedding_function=embeddings,
        collection_name=Config.COLLECTION_NAME,
        connection_args={"uri": Config.MILVUS_URI},
        auto_id=True,
    )
    return vectorstore


def get_milvus_client() -> MilvusClient:
    """Get raw Milvus client for advanced queries"""
    return MilvusClient(uri=Config.MILVUS_URI)


# ============== RETRIEVAL FUNCTIONS ==============
def retrieve_documents(
    query: str,
    vectorstore: Milvus,
    filters: Optional[Dict[str, Any]] = None,
    k: int = None
) -> List[Document]:
    """Retrieve documents with optional metadata filters"""

    k = k or Config.TOP_K_RETRIEVAL

    # Build filter expression
    filter_expr = None
    if filters:
        expr_parts = []
        if 'year' in filters and filters['year']:
            expr_parts.append(f"year == {filters['year']}")
        if 'month' in filters and filters['month']:
            expr_parts.append(f'month == "{filters["month"]}"')
        if 'report_type' in filters and filters['report_type']:
            expr_parts.append(f'report_type == "{filters["report_type"]}"')

        if expr_parts:
            filter_expr = " and ".join(expr_parts)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
            "param": Config.SEARCH_PARAMS,
            "expr": filter_expr
        }
    )

    docs = retriever.invoke(query)
    logger.info(f"Retrieved {len(docs)} documents")
    if filter_expr:
        logger.info(f"Applied filter: {filter_expr}")

    return docs


def retrieve_with_parent_context(
    query: str,
    vectorstore: Milvus,
    milvus_client: MilvusClient,
    k: int = 5
) -> List[Document]:
    """
    Retrieve documents and enhance with parent context.

    Strategy:
    1. Retrieve more candidates (k * 2)
    2. Rerank them
    3. Only fetch parent for TOP reranked documents that are child chunks
    """

    # Step 1: Retrieve more candidates
    child_docs = retrieve_documents(query, vectorstore, k=k * 2)

    if not child_docs:
        return []

    # Step 2: Rerank first (to get the best ones)
    reranked_docs = rerank_documents(query, child_docs)

    # Step 3: Take top k and identify which need parent context
    top_docs = reranked_docs[:k]

    docs_to_include = []
    parent_ids_to_fetch = set()

    for doc in top_docs:
        parent_id = doc.metadata.get('parent_id')
        is_parent = doc.metadata.get('is_parent', False)

        # If it's a child chunk with a parent, queue parent for fetch
        if parent_id and not is_parent:
            parent_ids_to_fetch.add(parent_id)
            docs_to_include.append(doc)
        else:
            docs_to_include.append(doc)

    # Step 4: Fetch parent chunks ONLY for top documents that need context
    if Config.ENABLE_PARENT_RETRIEVAL and parent_ids_to_fetch:
        for parent_id in parent_ids_to_fetch:
            try:
                results = milvus_client.query(
                    collection_name=Config.COLLECTION_NAME,
                    filter=f'chunk_id == "{parent_id}"',
                    output_fields=["chunk_id", "chunk_type", "text", "year", "month", "report_type"]
                )

                if results:
                    parent_doc = Document(
                        page_content=results[0].get('text', ''),
                        metadata={
                            'chunk_id': results[0].get('chunk_id'),
                            'chunk_type': 'parent_context',
                            'is_parent': True,
                            'year': results[0].get('year'),
                            'month': results[0].get('month'),
                            'report_type': results[0].get('report_type')
                        }
                    )
                    docs_to_include.append(parent_doc)
                    logger.info(f"Fetched parent context: {parent_id}")

            except Exception as e:
                logger.warning(f"Failed to fetch parent {parent_id}: {e}")

    return docs_to_include


def rerank_documents(query: str, documents: List[Document]) -> List[Document]:
    """Rerank retrieved documents"""
    if not documents:
        return []

    reranker = get_reranker()
    reranked_docs = reranker.compress_documents(query=query, documents=documents)
    logger.info(f"Reranked to top {len(reranked_docs)} documents")
    return list(reranked_docs)


def format_docs(docs: List[Document], include_metadata: bool = False) -> str:
    """Format documents for LLM context"""

    formatted = []
    for i, doc in enumerate(docs):
        chunk_type = doc.metadata.get('chunk_type', 'unknown')
        year = doc.metadata.get('year', 'N/A')
        month = doc.metadata.get('month', 'N/A')
        report_type = doc.metadata.get('report_type', 'N/A')

        header = f"[Document {i+1}]"
        if include_metadata:
            header += f" | Type: {chunk_type} | Year: {year} | Month: {month} | Report: {report_type}"

        formatted.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted)


# ============== RAG CHAIN ==============
def create_rag_chain():
    """Create RAG chain with proper prompts"""
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an AI assistant that answers questions based on the provided context
from piracy and armed robbery reports against ships.

STRICT RULES:
- Only use information explicitly present in the provided context.
- Never generate fictional incidents, vessel names, statistics, or dates.
- If information is missing, say so clearly.
- Answer in detail when asked for details, concisely for basic questions.

When answering:
1. If the exact answer exists → provide it with all relevant details.
2. If partially available → provide available details and state limitations.
3. If not available → state that the requested information is not found in the current dataset.

Context includes both specific incident details and full report documents.
Parent document context provides broader context, child chunks provide specific details."""),
        ("human", """Context:
{context}

Question: {question}

Answer:""")
    ])

    chain = prompt | llm | StrOutputParser()
    return chain


def generate_answer_with_history(
    question: str,
    rewritten_question: str,
    context: str,
    chat_history: "ChatHistory"
) -> str:
    """
    Generate answer with chat history awareness.
    """
    llm = get_llm()

    # Build prompt based on whether there's history
    if chat_history.history:
        # Use history-aware prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant that answers questions based on the provided context
and previous conversation history.

STRICT RULES:
- Only use information explicitly present in the provided context.
- Never generate fictional incidents, vessel names, statistics, or dates.
- If information is missing, say so clearly.
- Use previous conversation to understand context (e.g., "that ship" refers to previous mentions)

When answering:
1. If exact answer exists → provide with all relevant details.
2. If partially available → provide details and state limitations.
3. If not available → state that information is not in dataset."""),
            ("human", """Previous Conversation:
{chat_history}

Context:
{context}

Original Question: {original_question}
Rewritten Question: {rewritten_question}

Answer:""")
        ])

        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "chat_history": chat_history.get_formatted_history(),
            "context": context,
            "original_question": question,
            "rewritten_question": rewritten_question
        })
    else:
        # No history - use standard prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant that answers questions based on the provided context
from piracy and armed robbery reports against ships.

STRICT RULES:
- Only use information explicitly present in the provided context.
- Never generate fictional incidents, vessel names, statistics, or dates.
- If information is missing, say so clearly.

When answering:
1. If exact answer exists → provide with all relevant details.
2. If partially available → provide details and state limitations.
3. If not available → state that information is not in dataset."""),
            ("human", """Context:
{context}

Question: {question}

Answer:""")
        ])

        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "context": context,
            "question": question
        })

    return answer


# ============== MAIN QUERY FUNCTIONS ==============

def contextualize_question(question: str, chat_history: "ChatHistory") -> str:
    """
    Query rewriting: Reformulate question using chat history.

    If user asks "What about ships in April?" after asking about 2024,
    this rewrites to "What ships were in April 2024?"
    """
    if not chat_history.history:
        return question

    # Only rewrite if there's history
    llm = get_llm()

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a query rewriter.

    Your task:
    - If the follow-up question depends on chat history, rewrite it into a standalone question by ONLY resolving references (e.g., pronouns like "it", "that ship", etc.).
    - Do NOT add new details.
    - Do NOT expand the scope.
    - Do NOT infer or guess additional fields.
    - Keep the rewritten question as close as possible to the original wording.
    - If the question is already standalone, return it unchanged.
    - If the input is a greeting or not a real question (e.g., "hi", "hello"), return it unchanged.

    Return ONLY the final question text.
    """),
    ("human", """Chat History:  
    {chat_history}

    Follow-up Question: {question}
    
    Standalone Question:""")
    ])

    chain = contextualize_prompt | llm | StrOutputParser()

    try:
        rewritten = chain.invoke({
            "chat_history": chat_history.get_formatted_history(),
            "question": question
        })
        rewritten = rewritten.strip()
        logger.info(f"Original: {question} -> Rewritten: {rewritten}")
        return rewritten
    except Exception as e:
        logger.warning(f"Query rewrite failed: {e}, using original")
        return question


def query_rag(
    question: str,
    filters: Optional[Dict[str, Any]] = None,
    use_parent_context: bool = True,
    use_reranker: bool = True,
    return_docs: bool = False,
    chat_history: Optional["ChatHistory"] = None,
    use_query_rewrite: bool = True
) -> Dict[str, Any]:
    """
    Main query function for RAG with chat history support.

    Args:
        question: The question to ask
        filters: Optional metadata filters (year, month, report_type)
        use_parent_context: Whether to fetch parent documents for context
        use_reranker: Whether to use reranking
        return_docs: Whether to return the retrieved documents
        chat_history: Optional ChatHistory object for conversation context
        use_query_rewrite: Whether to rewrite question using history

    Returns:
        Dict with 'answer' and optionally 'documents'
    """
    logger.info(f"Query: {question}")
    logger.info(f"Filters: {filters}")
    logger.info(f"Chat history: {'Yes' if chat_history and chat_history.history else 'No'}")

    # Initialize chat history if not provided
    if chat_history is None:
        chat_history = ChatHistory(max_turns=Config.MAX_HISTORY_TURNS)

    # Step 1: Query rewriting (if history exists and enabled)
    original_question = question
    if use_query_rewrite and chat_history.history:
        question = contextualize_question(question, chat_history)

    # Get vectorstore
    vectorstore = asyncio.run(get_vectorstore())

    # Get raw client for parent retrieval
    milvus_client = get_milvus_client()

    # Step 2: Retrieve documents
    if use_parent_context:
        docs = retrieve_with_parent_context(question, vectorstore, milvus_client, k=Config.TOP_K_RETRIEVAL)
    else:
        docs = retrieve_documents(question, vectorstore, filters=filters)

    # Step 3: Rerank if enabled
    if use_reranker and docs:
        docs = rerank_documents(question, docs)

    # Step 4: Format context with history
    history_context = chat_history.get_formatted_history() if chat_history.history else None
    context = format_docs(docs, include_metadata=True)

    if history_context and history_context != "No previous conversation.":
        context = f"Previous Conversation:\n{history_context}\n\n---\n\nRelevant Documents:\n{context}"

    # Step 5: Generate answer with history-aware prompt
    answer = generate_answer_with_history(
        question=original_question,
        rewritten_question=question,
        context=context,
        chat_history=chat_history
    )

    # Update chat history
    chat_history.add_turn(original_question, answer)

    result = {"answer": answer}

    if return_docs:
        result["documents"] = docs

    return result

    if return_docs:
        result["documents"] = docs

    return result


def query_with_history(question: str, chat_history: ChatHistory) -> str:
    """Query with conversation history"""

    # Use history in context
    history_context = chat_history.get_formatted_history()

    # Get vectorstore
    vectorstore = get_vectorstore()
    milvus_client = get_milvus_client()

    # Retrieve with parent context
    docs = retrieve_with_parent_context(question, vectorstore, milvus_client)

    # Rerank
    docs = rerank_documents(question, docs)

    # Format with history
    context = f"Previous Conversation:\n{history_context}\n\n---\n\nRelevant Documents:\n{format_docs(docs, include_metadata=True)}"

    # Generate answer
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an AI assistant that answers questions based on provided context
and previous conversation history.

Rules:
- Use the conversation history to understand context
- Only use information from the provided documents
- If not found, say so clearly"""),
        ("human", "{context}\n\nQuestion: {question}\n\nAnswer:")
    ])

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    # Update history
    chat_history.add_turn(question, answer)

    return answer


# ============== HELPER FUNCTIONS ==============

def get_collection_stats():
    """Get collection statistics"""
    client = get_milvus_client()
    stats = client.get_collection_stats(Config.COLLECTION_NAME)
    return stats


def search_by_metadata(
    year: Optional[int] = None,
    month: Optional[str] = None,
    report_type: Optional[str] = None,
    limit: int = 10
) -> List[Dict]:
    """Search by metadata filters only (no semantic search)"""

    client = get_milvus_client()

    # Build filter
    expr_parts = []
    if year:
        expr_parts.append(f"year == {year}")
    if month:
        expr_parts.append(f'month == "{month}"')
    if report_type:
        expr_parts.append(f'report_type == "{report_type}"')

    if not expr_parts:
        logger.warning("No filters provided")
        return []

    filter_expr = " and ".join(expr_parts)

    results = client.query(
        collection_name=Config.COLLECTION_NAME,
        filter=filter_expr,
        limit=limit,
        output_fields=["chunk_id", "chunk_type", "text", "year", "month", "report_type"]
    )

    return results


# ============== MAIN INTERACTIVE ==============
def main():
    """Interactive RAG chat"""

    chat_history = ChatHistory(max_turns=Config.MAX_HISTORY_TURNS)

    # Show collection stats
    print("\n" + "="*60)
    print("RAG Retrieval Pipeline V2")
    print(f"Collection: {Config.COLLECTION_NAME}")
    print("="*60)

    try:
        stats = get_collection_stats()
        print(f"Total entities: {stats.get('row_count', 'N/A')}")
    except Exception as e:
        print(f"Could not get stats: {e}")

    print("\nCommands:")
    print("  - Type your question to chat")
    print("  - 'filter:year=2024' - Filter by year")
    print("  - 'filter:month=April' - Filter by month")
    print("  - 'filter:report_type=monthly' - Filter by report type")
    print("  - 'clear' - Reset chat history")
    print("  - 'quit' - Exit")
    print("="*60)

    # Current filters
    current_filters = {}

    while True:
        try:
            question = input("\nYou: ").strip()

            if not question:
                continue

            if question.lower() == 'quit':
                print("Goodbye!")
                break

            if question.lower() == 'clear':
                chat_history.clear()
                current_filters = {}
                print("Chat history and filters cleared.")
                continue

            # Handle filter commands
            if question.startswith('filter:'):
                # Parse filter
                parts = question[7:].split('=')
                if len(parts) == 2:
                    key, value = parts
                    if key == 'year':
                        current_filters['year'] = int(value)
                    else:
                        current_filters[key] = value
                    print(f"Filter applied: {current_filters}")
                continue

            # Regular query
            result = query_rag(
                question,
                filters=current_filters if current_filters else None,
                use_parent_context=True,
                use_reranker=True,      
                return_docs=True,chat_history=chat_history
            )

            print(f"\nAssistant: {result['answer']}")
            # print(f"\nDocuments: {result['documents']}")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()