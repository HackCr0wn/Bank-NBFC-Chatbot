import os
from typing import List, Literal, TypedDict
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_groq import ChatGroq
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
load_dotenv()
from memory import AgentMemory

llm = ChatNVIDIA(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    temperature=0,
    max_tokens=2048
)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

memory_llm = ChatGroq(
    model=os.getenv("memory_model"),
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

memory = AgentMemory(
    embeddings=embeddings,
    summarizer_llm=memory_llm,
    consolidation_threshold=5
)

class AgentState(TypedDict, total=False):
    question: str
    documents: List[Document]
    answer: str
    needs_retrieval: bool
    documents_relevant: bool
    semantic_memory: List[Document]
    episodic_memory: List[Document]

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pdf_dir = os.path.join(base_dir, "data", "pdf")

dir_loader = DirectoryLoader(
    pdf_dir,
    glob="**/*.pdf",
    loader_cls=PyMuPDFLoader,
    show_progress=False
)

raw_documents = dir_loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

documents = text_splitter.split_documents(raw_documents)

cleaned_documents = [
    doc for doc in documents
    if doc.page_content.strip()
]

vectorstore = FAISS.from_documents(
    cleaned_documents,
    embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

class RetrievalDecision(BaseModel):
    needs_retrieval: bool = Field(
        description="Whether the question requires knowledge base retrieval"
    )
    

def decide_retrieval(state: AgentState) -> AgentState:

    question = state["question"]

    prompt = f"""
        You are a retrieval decision evaluator.

        Decide whether the user's question requires information
        from the knowledge base.

        Return YES for:
        - Banking and NBFC-related questions.
        - RBI regulations, policies and guidelines.
        - Financial definitions and domain-specific information.
        - Questions requiring uploaded documents.

        Return NO for:
        - Greetings and casual conversations.
        - General small talk.
        - Simple conversational responses.

        Question: {question}

        Return only YES or NO.
    """

    response = memory_llm.invoke(prompt)

    result = str(response.content).strip().upper()

    return {
        **state,
        "needs_retrieval": result.startswith("YES")
    }

def retrieve_memories(state: AgentState) -> AgentState:
    question = state["question"]

    memories = memory.retrieve_memories(
        query=question,
        k=3
    )

    return {
        **state,
        "semantic_memory": memories["semantic_memory"],
        "episodic_memory": memories["episodic_memory"]
    }

def retrieve_documents(state: AgentState) -> AgentState:
    question = state["question"]

    documents = retriever.invoke(question)

    return {
        **state,
        "documents": documents
    }

def grade_documents(state: AgentState) -> AgentState:

    question = state["question"]
    documents = state.get("documents", [])

    if not documents:
        return {
            **state,
            "documents_relevant": False
        }

    context = "\n\n".join(
        [doc.page_content for doc in documents]
    )

    prompt = f"""
        You are a document relevance evaluator.

        Determine whether the retrieved context contains useful
        information to answer the user's question.

        Question:
        {question}

        Retrieved Context:
        {context}

        Rules:
        - Return YES if the context contains information that
          meaningfully answers the question.
        - Return NO if the context is unrelated or insufficient.
        - Do not mark documents relevant merely because they
          contain similar keywords.
        -Return only YES or NO
        """

    response = memory_llm.invoke(prompt)

    result = response.content

    result = str(result).strip().upper()

    return {
        **state,
        "documents_relevant": result.startswith("YES")
    }

def generate_answer(state: AgentState) -> AgentState:

    question = state["question"]
    documents = state.get("documents", [])

    semantic_memories = state.get("semantic_memory", [])
    episodic_memories = state.get("episodic_memory", [])

    semantic_context = "\n".join(
        [doc.page_content for doc in semantic_memories]
    )

    episodic_context = "\n\n".join(
        [doc.page_content for doc in episodic_memories]
    )

    if documents:
        context = "\n\n".join(
            [doc.page_content for doc in documents]
        )
    else:
        context = "No relevant documents retrieved."

    prompt = f"""
        You are a helpful AI assistant with access to user memory
        and a knowledge base.

        USER MEMORY:

        Semantic Memory (long-term facts about the user):
        {semantic_context if semantic_context else "No relevant semantic memories found."}

        Episodic Memory (relevant previous conversations):
        {episodic_context if episodic_context else "No relevant previous conversations found."}

        KNOWLEDGE BASE:

        {context}

        CURRENT QUESTION:

        {question}

        INSTRUCTIONS:

        1. Use semantic memory to understand the user's profile,
           preferences, projects and long-term goals.
        2. Use episodic memory to understand relevant previous
           discussions and maintain conversational continuity.
        3. Use knowledge-base documents when they contain relevant
           information for the current question.
        4. Do not treat memory as verified external knowledge.
        5. Never invent user facts or previous conversations.
        6. If the knowledge base does not contain sufficient
           information for a domain-specific question, clearly
           acknowledge the limitation.
        7. Answer naturally, like a conversational assistant.
        8. Do not mention internal memory retrieval unless relevant
           to the user's question.

        Answer:
    """

    response = llm.invoke(prompt)

    content = response.content

    if isinstance(content, str):
        answer = content
    else:
        answer = str(content)

    answer = answer.strip()

    return {
        **state,
        "answer": answer
    }


def should_retrieve(
    state: AgentState
) -> Literal["retrieve", "generate"]:

    if state["needs_retrieval"]:
        return "retrieve"

    return "generate"


workflow = StateGraph(AgentState)

workflow.add_node("memory", retrieve_memories)
workflow.add_node("decide", decide_retrieval)
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("grade", grade_documents)
workflow.add_node("generate", generate_answer)

workflow.set_entry_point("memory")

workflow.add_conditional_edges(
    "decide",
    should_retrieve,
    {
        "retrieve": "retrieve",
        "generate": "generate"
    }
)

workflow.add_edge("memory", "decide")
workflow.add_edge("retrieve", "grade")
workflow.add_edge("grade", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

if __name__ == "__main__":
    print("Please ask question below")

    try:
        while True:

            question = input("You: ").strip()

            if not question:
                continue

            result = app.invoke({
                "question": question
            })

            answer = result.get("answer")

            if not isinstance(answer, str):
                raise TypeError("Agent did not return a valid string answer.") 

            print(f"\nAssistant: {result['answer']}\n")

            memory.save_turn(
                user_message=question,
                assistant_reply=answer
            )

    except KeyboardInterrupt:
        print("GOODBYE!!!")