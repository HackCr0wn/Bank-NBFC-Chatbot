import uuid
import os
from datetime import datetime, timezone
from typing import List
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_chroma import Chroma

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
memory_dir = os.path.join(base_dir, "data", "memory")

semantic_dir = os.path.join(memory_dir, "semantic")
episodic_dir = os.path.join(memory_dir, "episodic")

consolidation_threshold = 5 

class SemanticFacts(BaseModel):
    facts: List[str] = Field(
        description=(
            "A list of durable facts about the user,"
            "including profile information, preferences,"
            "skills, projects, goals and long-term context."
        )
    )

class AgentMemory:
    def __init__(self, embeddings, summarizer_llm, consolidation_threshold=consolidation_threshold):
        self.embeddings = embeddings
        self.summarizer_llm = summarizer_llm
        self.consoldation_threshold = consolidation_threshold

        os.makedirs(semantic_dir, exist_ok=True)
        os.makedirs(episodic_dir, exist_ok=True)

        self.semantic_store = Chroma(
            collection_name="semantic_memory",
            embedding_function=self.embeddings,
            persist_directory=semantic_dir
        )

        self.episodic_store = Chroma(
            collection_name="episodic_memory",
            embedding_function=self.embeddings,
            persist_directory=episodic_dir
        )

        self.fact_extractor = self.summarizer_llm.with_structured_output(SemanticFacts)

    def save_turn(self, user_message: str, assistant_reply: str):
        timestamp = datetime.now(timezone.utc).isoformat()
        turn_id = str(uuid.uuid4())

        conversation = f"""
            User: {user_message}
            
            Assisant: {assistant_reply}"""

        episode = Document(
            page_content=conversation,
            metadata={
                "turn_id": turn_id,
                "timestamp": timestamp,
                "consolidated": False,
                "memory_type": "conversation" 
            }
        )

        self.episodic_store.add_documents(
            documents = [episode],
            ids = [turn_id]
        )

        print("\n[Memory] Conversation Saved.")

        self._check_colsoidation()

    def _check_colsoidation(self):
        pending = self.episodic_store._collection.get(
            where={"consolidated":False},
            include=["documents", "metadatas"]
        )

        pending_count = len(pending["ids"])

        print(f"[Memory] Pending conversations: {pending_count}/{self.consoldation_threshold}")

        if pending_count >= self.consoldation_threshold:

            print(f"\n[Memory] Starting consolidation...")

            self._consolidate_memories(pending)

    def _consolidate_memories(self, pending):

        conversations = pending["documents"]
        metadata = pending["metadatas"]
        episode_ids = pending["ids"]

        conversation_context = "\n\n".join(
            [
                f"Date: {meta['timestamp']}\n{conversation}"
                for conversation, meta in zip(conversations, metadata)
            ]
        )

        prompt = f"""
            You are a memory consolidation agent.

            Your task is to extract durable and useful facts about the user
            from the following conversation history.

            CONVERSATION HISTORY:
            {conversation_context}

            INSTRUCTIONS:

            1. Extract stable facts about the user.
            2. Identify user profile information.
            3. Identify long-term interests, skills and preferences.
            4. Identify ongoing projects and goals.
            5. Retain important constraints and decisions.
            6. Combine repeated information into one fact.
            7. Ignore temporary questions and casual conversations.
            8. Do not invent facts or make unsupported assumptions.
            9. Do not treat assistant-generated suggestions as facts
               about the user unless the user explicitly confirms them.
            10. Return an empty list if no durable facts are available.

            Return the extracted facts using the required structured output.
            """

        result = self.fact_extractor.invoke(prompt)

        facts = [
            fact.strip()
            for fact in result.facts
            if fact.strip()
        ]

        if facts:

            fact_documents = []

            consolidation_id = str(uuid.uuid4())

            for index, fact in enumerate(facts):

                document = Document(
                    page_content=fact,
                    metadata={
                        "memory_type": "durable_fact",
                        "consolidation_id": consolidation_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "source_turn_ids": ",".join(episode_ids)
                    }
                )

                fact_documents.append(document)

            fact_ids = [
                f"fact_{consolidation_id}_{index}"
                for index in range(len(fact_documents))
            ]

            self.semantic_store.add_documents(
                documents=fact_documents,
                ids=fact_ids
            )

            print(
                f"[Memory] {len(facts)} durable facts "
                f"stored in semantic memory."
            )

        else:

            print("[Memory] No durable facts found.")

        updated_metadata = []

        for meta in metadata:

            updated_meta = dict(meta)
            updated_meta["consolidated"] = True

            updated_metadata.append(updated_meta)

        self.episodic_store._collection.update(
            ids=episode_ids,
            metadatas=updated_metadata
        )

        print("[Memory] Consolidation completed.")


    def retrieve_semantic_memory(self, query, k=3):

        return self.semantic_store.similarity_search(
            query=query,
            k=k,
            filter={"memory_type": "durable_fact"}
        )


    def retrieve_episodic_memory(self, query, k=3):

        return self.episodic_store.similarity_search(
            query=query,
            k=k
        )


    def retrieve_memories(self, query, k=3):

        semantic_memories = self.retrieve_semantic_memory(
            query=query,
            k=k
        )

        episodic_memories = self.retrieve_episodic_memory(
            query=query,
            k=k
        )

        return {
            "semantic_memory": semantic_memories,
            "episodic_memory": episodic_memories
        }
