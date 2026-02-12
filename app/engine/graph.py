"""
LangGraph State Graph — Orchestrates the chat assistant flow.
Nodes: classify → retrieve → summarize → answer
10-minute session context for follow-up questions.
"""
import uuid
import time
import logging
from typing import TypedDict, Optional, Literal
from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)

# ─── Session Store (10-minute TTL) ─────────────────────────────────
SESSION_TTL = 600  # 10 minutes in seconds
sessions: dict = {}


def get_session(session_id: str) -> Optional[dict]:
    """Retrieve session if it exists and hasn't expired."""
    if session_id in sessions:
        session = sessions[session_id]
        if time.time() - session["timestamp"] < SESSION_TTL:
            return session
        else:
            del sessions[session_id]
    return None


def save_session(session_id: str, sop_context: str, sop_title: str, sop_link: str, chat_history: list):
    """Save or update session with current context."""
    sessions[session_id] = {
        "sop_context": sop_context,
        "sop_title": sop_title,
        "sop_link": sop_link,
        "chat_history": chat_history[-10:],  # keep last 10 messages
        "timestamp": time.time()
    }


# ─── Graph State ───────────────────────────────────────────────────
class ChatState(TypedDict):
    message: str
    session_id: str
    is_followup: bool
    intent: Optional[dict]
    sop_results: Optional[list]
    sop_context: Optional[str]
    sop_title: Optional[str]
    sop_link: Optional[str]
    summary: Optional[str]
    steps: Optional[list]
    answer: Optional[str]
    chat_history: Optional[list]


# ─── Node Functions ────────────────────────────────────────────────
def classify_node(state: ChatState) -> ChatState:
    """Determine if this is a new query or a follow-up."""
    session = get_session(state["session_id"])
    if session:
        state["is_followup"] = True
        state["sop_context"] = session["sop_context"]
        state["sop_title"] = session["sop_title"]
        state["sop_link"] = session["sop_link"]
        state["chat_history"] = session["chat_history"]
        logger.info(f"Follow-up detected for session {state['session_id']}")
    else:
        state["is_followup"] = False
        state["chat_history"] = []
        logger.info("New query — starting fresh retrieval")
    return state


def retrieve_node(state: ChatState, nlp_engine) -> ChatState:
    """Use NLP Engine to retrieve relevant SOPs."""
    retrieval = nlp_engine.retrieve_sops(state["message"])
    state["intent"] = retrieval["intent"]
    state["sop_results"] = retrieval["results"]

    if retrieval["results"]:
        best = retrieval["results"][0]
        # Combine top results for richer context
        all_content = "\n\n".join([r["content"] for r in retrieval["results"][:3]])
        state["sop_context"] = all_content
        state["sop_title"] = best["title"]
        state["sop_link"] = best.get("link", "")
    else:
        state["sop_context"] = ""
        state["sop_title"] = ""
        state["sop_link"] = ""

    return state


def summarize_node(state: ChatState, nn_engine) -> ChatState:
    """Use NN Engine to summarize SOP and extract steps."""
    if state["sop_context"]:
        state["summary"] = nn_engine.summarize(state["sop_context"])
        state["steps"] = nn_engine.extract_steps(state["sop_context"])
        state["answer"] = state["summary"]
    else:
        state["summary"] = "No relevant SOP found for your query."
        state["steps"] = []
        state["answer"] = state["summary"]
    return state


def answer_node(state: ChatState, nn_engine) -> ChatState:
    """Use NN Engine to answer follow-up questions from context."""
    if state["sop_context"]:
        state["answer"] = nn_engine.answer_question(
            question=state["message"],
            context=state["sop_context"]
        )
    else:
        state["answer"] = "No SOP context available. Please ask a new question."
    return state


def save_node(state: ChatState) -> ChatState:
    """Save session context for future follow-ups."""
    history = state.get("chat_history", [])
    history.append({"role": "user", "content": state["message"]})
    history.append({"role": "assistant", "content": state.get("answer", "")})

    save_session(
        session_id=state["session_id"],
        sop_context=state.get("sop_context", ""),
        sop_title=state.get("sop_title", ""),
        sop_link=state.get("sop_link", ""),
        chat_history=history
    )
    state["chat_history"] = history
    return state


# ─── Route Logic ───────────────────────────────────────────────────
def route_after_classify(state: ChatState) -> Literal["retrieve", "answer"]:
    """Route to retrieve (new query) or answer (follow-up)."""
    return "answer" if state["is_followup"] else "retrieve"


# ─── Build Graph ───────────────────────────────────────────────────
def build_chat_graph(nlp_engine, nn_engine):
    """Build and compile the LangGraph StateGraph."""

    graph = StateGraph(ChatState)

    # Add nodes with engine dependencies injected
    graph.add_node("classify", classify_node)
    graph.add_node("retrieve", lambda s: retrieve_node(s, nlp_engine))
    graph.add_node("summarize", lambda s: summarize_node(s, nn_engine))
    graph.add_node("answer", lambda s: answer_node(s, nn_engine))
    graph.add_node("save", save_node)

    # Set entry point
    graph.set_entry_point("classify")

    # Edges
    graph.add_conditional_edges("classify", route_after_classify)
    graph.add_edge("retrieve", "summarize")
    graph.add_edge("summarize", "save")
    graph.add_edge("answer", "save")
    graph.add_edge("save", END)

    return graph.compile()
