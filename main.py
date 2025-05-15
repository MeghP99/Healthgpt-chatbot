# main.py — Memory Chatbot 3004 🩺🤖 (Optimized Async Edition)
"""
Run:

1. Put OPENAI_API_KEY / PINECONE_API_KEY in a .env
2. pip install -r requirements.txt
3. streamlit run main.py
"""

from __future__ import annotations

import os
import asyncio
import concurrent.futures
from typing import List, Dict, Any, cast

import streamlit as st
from dotenv import load_dotenv
import yaml
from openai import OpenAI, AsyncOpenAI
from mem0 import Memory

# ── Set page config FIRST before any other Streamlit calls ──────────
st.set_page_config(page_title="🩺 Memory Chat", page_icon="🤖")

# ── Env ────────────────────────────────────────────────────────────
load_dotenv()
openai_client = OpenAI()
async_openai_client = AsyncOpenAI()

# Create a thread pool for parallel operations
executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)

# ── Load system prompt from YAML ───────────────────────────────────
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_system_prompt():
    with open("prompts.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)["system_prompt"]

SYSTEM_PROMPT = load_system_prompt()

# ── Memory backend (Pinecone serverless in this demo) ──────────────
@st.cache_resource  # Cache memory store connection
def initialize_memory_store():
    return Memory.from_config(
        {
            "vector_store": {
                "provider": "pinecone",
                "config": {
                    "collection_name": "my-personal-index",
                    "embedding_model_dims": 1536,
                    "serverless_config": {"cloud": "aws", "region": "us-east-1"},
                    "metric": "cosine",
                },
            }
        }
    )

memory_store = initialize_memory_store()

# ── Streamlit chrome ───────────────────────────────────────────────
st.markdown(
    """
    <style>
      .stChatMessage {border-radius: 18px !important;}
      .stChatMessage:is(.st-bc) > div:last-child {background:#f3f7ff;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session defaults ───────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = cast(List[Dict[str, str]], [])
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "username" not in st.session_state:
    st.session_state.username = None
if "memory_queue" not in st.session_state:
    st.session_state.memory_queue = []
if "persist_counter" not in st.session_state:
    st.session_state.persist_counter = 0

# ── Login gate ─────────────────────────────────────────────────────
if st.session_state.user_id is None:
    st.title("🔐 Welcome!")
    st.caption("Pick a username to start chatting.")
    username = st.text_input("Username")
    if st.button("Enter Chat") and username.strip():
        user_id = username.strip().lower()

        # Add the user's name as their first memory (if new)
        # Do this in the main thread where session state is guaranteed to be available
        memory_store.add(
            [{"role": "system", "content": f"User's name is {username}"}],
            user_id=user_id,
        )

        st.session_state.user_id = user_id
        st.session_state.username = username
        st.rerun()
    st.stop()

# ── Inject welcome message on fresh session ───────────────────────
if not st.session_state.history:
    st.session_state.history.append(
        {
            "role": "assistant",
            "content": f"Hey 👋 {st.session_state.username}! "
            "I'm HealthGPT. Tell me something about yourself!",
        }
    )

# ── Helpers ────────────────────────────────────────────────────────
async def async_llm_chat(user_msg: str, memories: List[str], history: List[Dict[str, str]]) -> str:
    """Async call to GPT-4o-mini and return a plain-text assistant reply.
    Takes history as a parameter rather than accessing session_state directly."""
    memory_blob = (
        "Known user memories:\n" + "\n".join(f"- {m}" for m in memories)
        if memories
        else "No prior memories for this user."
    )

    # Build messages array properly
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": memory_blob},
    ]
    
    # Add history messages directly to the messages array
    # Using passed history parameter instead of session_state
    for m in history[-20:]:
        messages.append({"role": m["role"], "content": m["content"]})
    
    # Add the current user message
    messages.append({"role": "user", "content": user_msg})

    try:
        response = await async_openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"LLM call failed: {e}")
        return "I'm having trouble processing your request right now. Please try again."

def llm_chat(user_msg: str, memories: List[str]) -> str:
    """Synchronous wrapper for async_llm_chat."""
    # Capture the history here in the main thread
    history = list(st.session_state.history)  # Make a copy to avoid thread issues
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(async_llm_chat(user_msg, memories, history))
        return result
    finally:
        loop.close()

def fetch_memories(query: str) -> List[str]:
    """Fetch memories synchronously."""
    try:
        # Capture user_id in the main thread
        user_id = st.session_state.user_id
        
        # Direct call in the main thread is simpler and safer
        res = memory_store.search(
            query=query,
            user_id=user_id,
            limit=3,
        )
        
        return [e["memory"] for e in res.get("results", [])]
    except Exception as exc:
        st.error(f"Memory search failed: {exc}")
        return []

def memory_engine(user_msg: str, assistant_msg: str) -> None:
    """Queue memory for batch storage."""
    # Capture user_id in the main thread
    user_id = st.session_state.user_id
    
    # Add to the memory queue
    st.session_state.memory_queue.append({
        "messages": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ],
        "user_id": user_id
    })
    
    # Process in batches of 5
    if len(st.session_state.memory_queue) >= 5:
        batch_store_memories()

def batch_store_memories():
    """Store batched memories."""
    if not st.session_state.memory_queue:
        return
    
    # Create a local copy of the queue to process
    queue_to_process = list(st.session_state.memory_queue)
    st.session_state.memory_queue.clear()
    
    # Process immediately in the main thread
    for item in queue_to_process:
        try:
            memory_store.add(item["messages"], user_id=item["user_id"])
        except Exception as e:
            st.error(f"Failed to store memory: {e}")

def process_chat_input(prompt: str):
    """Process chat input synchronously."""
    # Get memories
    retrieved = fetch_memories(prompt)
    
    # Get LLM response
    reply = llm_chat(prompt, retrieved)
    
    return reply, retrieved

# ── Sidebar controls ───────────────────────────────────────────────
st.sidebar.title(f"🧰 Controls — {st.session_state.username}")

if st.sidebar.button("🏠  Home / change user"):
    # Ensure any pending memories are stored before changing users
    batch_store_memories()
    st.session_state.user_id = None
    st.session_state.history.clear()
    st.rerun()

if st.sidebar.button("🗑️ Clear chat"):
    st.session_state.history.clear()
    st.rerun()

# ── Chat UI ────────────────────────────────────────────────────────
st.title("HealthGPT aka MasterDoc 🧠💬")

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Your message..."):
    # Show user bubble
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to history once
    st.session_state.history.append({"role": "user", "content": prompt})
    
    with st.spinner("🧠✨ MasterDoc is processing your request..."):
        # Process input and get reply
        reply, retrieved = process_chat_input(prompt)
        
        # Show assistant bubble
        with st.chat_message("assistant"):
            st.markdown(reply)
        
        # Add assistant reply to session history once
        st.session_state.history.append({"role": "assistant", "content": reply})
        
        # Add memories in the main thread
        memory_engine(prompt, reply)

# Ensure we store pending memories when the app closes or is idle
if st.session_state.memory_queue:
    batch_store_memories()