import os
import subprocess
from typing import List, Tuple

import gradio as gr

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory

from langsmith import traceable

# ------------------------- CONFIG ------------------------- #

PDF_PATH = "movies_trivia.pdf"
CHROMA_DIR = "chroma_db"
CHROMA_COLLECTION = "movie_trivia"

DEFAULT_MODEL = "mistral"  
qa_chain: RetrievalQA | None = None
chat_history = ChatMessageHistory()  
vectorstore: Chroma | None = None
AVAILABLE_MODELS: List[str] = []


# ------------------------- HELPERS ------------------------- #

def tracing_enabled() -> bool:
    return (
        os.getenv("LANGCHAIN_TRACING_V2") == "true"
        or os.getenv("LANGSMITH_TRACING") == "true"
    )


def get_ollama_models() -> List[str]:
    try:
        out = subprocess.check_output(["ollama", "list"], text=True)
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        if len(lines) <= 1:
            return []
        names: List[str] = []
        # first line is header: "NAME  SIZE  MODIFIED"
        for line in lines[1:]:
            parts = line.split()
            if parts:
                names.append(parts[0])
        return names
    except Exception:
        return []


# ------------------------- PDF & VECTOR STORE ------------------------- #

@traceable(
    name="Load and split PDF",
    run_type="tool",
    tags=["moviebot", "setup"],
    metadata={"stage": "load_pdf"},
)
def load_and_split_pdf(pdf_path: str):

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = CharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=20,
        separator="\n",
    )
    return splitter.split_documents(docs)


@traceable(
    name="Create Chroma Vectorstore",
    run_type="tool",
    tags=["moviebot", "setup", "chroma"],
    metadata={"stage": "vectorstore"},
)
def create_vectorstore(chunks) -> Chroma:
    embeddings = OllamaEmbeddings(model="granite-embedding:latest")

    if os.path.exists(CHROMA_DIR):
        vectorstore = Chroma(
            collection_name=CHROMA_COLLECTION,
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
        )
    else:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=CHROMA_COLLECTION,
            persist_directory=CHROMA_DIR,
        )
        vectorstore.persist()

    return vectorstore


# ------------------------- LLM & RETRIEVAL QA ------------------------- #

@traceable(
    name="Build QA Chain",
    run_type="chain",
    tags=["moviebot", "setup", "qa_chain"],
)
def build_qa_chain(vector: Chroma, model_name: str) -> RetrievalQA:
    
    llm = ChatOllama(
        model=model_name,
        temperature=0.2,
        streaming=False,  
    )

    prompt_template = """
You are MovieBot, a friendly and knowledgeable movie trivia expert.
Use ONLY the provided context (which comes from a movie trivia PDF)
to answer the question. If the answer is not in the context, say
you don't know.

Context:
{context}

Question:
{question}

Answer (concise but helpful):
"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template.strip(),
    )

    retriever = vector.as_retriever(search_kwargs={"k": 4})

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return qa


def format_history(history: ChatMessageHistory) -> str:
    
    lines: List[str] = []
    for msg in history.messages:
        role = "User" if msg.type == "human" else "Assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


# ------------------------- GRADIO CALLBACKS ------------------------- #

@traceable(
    name="Answer Question",
    run_type="chain",
    tags=["moviebot", "chat"],
    metadata={"component": "answer_question"},
)
def answer_question(
    user_message: str, chat_ui_history: list[list[str]]
) -> Tuple[list[list[str]], str]:
    """
    Handle a user message:
    - update ChatMessageHistory
    - feed combined history+question into RetrievalQA
    - return updated chat UI history and clear input box
    """
    global qa_chain, chat_history

    if qa_chain is None:
        chat_ui_history.append(
            ["System", "QA chain not initialized. Try restarting the app."]
        )
        return chat_ui_history, ""

    if not user_message or user_message.strip() == "":
        chat_ui_history.append(
            ["MovieBot", "Please type a movie-related question."]
        )
        return chat_ui_history, ""

    
    chat_history.add_user_message(user_message)

    history_str = format_history(chat_history)
    if history_str:
        combined_question = (
            f"Conversation so far:\n{history_str}\n\n"
            f"Latest user question: {user_message}"
        )
    else:
        combined_question = user_message

    try:
        # This LangChain chain call will be auto-traced when tracing is enabled
        result = qa_chain.invoke({"query": combined_question})
        answer = result["result"]
    except Exception as e:
        answer = (
            "⚠️ I had a problem talking to the local Ollama model.\n\n"
            f"Error: `{e}`\n\n"
            "Check that Ollama is running and the selected model works from the CLI."
        )

    chat_history.add_ai_message(answer)
    chat_ui_history.append([user_message, answer])
    return chat_ui_history, ""  # clear textbox


def clear_history() -> list[list[str]]:
    global chat_history
    chat_history = ChatMessageHistory()
    return []


@traceable(
    name="Change Model",
    run_type="tool",
    tags=["moviebot", "model"],
)
def change_model(selected_model: str) -> str:
    global qa_chain, vectorstore

    if vectorstore is None:
        return "❌ Vector store not ready yet. Try again in a moment."

    try:
        qa_chain = build_qa_chain(vectorstore, selected_model)
        # also reset history when changing model (optional)
        clear_history()
        return f"✅ Switched to model: `{selected_model}`"
    except Exception as e:
        return (
            f"⚠️ Failed to switch model to `{selected_model}`.\n\n"
            f"Error: `{e}`\n\n"
            "Make sure this model exists in Ollama and can respond from the CLI."
        )


# ------------------------- FANCY UI ------------------------- #

CUSTOM_CSS = """
body {
    background: radial-gradient(circle at top, #1e293b 0, #020617 55%, #000 100%);
}
#movie-title {
    text-align: center;
    font-size: 2.1rem;
    font-weight: 800;
    margin-top: 0.8rem;
    margin-bottom: 0.3rem;
    background: linear-gradient(120deg, #f97316, #facc15, #22c55e, #38bdf8);
    -webkit-background-clip: text;
    color: transparent;
}
#movie-subtitle {
    text-align: center;
    color: #e5e7eb;
    font-size: 0.95rem;
    margin-bottom: 1.0rem;
}
.gradio-container {
    max-width: 900px !important;
    margin: 0 auto !important;
}
.chatbot {
    border-radius: 18px !important;
    border: 1px solid rgba(148, 163, 184, 0.6) !important;
    background: rgba(15, 23, 42, 0.85) !important;
    backdrop-filter: blur(16px);
}
"""

def build_interface(models: List[str], default_model: str):
    with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Soft(primary_hue="orange")) as demo:
        gr.HTML('<div id="movie-title">🎬 MovieBot – Movie Trivia Chatbot</div>')
        gr.HTML(
            '<div id="movie-subtitle">'
            'Ask me anything about films! I use a PDF knowledge base + local Ollama models.'
            "</div>"
        )

        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=models,
                value=default_model,
                label="Ollama model",
                interactive=True,
            )
            model_status = gr.Markdown(
                value=f"✅ Using model: `{default_model}`",
                elem_id="model-status",
            )

        chatbot = gr.Chatbot(
            label="MovieBot Chat",
            elem_classes="chatbot",
            height=430,
            bubble_full_width=False,
        )

        with gr.Row():
            user_box = gr.Textbox(
                placeholder="Type your movie question here and press Enter…",
                show_label=False,
                lines=2,
                autofocus=True,
            )

        with gr.Row():
            send_btn = gr.Button("🎥 Ask MovieBot", variant="primary")
            clear_btn = gr.Button("🧹 Clear Conversation")

        with gr.Accordion("Example questions", open=False):
            gr.Markdown(
                "- *who directed Shawshank Redemption?*"
            )

        # Wiring: model change
        model_dropdown.change(
            fn=change_model,
            inputs=model_dropdown,
            outputs=model_status,
        )

        # Wiring: send button / enter
        send_btn.click(
            fn=answer_question,
            inputs=[user_box, chatbot],
            outputs=[chatbot, user_box],
        )
        user_box.submit(
            fn=answer_question,
            inputs=[user_box, chatbot],
            outputs=[chatbot, user_box],
        )
        clear_btn.click(
            fn=clear_history,
            inputs=None,
            outputs=chatbot,
        )

    return demo


# ------------------------- MAIN ------------------------- #

def main():
    global qa_chain, vectorstore, AVAILABLE_MODELS, DEFAULT_MODEL

    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(
            f"Could not find {PDF_PATH}. Put your PDF in this folder with that name."
        )

    print("Loading and splitting PDF...")
    chunks = load_and_split_pdf(PDF_PATH)

    print("Initializing Chroma vector store with granite embeddings...")
    vectorstore = create_vectorstore(chunks)

    print("Discovering available Ollama models...")
    AVAILABLE_MODELS = get_ollama_models()
    if not AVAILABLE_MODELS:
        print("⚠️ Could not detect models via `ollama list`; using fallback model list.")
        AVAILABLE_MODELS = [DEFAULT_MODEL]
    else:
        print("Found models:", ", ".join(AVAILABLE_MODELS))


    if DEFAULT_MODEL not in AVAILABLE_MODELS:
        DEFAULT_MODEL = AVAILABLE_MODELS[0]

    print(f"Building RetrievalQA chain with Ollama model: {DEFAULT_MODEL} ...")
    qa_chain = build_qa_chain(vectorstore, DEFAULT_MODEL)

    if tracing_enabled():
        print("✅ LangSmith / LangChain tracing is ENABLED.")
        print(f"Project: {os.getenv('LANGCHAIN_PROJECT') or os.getenv('LANGSMITH_PROJECT')}")
    else:
        print(
            "ℹ️ LangSmith tracing is DISABLED. "
            "Set LANGCHAIN_TRACING_V2='true' (and LANGCHAIN_API_KEY) to enable it."
        )

    print("Launching Gradio app at http://127.0.0.1:7860 ...")
    demo = build_interface(AVAILABLE_MODELS, DEFAULT_MODEL)
    demo.launch()


if __name__ == "__main__":
    main()
