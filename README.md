# movie_trivia_chatbot
Gen AI Deep Dive Batch 13 Assignment

Local RAG Chatbot Using Ollama, LangChain & ChromaDB

- **Ollama**: (Mistral / Llama3 / any installed model)
- **ChromaDB**: For vector search
- **Granite Embeddings**: (via Ollama)
- **LangChain**: For retrieval + orchestration
- **Gradio**
- **LangSmith**: For tracing

The bot answers questions strictly using **your PDF knowledge base (`movies_trivia.pdf`)**.  
If the answer is not present in the PDF, MovieBot will say **“I don't know.”**

---

## 🚀 Features

- Load & split a movie-trivia PDF
- Create and persist a Chroma vector database
- Granite embedding model for vector embeddings
- Local LLM inference with Ollama (Mistral by default)
- LangChain Retrieval-QA chain
- Chat history memory
- Model switcher for Ollama models
- Beautiful Gradio UI with custom theme
- Optional LangSmith tracing support

---

## 📦 Project Structure

```
project/
│── main.py
│── movies_trivia.pdf
│── chroma_db/
└── requirements.txt
└── .env
```

---

## 📦 Prerequisites

To set up the project, ensure the following environment variables are configured:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your_api_key
export LANGCHAIN_PROJECT="moviebot"
```

---

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/prakash888kp/movie_trivia_chatbot.git
   cd movie_trivia_chatbot
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv .venv
    .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your PDF knowledge base**:
   Place your `movies_trivia.pdf` file in the project root directory.

5. **Run the application**:
   ```bash
   python main.py
   ```

---

## 🧪 Screenshots
**Multi Model Selection**
![Screenshot 1](screenshots/screenshot5.png)
*Select any existing ollama downloaded ollama model*

**Retriving answers based on PDF context**
![Screenshot 2](screenshots/screenshot1.png)
*+ve flow of retriving output from pdf context with mistral:latest*

**Tracing langchain invoke calls**
![Screenshot 3](screenshots/screenshot2.png)
*LangChain trace of a QA chain using Mistral and Chroma vectorstore, built with StuffDocumentsChain and LLMChain. Shows prompt structure, input/output flow, and document retrieval setup.*

![Screenshot 4](screenshots/screenshot3.png)
*LangChain trace of a QA agent answering 'Who directed Shawshank Redemption?' using Chroma retrieval and Mistral.*

![Screenshot 5](screenshots/screenshot4.png)
*-ve flow of retriving output from pdf context with mistral:latest*

---

## 🌟 References

- [LangChain](https://github.com/hwchase17/langchain)
- [ChromaDB](https://github.com/chroma-core/chroma)
- [Ollama](https://ollama.ai/)
- [Gradio](https://gradio.app/)
- [LangSmith](https://docs.langchain.com/docs/langsmith)