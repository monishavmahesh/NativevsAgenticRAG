# Agentic RAG School Assistant

## 1. Introduction

The Agentic RAG School Assistant is an educational AI system that uses Retrieval-Augmented Generation (RAG) and multiple specialized agents to provide subject-specific tutoring for Mathematics, English, and Environmental Studies (EVS). Each subject agent uses its own textbook PDF, vector database, and memory, while a meta-agent routes questions to the appropriate subject automatically.[1]

***

## 2. System Overview
text
📁 Project Structure
├── app.py                 # Streamlit web interface
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── agents/
│   ├── base_agent.py      # Core agent implementation
│   └── meta_agent.py      # Routing and multi-question handling
├── core/
│   ├── memory.py          # Chat memory management
│   ├── vectorstore.py     # FAISS vector store operations
│   └── loader.py          # PDF document loading
└── textbooks/             # PDF textbook storage
    ├── math.pdf
    ├── english.pdf
    └── evs.pdf


### 2.1 Core Capabilities

- Subject-specific agents for Math, English, and EVS.[1]
- Dynamic PDF upload per subject through the web UI (user-provided textbooks).[1]
- Automatic question routing via a meta-agent with embedding-based similarity and keyword overrides.[1]
- Retrieval-Augmented Generation using FAISS vector stores and local embeddings.[1]
- Persistent conversation memory per subject.[1]
- Math equation solving using SymPy.[1]
- Streamlit-based interactive interface with tabs for each subject and an Auto (Meta-Agent) tab.[1]
- Local LLM inference using Ollama (Gemma 2B + Nomic Embed Text).[1]

***

## 3. Architecture


### 3.1 Components

- **app.py** – Streamlit web interface, tabbed UI, PDF upload, agent initialization.[1]
- **config.py** – Model names, device settings, default textbook paths.[1]
- **agents/base_agent.py** – Core per-subject agent: PDF loading, vector retrieval, prompting, math solver.[1]
- **agents/meta_agent.py** – Meta-agent: subject detection, question splitting, routing.[1]
- **core/loader.py** – PDF loading using PyPDFLoader.[1]
- **core/vectorstore.py** – FAISS vector index building and loading, chunking logic.[1]
- **core/memory.py** – Persistent memory handling for chat history.[1]
- **db/** – Persisted FAISS indexes for each subject.[1]
- **textbooks/** – PDF storage for subject textbooks.[1]

### 3.2 Data Flow

1. User uploads a subject PDF in the UI.  
2. The system loads the PDF, splits it into text chunks, embeds them, and builds a FAISS index under `db/{subject}`.[1]
3. When the user asks a question:
   - For subject tabs: the corresponding BaseAgent handles retrieval and answer generation.[1]
   - For Auto tab: the MetaAgent selects a subject and instantiates a temporary BaseAgent.[1]
4. The BaseAgent retrieves relevant chunks, builds a prompt with context and history, and queries Gemma 2B via Ollama.[1]

***

## 4. Retrieval and RAG Pipeline

### 4.1 Document Processing

- PDFs are parsed using LangChain’s `PyPDFLoader`.[1]
- Text is split into overlapping chunks (default 400 characters with 100 overlap; Math may use larger chunks to keep problems intact).[1]
- Each chunk is embedded using `nomic-embed-text` via Ollama.[1]

### 4.2 Vector Store

- FAISS is used as the vector database.[1]
- Per-subject indexes are stored under `./db/{subject}/faiss_index`.[1]

### 4.3 Retrieval

- For each query, the system retrieves the top-k relevant chunks (default k=3; can be extended with MMR for better diversity).[1]
- Retrieved chunks form the “Textbook Context” section in the prompt.[1]

### 4.4 Answer Generation

- The prompt includes:
  - Strict rules to use only textbook context.
  - Recent chat history.
  - The latest student question.[1]
- The Gemma 2B LLM returns an answer tailored to elementary school students.[1]

***

## 5. Meta-Agent: Automatic Subject Routing

### 5.1 Subject Detection

- Uses `nomic-embed-text` embeddings for:
  - Predefined subject descriptors (Math, English, EVS).
  - User question.[1]
- Computes cosine similarity between the question vector and each subject vector.[1]
- Applies explicit keyword overrides (“math”, “solve”, “equation”, etc.) to force routing when clear.[1]

### 5.2 Multi-Question Handling

- Splits compound questions into sub-questions based on `?`, newlines, and “and”.[1]
- Routes each sub-question independently and combines the answers.[1]

### 5.3 Stateless Execution

- For each routed sub-question, a temporary BaseAgent with fresh memory is created while still using the same textbook/vector store.[1]

***

## 6. Installation and Setup

### 6.1 Prerequisites

- Python 3.8 or higher.[1]
- Ollama installed on the local machine.[1]
- Ollama models:
  - `gemma:2b`
  - `nomic-embed-text`[1]

### 6.2 Steps

1. Clone repository and change directory.[1]
2. Install Python dependencies using `pip install -r requirements.txt`.[1]
3. Pull required Ollama models with `ollama pull gemma:2b` and `ollama pull nomic-embed-text`.[1]
4. Run the app with `streamlit run app.py`.[1]
5. In the UI, upload Math, English, and EVS PDFs from the sidebar.[1]

***

## 7. Usage

### 7.1 Subject Tabs

- Select Math, English, or EVS tab.  
- Enter a question related to that subject.  
- The corresponding BaseAgent retrieves from its textbook and answers.[1]

### 7.2 Auto (Meta-Agent) Tab

- Ask any educational question.  
- The MetaAgent:
  - Detects the most relevant subject.
  - Routes the question to that subject’s BaseAgent.[1]

### 7.3 Typical Scenarios

- Homework help based on the exact textbook content.[1]
- Concept explanations using age-appropriate language.[1]
- Step-by-step math problem solving using symbolic computation and context.[1]

***

## 8. Technical Stack

- **LLM**: Ollama Gemma 2B (text generation).[1]
- **Embeddings**: Nomic Embed Text (semantic search).[1]
- **Vector Store**: FAISS.[1]
- **UI**: Streamlit.[1]
- **Math Engine**: SymPy.[1]
- **Document Processing**: LangChain + PyPDF.[1]

***

## 9. Strengths and Limitations

### 9.1 Advantages

- Strong subject specialization via separate agents.[1]
- Completely local: no external API usage, privacy-friendly.[1]
- Scalable to long textbooks (100–1000 pages).[1]
- Cost-effective (no per-token API costs).[1]

### 9.2 Limitations

- Requires sufficient local CPU/GPU and RAM.[1]
- Restricted to the quality and coverage of the input PDFs.[1]
- Single-user focused (Streamlit).[1]
- English-only and text-only (no image understanding).[1]

***

