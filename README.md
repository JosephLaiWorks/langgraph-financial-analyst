# Assignment 3: Autonomous Multi-Doc Financial Analyst

## 1. Project Overview
This project implements a multi-document financial analyst system for Apple and Tesla financial filings.

It includes:
- a **LangChain Legacy Agent** based on the ReAct paradigm
- a **LangGraph Agent** with intelligent routing, relevance grading, query rewriting, and answer generation
- separate vector databases for Apple and Tesla financial documents
- experiments on different embedding models and chunk sizes

The goal of this assignment is to compare LangChain and LangGraph, and to build a more controllable state-aware RAG system using LangGraph.

---

## 2. Features
- Supports questions about **Apple**, **Tesla**, **both**, or **none**
- Handles:
  - single-company questions
  - comparison questions
  - irrelevant questions
- Uses a **router** to choose the correct retrieval path
- Uses a **grader** to judge whether retrieved documents are sufficient
- Uses a **rewriter** to refine vague financial questions
- Returns **I don't know** when evidence is insufficient
- Includes a LangChain ReAct baseline for comparison

---

## 3. Project Structure
```bash id="7s75es"
.
├── data/
│   ├── FY24_Q4_Consolidated_Financial_Statements.pdf
│   └── tsla-20241231-gen.pdf
├── build_rag.py
├── config.py
├── langgraph_agent.py
├── README.md
└── report.pdf
```

## 4. Environment Setup
### 4.1 Create a virtual environment
```txt=
python -m venv .venv
```

### 4.2 Activate the environment
Windows
```txt=
.venv\Scripts\activate
```

macOS / Linux
```txt=
source .venv/bin/activate
```

### 4.3 Install dependencies
```txt=
pip install -r requirements.txt
```

## 5. API Configuration

Create a `.env` file in the project folder:
```txt=
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini
```

You may also switch to other providers supported in `config.py`, such as Google or Anthropic.

## 6. Build the Vector Databases

Place the Apple and Tesla PDF files in the data/ folder, then run:
```txt=
python build_rag.py
```

This script will:
1. load the PDF files
2. clean the text
3. split documents into chunks
4. generate embeddings
5. store the embeddings in Chroma vector databases

## 7. Run the LangGraph Agent
Start Python:
```txt=
python
```

Then run:
```txt=
from langgraph_agent import run_graph_agent

print(run_graph_agent("What was Apple's revenue in 2024?"))
print(run_graph_agent("How much did Apple spend on new tech in 2024?"))
print(run_graph_agent("Compare Apple and Tesla revenue in 2024."))
print(run_graph_agent("What is NVIDIA's revenue in 2024?"))
```

## 8. Run the LangChain Legacy Agent
Start Python:
```txt=
python
```
Then run:
```txt=
from langgraph_agent import run_legacy_agent

print(run_legacy_agent("What was Apple's revenue in 2024?"))
```
## 9. Embedding Model Experiments
Two embedding models were tested:
```txt=
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
sentence-transformers/all-MiniLM-L6-v2
```

To switch embedding models, edit LOCAL_EMBEDDING_MODEL in `config.py`, delete the chroma_db/ folder, and rebuild the vector databases.

Rebuild after changing the embedding model
* Windows cmd
```txt=
rmdir /s /q chroma_db
python build_rag.py
```

* PowerShell
```txt=
Remove-Item -Recurse -Force chroma_db
python build_rag.py
```

## 10. Chunk Size Experiments

Chunk sizes tested:
```txt=
1000
2000
4000
```

To test a different chunk size:

1. edit chunk_size in `build_rag.py`
2. delete `chroma_db/`
3. rebuild the vector databases

Windows cmd
```txt=
rmdir /s /q chroma_db
python build_rag.py
```

PowerShell
```txt=
Remove-Item -Recurse -Force chroma_db
python build_rag.py
```

## 11. Notes
* The final answer is generated in English only
* The system is designed to avoid hallucination
* If evidence is insufficient, the system answers: I don't know
* Apple and Tesla use separate retrievers backed by separate vector databases
* Comparison questions are handled through a dedicated dual-branch LangGraph workflow

## 12. Author
LAI, YU-SHENG

This project is based on the TA-provided sample code and further extended for Assignment 3.

My main contributions include:
* designing the LangGraph workflow
* implementing the Apple/Tesla comparison branch
* refining router, grader, rewriter, and generator behavior
* conducting embedding model experiments
* conducting chunk size experiments
* writing the report and project documentation

