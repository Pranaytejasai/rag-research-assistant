# A RAG-Based Academic Research Assistant with Retrieval Grounding and Real-Time Paper Search

**Author:** Pranay Teja Chintakunta
**Student ID:** 25079476
**Programme:** MSc Artificial Intelligence and Machine Learning, University of Limerick
**Supervisor:** Dr. Fazilat Hojaji

---

## Quick start

The project has two parts that run at the same time in two separate terminals: a Python backend
and a React frontend.

**Terminal 1 — backend (from the project root):**

```bash
python -m venv venv
venv\Scripts\activate            # Windows   (use: source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
```

**Terminal 2 — frontend (from the project root):**

```bash
cd frontend
npm install
npm run dev
```

Then open `http://localhost:5173` in a browser.

Before starting, create a `.env` file in the project root with your OpenAI key (see
[Configuration](#4-configuration-api-keys)). The application will not run without it.

If you only want to see the working application without any setup, a deployed version is available
(see [Live version](#12-live-version)).

---

## 1. Overview

This project is a Retrieval-Augmented Generation (RAG) research assistant. It answers questions
about uploaded academic papers, grounds every answer in the retrieved source text, and returns a
trust score that reports how much of the answer is supported by the papers. It also searches live
scholarly databases (ArXiv, PubMed, CrossRef), generates citations, and sends email alerts about
new papers on a topic.

The system has three parts:

1. A FastAPI backend that runs the RAG pipeline, the verification step, and the search and alert
   features.
2. A React (Vite) frontend that provides the user interface.
3. An evaluation notebook that benchmarks the system against a non-grounded baseline on SQuAD 2.0
   and PubMedQA.

A Streamlit version (`main.py`) is also included as a simple alternative interface.

---

## 2. Repository structure

```
.
├── api.py                 FastAPI backend (endpoints, sessions, verification)
├── multi_doc.py           Core RAG class (loading, chunking, retrieval, generation, verification)
├── arxiv_search.py        ArXiv, PubMed and CrossRef search + email alerts
├── main.py                Streamlit interface (alternative to the React app)
├── evaluation.ipynb       Evaluation notebook (SQuAD 2.0, PubMedQA, verification)
├── requirements.txt       Python dependencies
├── .env                   API keys (NOT included in submission — you create this, see section 4)
├── frontend/              React (Vite) frontend
│   ├── src/App.jsx        Main application component
│   ├── src/App.css        Styles
│   └── package.json       Frontend dependencies
└── README.md              This file
```

---

## 3. Requirements

- Python 3.11
- Node.js 18 or newer (for the frontend)
- An OpenAI API key (the system uses `gpt-4o-mini`, `text-embedding-3-small`, `whisper-1`
  and `tts-1`)
- A Resend API key (optional — only for the email alert feature)

---

## 4. Configuration (API keys)

The application reads its keys from a file named `.env` in the project root. This file is not
included in the submission for security. Create it before running:

```
OPENAI_API_KEY=your_openai_key_here
RESEND_API_KEY=your_resend_key_here
```

The `RESEND_API_KEY` is optional and only required for the email alert feature. Everything else
runs with just the OpenAI key.

---

## 5. Running the backend (step by step)

Open a terminal in the project root (the folder that contains `api.py`).

**Step 1 — create a virtual environment:**

```bash
python -m venv venv
```

**Step 2 — activate it:**

```bash
# Windows (PowerShell)
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

On Windows, if activation is blocked with a script-execution error, run this once in the same
PowerShell window, then activate again:

```bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

**Step 3 — install dependencies:**

```bash
pip install -r requirements.txt
```

If pip reports an "externally managed environment" error, add `--break-system-packages`:

```bash
pip install -r requirements.txt --break-system-packages
```

**Step 4 — start the backend:**

```bash
uvicorn api:app --reload --port 8000
```

When it is ready you will see `Application startup complete`. The backend runs at
`http://localhost:8000`. Open `http://localhost:8000/docs` to see the API documentation.

Common mistake: run this command from the project root, not from the `frontend` folder. Running it
from the wrong folder gives the error `Could not import module "api"`.

---

## 6. Running the frontend (step by step)

Open a second terminal (leave the backend running in the first one).

**Step 1 — enter the frontend folder:**

```bash
cd frontend
```

**Step 2 — install dependencies (first time only):**

```bash
npm install
```

**Step 3 — start the frontend:**

```bash
npm run dev
```

Open `http://localhost:5173` in a browser.

Common mistake: run `npm run dev` from inside the `frontend` folder, not the project root. Running
it from the root gives the error `Missing script: "dev"`.

The frontend talks to the backend at `http://localhost:8000` by default. To point it at a
different backend, set `VITE_API_URL` to the backend URL.

---

## 7. Running the Streamlit version (optional)

An alternative single-file interface is provided and uses the same core RAG code:

```bash
streamlit run main.py
```

This does not require the React frontend.

---

## 8. Using the application

1. Open the frontend at `http://localhost:5173`.
2. Upload one or more PDF papers from the sidebar.
3. Ask a question. The answer appears with a trust score and citations.
4. Use the other tabs to compare papers, summarise, find gaps, generate a literature review,
   search ArXiv/PubMed/CrossRef, or set up email alerts.

The first question after uploading may take a few seconds while the papers are indexed.

---

## 9. Running the evaluation

The evaluation is in `evaluation.ipynb`. Open it in Jupyter or VS Code, select the project's
virtual environment as the kernel, and run the cells in order from the top.

The notebook:

- loads SQuAD 2.0 and PubMedQA,
- runs the grounded system and a non-grounded baseline on 50 questions per dataset,
- reports accuracy, precision, recall, F1, faithfulness and hallucination counts,
- runs a separate evaluation of the verification/trust-score feature on grounded, partially
  grounded and hallucinated statements,
- saves the resulting charts and CSV summaries.

Running the full notebook makes a number of OpenAI API calls, so a valid `OPENAI_API_KEY` and some
API credit are required. Run the cells from the top in order, because later cells depend on
variables created by earlier ones.

---

## 10. Datasets

The evaluation datasets are downloaded automatically inside the notebook using the Hugging Face
`datasets` library, so no manual dataset upload or download link is needed:

- SQuAD 2.0 (`rajpurkar/squad_v2`)
- PubMedQA (`qiaojin/PubMedQA`)

Uploaded PDF papers are provided by the user at run time and are stored per user session.

---

## 11. Troubleshooting

| Symptom | Cause and fix |
|---------|---------------|
| `Could not import module "api"` | The backend command was run from the wrong folder. Run `uvicorn api:app --reload --port 8000` from the project root. |
| `Missing script: "dev"` | The frontend command was run from the root. Run `npm run dev` inside the `frontend` folder. |
| Script-execution error on Windows | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`, then activate the venv again. |
| `externally managed environment` on pip | Add `--break-system-packages` to the pip install command. |
| Backend starts but answers fail | The `.env` file is missing or the `OPENAI_API_KEY` is wrong. Check section 4. |
| ArXiv search is slow or empty | ArXiv rate limits requests. It times out gracefully; CrossRef and PubMed are faster alternatives. |
| Email alert fails | The `RESEND_API_KEY` is missing, or the recipient is not a verified address on the Resend free tier. |

---

## 12. Live version and links

A deployed version of the application is available, so the working system can be seen without any
local setup.

**Live application (frontend):** https://rag-research-assistant-murex.vercel.app
(open this link to use the deployed app)

**Backend API:** https://rag-research-assistant-d986.onrender.com
(a status message confirms it is running; `…/docs` shows the API endpoints)

**Source code (GitHub):** https://github.com/Pranaytejasai/rag-research-assistant

Note: the backend is hosted on a free tier that sleeps after a period of inactivity, so the first
request after an idle period can take around 30 to 60 seconds while it wakes up. After that it
responds normally.

### Datasets used in the evaluation

Both datasets are downloaded automatically by the evaluation notebook through the Hugging Face
`datasets` library, so no manual download is required. Their sources are:

- SQuAD 2.0 — https://huggingface.co/datasets/rajpurkar/squad_v2
- PubMedQA — https://huggingface.co/datasets/qiaojin/PubMedQA

### External services and APIs used

- OpenAI API (language model, embeddings, speech) — https://platform.openai.com
- Resend (email alerts over HTTPS) — https://resend.com
- ArXiv API — https://info.arxiv.org/help/api/index.html
- PubMed / NCBI E-utilities — https://www.ncbi.nlm.nih.gov/books/NBK25501/
- CrossRef API — https://www.crossref.org/documentation/retrieve-metadata/rest-api/

### Key libraries

- LangChain — https://www.langchain.com
- Chroma (vector store) — https://www.trychroma.com
- FastAPI — https://fastapi.tiangolo.com
- React with Vite — https://vitejs.dev

---

## 13. Notes

- The Streamlit app and the React app share the same backend logic in `multi_doc.py` and
  `arxiv_search.py`.
- For a smooth demonstration, use CrossRef and PubMed, which are faster than ArXiv.
- No API keys are included in this submission. The application will not run until a `.env` file
  with a valid `OPENAI_API_KEY` is created as described in section 4.
