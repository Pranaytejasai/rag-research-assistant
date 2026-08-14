# api.py - Multi-user FastAPI backend for RAG Research Assistant
from fastapi import FastAPI, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from multi_doc import MultiDocRAG
from arxiv_search import ArxivSearcher, PubMedSearcher, CrossRefSearcher, ResearchAlertSystem
import os
import shutil
import base64
import requests as _requests

app = FastAPI(title="RAG Research Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# shared searchers (these don't hold user data, so they can be shared)
arxiv_searcher = ArxivSearcher()
pubmed_searcher = PubMedSearcher()
crossref_searcher = CrossRefSearcher()
alert_system = ResearchAlertSystem()

BASE_UPLOAD_DIR = "./api_uploads"
os.makedirs(BASE_UPLOAD_DIR, exist_ok=True)

# ---------- SESSION MANAGEMENT ----------
# each session_id maps to its own rag instance + sources + upload folder
sessions = {}

def get_session(session_id):
    if not session_id:
        session_id = "default"
    if session_id not in sessions:
        user_dir = os.path.join(BASE_UPLOAD_DIR, session_id)
        os.makedirs(user_dir, exist_ok=True)
        sessions[session_id] = {
            "rag": MultiDocRAG(),
            "sources": {},
            "dir": user_dir
        }
    return sessions[session_id]


def make_verification(rag, content):
    try:
        v = rag.verify_against_papers(content)
        return {
            "score": v.get("score", 0), "level": v.get("level", ""),
            "claims_total": v.get("claims_total", 0), "claims_supported": v.get("claims_supported", 0),
            "assessment": v.get("assessment", "")
        }
    except Exception as e:
        print(f"verify error: {e}")
        return None


def rebuild_session(sess):
    sess["rag"] = MultiDocRAG()
    remaining = [os.path.join(sess["dir"], f) for f in os.listdir(sess["dir"])]
    if remaining:
        pages = sess["rag"].load_multiple_pdfs(remaining)
        if pages:
            sess["rag"].build_multi_index(pages)


# ---------- CORE ----------

@app.get("/")
def home():
    return {"status": "RAG API is running!", "active_sessions": len(sessions)}

@app.get("/papers")
def get_papers(x_session_id: str = Header(default="default")):
    sess = get_session(x_session_id)
    return {"papers": sess["rag"].get_loaded_papers()}

@app.get("/paper-sources")
def get_paper_sources(x_session_id: str = Header(default="default")):
    sess = get_session(x_session_id)
    return {"sources": sess["sources"]}

@app.post("/upload")
async def upload_paper(file: UploadFile = File(...), x_session_id: str = Header(default="default")):
    sess = get_session(x_session_id)
    file_path = os.path.join(sess["dir"], file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    pages = sess["rag"].load_multiple_pdfs([file_path])
    if pages:
        sess["rag"].build_multi_index(pages)
    return {"message": f"Uploaded {file.filename}", "papers": sess["rag"].get_loaded_papers()}

@app.post("/remove-paper")
async def remove_paper(data: dict, x_session_id: str = Header(default="default")):
    sess = get_session(x_session_id)
    filename = data.get("filename", "")
    path = os.path.join(sess["dir"], filename)
    if os.path.exists(path):
        os.remove(path)
    stem = os.path.splitext(filename)[0]
    sess["sources"].pop(stem, None)
    rebuild_session(sess)
    return {"papers": sess["rag"].get_loaded_papers()}

@app.post("/reset-library")
async def reset_library(x_session_id: str = Header(default="default")):
    sess = get_session(x_session_id)
    for f in os.listdir(sess["dir"]):
        os.remove(os.path.join(sess["dir"], f))
    sess["sources"].clear()
    sess["rag"] = MultiDocRAG()
    return {"papers": sess["rag"].get_loaded_papers()}

@app.post("/ask")
async def ask_question(data: dict, x_session_id: str = Header(default="default")):
    sess = get_session(x_session_id)
    rag = sess["rag"]
    question = data.get("question", "")
    if not question:
        return {"error": "No question provided"}
    result = rag.ask_across_papers(question)
    verification = None
    try:
        v = rag.verify_answer(question, result["answer"], result.get("source_docs", []))
        verification = {
            "score": v.get("score", 0), "level": v.get("level", ""),
            "claims_total": v.get("claims_total", 0), "claims_supported": v.get("claims_supported", 0),
            "assessment": v.get("assessment", "")
        }
    except Exception as e:
        print(f"verify error: {e}")
    return {
        "answer": result["answer"], "citations": result.get("citations", []),
        "papers_used": result.get("papers_used", []), "verification": verification
    }

@app.post("/compare")
async def compare(data: dict, x_session_id: str = Header(default="default")):
    sess = get_session(x_session_id)
    rag = sess["rag"]
    topic = data.get("topic", "")
    result = rag.compare_papers(topic)
    verify_text = result.get("answer", "") if isinstance(result, dict) else result
    return {"result": result, "verification": make_verification(rag, verify_text)}

@app.get("/summaries")
def summaries(x_session_id: str = Header(default="default")):
    sess = get_session(x_session_id)
    return {"result": sess["rag"].generate_summaries()}

@app.get("/contradictions")
def contradictions(x_session_id: str = Header(default="default")):
    sess = get_session(x_session_id)
    return {"result": sess["rag"].detect_contradictions()}

@app.get("/key-findings")
def key_findings(x_session_id: str = Header(default="default")):
    sess = get_session(x_session_id)
    rag = sess["rag"]
    result = rag.extract_key_findings()
    return {"result": result, "verification": make_verification(rag, result)}

@app.get("/gaps")
def gaps(x_session_id: str = Header(default="default")):
    sess = get_session(x_session_id)
    return {"result": sess["rag"].identify_research_gaps()}

@app.get("/hypotheses")
def hypotheses(x_session_id: str = Header(default="default")):
    sess = get_session(x_session_id)
    return {"result": sess["rag"].generate_hypotheses()}

@app.post("/literature-review")
async def literature_review(data: dict, x_session_id: str = Header(default="default")):
    sess = get_session(x_session_id)
    rag = sess["rag"]
    length = data.get("length", "medium")
    result = rag.generate_literature_review(length)
    return {"result": result, "verification": make_verification(rag, result)}

@app.get("/similarity")
def similarity(x_session_id: str = Header(default="default")):
    sess = get_session(x_session_id)
    return {"result": sess["rag"].calculate_similarity()}

@app.get("/mindmap")
def mindmap(x_session_id: str = Header(default="default")):
    sess = get_session(x_session_id)
    connections = sess["rag"].generate_mindmap_data()
    return {"connections": [[a, b] for a, b in connections]}

@app.get("/timeline")
def timeline(x_session_id: str = Header(default="default")):
    sess = get_session(x_session_id)
    return {"timeline": sess["rag"].build_timeline()}


# ---------- CITATIONS ----------

@app.post("/cite")
async def cite(data: dict):
    title = data.get("title", "")
    authors = data.get("authors", [])
    year = data.get("published", "n.d.")
    journal = data.get("journal", "")
    doi = data.get("doi", "")
    url = data.get("url", "")

    if isinstance(authors, list):
        author_str = ", ".join(authors)
        apa_authors = " & ".join(authors) if len(authors) <= 2 else authors[0] + " et al."
    else:
        author_str = str(authors)
        apa_authors = author_str

    citations = {}
    if doi:
        try:
            headers = {"Accept": "text/x-bibliography; style=apa"}
            r = _requests.get(f"https://doi.org/{doi}", headers=headers, timeout=10)
            if r.status_code == 200 and r.text.strip():
                citations["APA (official)"] = r.text.strip()
        except Exception as e:
            print(f"doi cite error: {e}")

    src = journal if journal else url
    citations["APA"] = f"{apa_authors} ({year}). {title}. {src}."
    citations["MLA"] = f'{author_str}. "{title}." {src}, {year}.'
    citations["Harvard"] = f"{author_str} ({year}) '{title}', {src}."
    if url:
        citations["BibTeX-style"] = f"@article{{{year}, title={{{title}}}, author={{{author_str}}}, year={{{year}}}, url={{{url}}}}}"

    return {"citations": citations}


# ---------- VOICE & MULTILINGUAL (shared, no user data) ----------

@app.post("/translate")
async def translate(data: dict, x_session_id: str = Header(default="default")):
    sess = get_session(x_session_id)
    result = sess["rag"].translate_text(data.get("text", ""), data.get("language", "English"))
    return {"result": result}

@app.post("/text-to-speech")
async def tts(data: dict, x_session_id: str = Header(default="default")):
    sess = get_session(x_session_id)
    audio_bytes = sess["rag"].text_to_speech(data.get("text", ""))
    if audio_bytes:
        return {"audio": base64.b64encode(audio_bytes).decode('utf-8')}
    return {"audio": None}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), x_session_id: str = Header(default="default")):
    sess = get_session(x_session_id)
    audio_bytes = await file.read()
    return {"text": sess["rag"].transcribe_audio(audio_bytes)}


# ---------- LIVE SEARCH (shared searchers) ----------

@app.post("/search-arxiv")
async def search_arxiv(data: dict):
    return {"results": arxiv_searcher.search_papers(
        data.get("query", ""), max_results=data.get("max_results", 5),
        from_year=data.get("from_year"), to_year=data.get("to_year"))}

@app.post("/search-pubmed")
async def search_pubmed(data: dict):
    return {"results": pubmed_searcher.search_papers(
        data.get("query", ""), max_results=data.get("max_results", 5),
        from_year=data.get("from_year"), to_year=data.get("to_year"))}

@app.post("/search-crossref")
async def search_crossref(data: dict):
    return {"results": crossref_searcher.search_papers(
        data.get("query", ""), max_results=data.get("max_results", 5),
        from_year=data.get("from_year"), to_year=data.get("to_year"))}


# ---------- LOAD SEARCHED PAPERS (per session) ----------

@app.post("/load-arxiv")
async def load_arxiv(data: dict, x_session_id: str = Header(default="default")):
    sess = get_session(x_session_id)
    pdf_url = data.get("pdf_url", "")
    arxiv_id = data.get("arxiv_id", "")
    path = arxiv_searcher.download_paper(pdf_url, arxiv_id, save_folder=sess["dir"])
    if path:
        stem = os.path.splitext(os.path.basename(path))[0]
        sess["sources"][stem] = f"https://arxiv.org/abs/{arxiv_id}"
        pages = sess["rag"].load_multiple_pdfs([path])
        if pages:
            sess["rag"].build_multi_index(pages)
    return {"message": "loaded", "papers": sess["rag"].get_loaded_papers()}

@app.post("/load-pubmed")
async def load_pubmed(data: dict, x_session_id: str = Header(default="default")):
    sess = get_session(x_session_id)
    path = pubmed_searcher.save_abstract_as_text(data, save_folder=sess["dir"])
    if path:
        stem = os.path.splitext(os.path.basename(path))[0]
        sess["sources"][stem] = data.get("pubmed_url", "")
        pages = sess["rag"].load_multiple_pdfs([path])
        if pages:
            sess["rag"].build_multi_index(pages)
    return {"message": "loaded", "papers": sess["rag"].get_loaded_papers()}

@app.post("/load-crossref")
async def load_crossref(data: dict, x_session_id: str = Header(default="default")):
    sess = get_session(x_session_id)
    path = crossref_searcher.save_metadata_as_text(data, save_folder=sess["dir"])
    if path:
        stem = os.path.splitext(os.path.basename(path))[0]
        sess["sources"][stem] = data.get("doi_url", "")
        pages = sess["rag"].load_multiple_pdfs([path])
        if pages:
            sess["rag"].build_multi_index(pages)
    return {"message": "loaded", "papers": sess["rag"].get_loaded_papers()}


# ---------- RESEARCH ALERTS ----------

@app.get("/alert-topics")
def alert_topics():
    return {"topics": alert_system.get_topics()}

@app.post("/alert-add")
async def alert_add(data: dict):
    added = alert_system.add_topic(data.get("topic", ""))
    return {"added": added, "topics": alert_system.get_topics()}

@app.post("/alert-check")
async def alert_check(data: dict):
    result = alert_system.check_for_new_papers(data.get("topic", ""))
    return {"new": result["new"], "all": result["all"]}

@app.post("/alert-email")
async def alert_email(data: dict):
    to_email = data.get("email", "")
    topic = data.get("topic", "")
    if not to_email or not topic:
        return {"success": False, "error": "Email and topic required"}
    result = alert_system.check_for_new_papers(topic)
    papers = result.get("all", [])
    return alert_system.send_email_alert(to_email, topic, papers)