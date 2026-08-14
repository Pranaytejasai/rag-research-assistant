# Pranay Teja Chintakunta - 25079476
# MSc AI & ML - University of Limerick
# Module 2 - Multi Document Reasoning - Bug Fixed Version

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from dotenv import load_dotenv
import pypdf
import os
import shutil

load_dotenv()

class MultiDocRAG:

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_store = None
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        # using a set to prevent duplicate paper names
        self.loaded_papers = []
        self.paper_contents = {}
        self.paper_full_text = {}
        self.temp_files = []
        print("multi-document RAG ready")

    def clear_all(self):
        # properly clear everything including database
        self.loaded_papers = []
        self.paper_contents = {}
        self.vector_store = None
        # delete temp files from disk
        for f in self.temp_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except:
                pass
        self.temp_files = []
        # delete old database folder
        if os.path.exists("./multi_doc_database"):
            try:
                shutil.rmtree("./multi_doc_database")
                print("old database cleared")
            except Exception as e:
                print(f"could not clear database: {e}")
        # also clear the uploaded papers folder
        upload_dir = os.path.join(os.getcwd(), "uploaded_papers")
        if os.path.exists(upload_dir):
            try:
                shutil.rmtree(upload_dir)
                print("uploaded papers folder cleared")
            except Exception as e:
                print(f"could not clear uploads: {e}")
        print("all papers and database cleared")

    # Extract a clean display title for any paper source
    def extract_display_title(self, path, paper_name, text_sample):
        import re
        # CASE 1: PubMed/CrossRef text files have "Title:" line
        if text_sample.strip().startswith("Title:"):
            for line in text_sample.split("\n"):
                if line.strip().startswith("Title:"):
                    title = line.replace("Title:", "").strip()
                    short = title[:55] + "..." if len(title) > 55 else title
                    file_id = paper_name.replace(".txt", "")
                    return f"{short} ({file_id})"

        # CASE 2: ArXiv PDF - filename encodes the arxiv id
        arxiv_match = re.match(r"(\d{4}\.\d+)", paper_name)

        # CASE 3: For PDFs, ask the AI to pull the real title
        try:
            prompt = f"""
            Extract ONLY the title of this research paper from the text below.
            Reply with just the title, nothing else. Keep it under 15 words.
            Text: {text_sample[:1500]}
            Title:
            """
            response = self.llm.invoke(prompt)
            title = response.content.strip().strip('"')
            short = title[:55] + "..." if len(title) > 55 else title
            if arxiv_match:
                return f"{short} (arXiv:{arxiv_match.group(1)})"
            return short
        except Exception:
            # fallback to filename if extraction fails
            return paper_name

    def load_multiple_pdfs(self, pdf_paths):
        import re
        all_pages = []
        for path in pdf_paths:
            try:
                # clean the paper name
                paper_name = os.path.basename(path)
                paper_name = paper_name.replace("temp_", "")
                paper_name = re.sub(r"\s*\(\d+\)", "", paper_name)

                # skip if already loaded - prevents duplicates
                if paper_name in self.loaded_papers:
                    print(f"skipping duplicate: {paper_name}")
                    continue

                pages_loaded = 0
                full_text = ""

                # check the file type - PDF or text file
                if path.lower().endswith(".pdf"):
                    reader = pypdf.PdfReader(path)
                    # get first page text to extract the title
                    first_page_text = ""
                    if len(reader.pages) > 0:
                        first_page_text = reader.pages[0].extract_text() or ""
                    display_name = self.extract_display_title(
                        path, paper_name, first_page_text
                    )
                    for page_num, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if text and len(text.strip()) > 50:
                            all_pages.append({
                                "text": text,
                                "source": display_name,
                                "page": page_num + 1,
                                "full_path": path
                            })
                            full_text += text
                            pages_loaded += 1

                elif path.lower().endswith(".txt"):
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read()
                    if text and len(text.strip()) > 20:
                        display_name = self.extract_display_title(
                            path, paper_name, text
                        )
                        all_pages.append({
                            "text": text,
                            "source": display_name,
                            "page": 1,
                            "full_path": path
                        })
                        full_text += text
                        pages_loaded = 1

                if pages_loaded > 0:
                    self.loaded_papers.append(paper_name)
                    self.paper_contents[paper_name] = full_text[:3000]
                    # store the complete text separately for accurate similarity
                    self.paper_full_text[paper_name] = full_text
                    print(f"loaded {paper_name} - {pages_loaded} pages/sections")
                else:
                    print(f"no readable text in {paper_name}")

            except Exception as e:
                print(f"error loading {path}: {e}")

        print(f"total papers loaded: {len(self.loaded_papers)}")
        return all_pages

    def build_multi_index(self, pages):
        if not pages:
            print("no pages to index")
            return
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300
        )
        docs = [
            Document(
                page_content=p["text"],
                metadata={
                    "source": p["source"],
                    "page": p["page"],
                    "full_path": p["full_path"]
                }
            )
            for p in pages
        ]
        chunks = splitter.split_documents(docs)
        print(f"created {len(chunks)} chunks from {len(self.loaded_papers)} papers")
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="./multi_doc_database"
        )
        print("database saved successfully")

    def ask_across_papers(self, question):
        if self.vector_store is None:
            return {
                "answer": "no documents loaded yet - please upload papers first",
                "citations": [],
                "papers_used": [],
                "source_docs": []
            }

        # get author-year citations for the loaded papers
        citations_map = self.extract_citations()

        # build a citation guide for the prompt
        cite_guide = ""
        for name, cite in citations_map.items():
            cite_guide += f"- {name}: cite as {cite.get('intext', name)}\n"

        prompt = ChatPromptTemplate.from_template("""
        You are a research assistant analyzing multiple academic papers.
        Answer the question by synthesizing information from all provided sources.
        If different papers say different things, mention both perspectives.

        CITATION RULES:
        - Use academic in-text citations in author-year format, e.g. (Lewis et al., 2020).
        - Do NOT use "Document id" or long random identifiers or UUIDs.
        - Use these citation labels for each paper:
        {cite_guide}

        Context from multiple papers: {context}
        Question: {question}

        Write a detailed answer with proper in-text citations:
        """)

        retriever = self.vector_store.as_retriever(search_kwargs={"k": 8})
        source_docs = retriever.invoke(question)

        # build context text
        context_text = "\n\n".join([doc.page_content for doc in source_docs])

        # generate answer
        final_prompt = prompt.format(
            cite_guide=cite_guide,
            context=context_text,
            question=question
        )
        answer = self.llm.invoke(final_prompt).content

        # build the clean citation list (with pretty titles + pages)
        seen = set()
        citations = []
        for doc in source_docs:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "unknown")
            citation = f"{source} - page {page}"
            if citation not in seen:
                seen.add(citation)
                citations.append(citation)

        return {
            "answer": answer,
            "citations": citations,
            "papers_used": list(set([
                doc.metadata.get("source") for doc in source_docs
            ])),
            "source_docs": source_docs
        }
    # MODULE 4 - Hallucination Prevention / Answer Verification
    def verify_answer(self, question, answer, source_docs):
        # check if the answer is actually supported by the source documents
        if not source_docs:
            return {
                "score": 0,
                "level": "unverified",
                "claims_total": 0,
                "claims_supported": 0,
                "assessment": "No sources available to verify against"
            }

        # build the source context from retrieved documents
        source_text = "\n\n".join([doc.page_content for doc in source_docs])

        # ask the model to act as a fact-checker
        verify_prompt = f"""
        You are a fact-checker verifying whether an answer's factual claims
        are supported by the provided source documents.

        IMPORTANT: Focus ONLY on the factual content and claims. Do NOT penalise
        the answer for mentioning paper IDs, filenames, or titles that may differ
        from the source labels. Judge whether the actual information (facts,
        findings, methods) is supported by the text in the sources.

        SOURCE DOCUMENTS:
        {source_text[:8000]}

        QUESTION: {question}

        ANSWER TO VERIFY: {answer}

        Reply in EXACTLY this format:
        CLAIMS_TOTAL: [number of distinct factual claims in the answer]
        CLAIMS_SUPPORTED: [number of those claims supported by the source text]
        ASSESSMENT: [one sentence on how well grounded the factual content is]
        """

        try:
            response = self.llm.invoke(verify_prompt)
            content = response.content

            # parse the response
            claims_total = 0
            claims_supported = 0
            assessment = "Verification completed"

            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("CLAIMS_TOTAL:"):
                    nums = ''.join(filter(str.isdigit, line))
                    claims_total = int(nums) if nums else 0
                elif line.startswith("CLAIMS_SUPPORTED:"):
                    nums = ''.join(filter(str.isdigit, line))
                    claims_supported = int(nums) if nums else 0
                elif line.startswith("ASSESSMENT:"):
                    assessment = line.replace("ASSESSMENT:", "").strip()

            # calculate the trust score as a percentage
            if claims_total > 0:
                score = int((claims_supported / claims_total) * 100)
            else:
                score = 0
            score = min(100, max(0, score))

            # decide the trust level
            if score >= 80:
                level = "high"
            elif score >= 50:
                level = "medium"
            else:
                level = "low"

            return {
                "score": score,
                "level": level,
                "claims_total": claims_total,
                "claims_supported": claims_supported,
                "assessment": assessment
            }

        except Exception as e:
            return {
                "score": 0,
                "level": "error",
                "claims_total": 0,
                "claims_supported": 0,
                "assessment": f"Verification failed: {e}"
            }
    def compare_papers(self, topic):
        if self.vector_store is None:
            return {
                "answer": "no documents loaded yet - please upload papers first",
                "citations": [],
                "papers_used": []
            }
        question = f"""
        Compare and contrast what different papers say about: {topic}
        List the key similarities and differences between the papers.
        """
        return self.ask_across_papers(question)
    
    # MODULE 4 - Verify content against full paper text (for analysis features)
    def verify_against_papers(self, content):
        # verify generated content against the full text of all papers
        if not self.paper_full_text:
            return {
                "score": 0, "level": "unverified",
                "claims_total": 0, "claims_supported": 0,
                "assessment": "No papers available to verify against"
            }

        # build source text from all papers
        source_text = ""
        for name, full_text in self.paper_full_text.items():
            source_text += f"\n\n{full_text[:5000]}"

        verify_prompt = f"""
        You are a strict fact-checker. Verify whether the following content
        is supported by the provided source papers.

        SOURCE PAPERS:
        {source_text[:10000]}

        CONTENT TO VERIFY:
        {content[:4000]}

        Check the main claims against the sources. Reply EXACTLY in this format:
        CLAIMS_TOTAL: [number of main claims]
        CLAIMS_SUPPORTED: [number supported by sources]
        ASSESSMENT: [one sentence on how well grounded the content is]
        """

        try:
            response = self.llm.invoke(verify_prompt)
            content_resp = response.content
            claims_total = 0
            claims_supported = 0
            assessment = "Verification completed"

            for line in content_resp.split("\n"):
                line = line.strip()
                if line.startswith("CLAIMS_TOTAL:"):
                    nums = ''.join(filter(str.isdigit, line))
                    claims_total = int(nums) if nums else 0
                elif line.startswith("CLAIMS_SUPPORTED:"):
                    nums = ''.join(filter(str.isdigit, line))
                    claims_supported = int(nums) if nums else 0
                elif line.startswith("ASSESSMENT:"):
                    assessment = line.replace("ASSESSMENT:", "").strip()

            if claims_total > 0:
                score = int((claims_supported / claims_total) * 100)
            else:
                score = 0
            score = min(100, max(0, score))

            if score >= 80:
                level = "high"
            elif score >= 50:
                level = "medium"
            else:
                level = "low"

            return {
                "score": score, "level": level,
                "claims_total": claims_total,
                "claims_supported": claims_supported,
                "assessment": assessment
            }
        except Exception as e:
            return {
                "score": 0, "level": "error",
                "claims_total": 0, "claims_supported": 0,
                "assessment": f"Verification failed: {e}"
            }
    
    # MODULE 5 - Citation Extraction
    def extract_citations(self):
        # pull author, year, title from each paper for proper citing
        if not self.paper_full_text:
            return {}

        citations = {}
        for name, full_text in self.paper_full_text.items():
            # check if it's an API paper with metadata in the text
            # (PubMed/CrossRef save files start with "Title:" and "Authors:")
            if full_text.strip().startswith("Title:"):
                citation = self._parse_metadata_citation(full_text, name)
                citations[name] = citation
                continue

            # otherwise ask the AI to extract citation info from the paper
            prompt = f"""
            Extract the citation information from this research paper.
            Look at the first pages for the title, authors, and year.
            Paper filename: {name}
            Paper content: {full_text[:3000]}

            Reply in EXACTLY this format:
            AUTHORS: [first author's last name, or "First Author et al." if multiple]
            YEAR: [4 digit year]
            TITLE: [the paper title]
            """
            try:
                response = self.llm.invoke(prompt)
                content = response.content
                authors = "Unknown"
                year = "n.d."
                title = name

                for line in content.split("\n"):
                    line = line.strip()
                    if line.startswith("AUTHORS:"):
                        authors = line.replace("AUTHORS:", "").strip()
                    elif line.startswith("YEAR:"):
                        y = ''.join(filter(str.isdigit, line))
                        if len(y) >= 4:
                            year = y[:4]
                    elif line.startswith("TITLE:"):
                        title = line.replace("TITLE:", "").strip()

                # also try arxiv id for year as backup
                import re
                match = re.match(r"(\d{2})(\d{2})\.\d+", name)
                if match and year == "n.d.":
                    year = str(2000 + int(match.group(1)))

                citations[name] = {
                    "authors": authors,
                    "year": year,
                    "title": title,
                    "intext": f"{authors} ({year})"
                }
            except Exception as e:
                citations[name] = {
                    "authors": "Unknown",
                    "year": "n.d.",
                    "title": name,
                    "intext": f"({name})"
                }
        return citations

    def _parse_metadata_citation(self, full_text, name):
        # parse citation from PubMed/CrossRef saved text files
        authors = "Unknown"
        year = "n.d."
        title = name
        journal = ""
        for line in full_text.split("\n"):
            line = line.strip()
            if line.startswith("Title:"):
                title = line.replace("Title:", "").strip()
            elif line.startswith("Authors:"):
                auth_full = line.replace("Authors:", "").strip()
                # take first author's name for in-text citation
                first = auth_full.split(",")[0].strip() if auth_full else "Unknown"
                authors = first + (" et al." if "," in auth_full else "")
            elif line.startswith("Year:"):
                year = line.replace("Year:", "").strip()
            elif line.startswith("Journal:"):
                journal = line.replace("Journal:", "").strip()
        return {
            "authors": authors,
            "year": year,
            "title": title,
            "journal": journal,
            "intext": f"{authors} ({year})"
        }

    # MODULE 5 - Auto Literature Review Generator (with citations)
    def generate_literature_review(self, length="medium"):
        if not self.paper_full_text:
            return "no papers loaded - please upload papers first"

        # first extract proper citations for all papers
        citations = self.extract_citations()

        # build content with citation labels so the AI cites properly
        papers_content = ""
        for name, full_text in self.paper_full_text.items():
            cite = citations.get(name, {})
            intext = cite.get("intext", name)
            papers_content += (
                f"\n\n=== PAPER (cite this as: {intext}) ===\n"
                f"{full_text[:6000]}\n"
            )

        # build the references list
        references = "\n\nREFERENCES\n\n"
        for name, cite in citations.items():
            authors = cite.get("authors", "Unknown").rstrip(".")
            year = cite.get("year", "n.d.")
            title = cite.get("title", name).rstrip(".")
            journal = cite.get("journal", "").rstrip(".")
            ref_line = f"{authors} ({year}). {title}."
            if journal:
                ref_line += f" {journal}."
            references += ref_line + "\n\n"

        # comprehensive = section by section
        if length == "comprehensive":
            body = self._generate_long_review(papers_content)
            return body + references

        if length == "short":
            depth = "Write concisely. 1 short paragraph per section."
        elif length == "detailed":
            depth = "Write in-depth. 2 to 3 detailed paragraphs per section."
        else:
            depth = "Write a balanced review. 1 to 2 paragraphs per section."

        prompt = f"""
        You are an academic researcher writing a formal literature review.
        Write a well-structured review with these sections:
        1. INTRODUCTION
        2. THEMES AND APPROACHES
        3. COMPARISON OF METHODS
        4. RESEARCH GAPS
        5. CONCLUSION
        {depth}

        IMPORTANT: Use proper in-text citations. When you refer to a paper,
        cite it using the format given for each paper (e.g. "Smith et al. (2023)").
        Do NOT use filenames. Use the author-year citations provided.

        Formal academic tone.

        RESEARCH PAPERS:
        {papers_content}

        Write the complete literature review now:
        """
        try:
            response = self.llm.invoke(prompt)
            return response.content + references
        except Exception as e:
            return f"error generating literature review: {e}"

    def _generate_long_review(self, papers_content):
        # generate each section separately for a longer thesis-style review
        sections = [
            ("1. INTRODUCTION",
             "Write 3-4 detailed paragraphs introducing the research area, "
             "its importance, background, and scope of this review."),
            ("2. THEMES AND APPROACHES",
             "Write 4-5 detailed paragraphs identifying main themes and "
             "explaining what each paper contributes in depth."),
            ("3. COMPARISON OF METHODS",
             "Write 3-4 detailed paragraphs comparing methodologies with "
             "specific examples."),
            ("4. RESEARCH GAPS",
             "Write 3-4 detailed paragraphs identifying unexplored areas, "
             "limitations, and future research opportunities."),
            ("5. CONCLUSION",
             "Write 2-3 paragraphs summarizing the state of research and "
             "key takeaways."),
        ]

        full_review = "LITERATURE REVIEW\n\n"
        for section_title, instruction in sections:
            prompt = f"""
            You are writing the "{section_title}" section of an academic
            literature review based on these research papers.
            {instruction}

            IMPORTANT: Use the author-year in-text citations provided for
            each paper (e.g. "Smith et al. (2023)"). Never use filenames.

            Formal academic tone.
            RESEARCH PAPERS:
            {papers_content}
            Write only the {section_title} section now:
            """
            try:
                response = self.llm.invoke(prompt)
                full_review += f"\n\n{response.content}\n"
            except Exception as e:
                full_review += f"\n\n{section_title}\n\n[error: {e}]\n"

        return full_review
    
    # WP6 - Research Gap Identifier
    def identify_research_gaps(self):
        if not self.paper_full_text:
            return "no papers loaded - please upload papers first"

        papers_content = ""
        for name, full_text in self.paper_full_text.items():
            papers_content += f"\n\n=== PAPER: {name} ===\n{full_text[:6000]}\n"

        prompt = f"""
        You are a research advisor analyzing a collection of academic papers.
        Your task is to identify RESEARCH GAPS - important areas that these
        papers have NOT fully explored or addressed.

        Read the papers carefully and identify 5 to 7 genuine research gaps.
        For each gap, provide:
        - GAP: a clear description of what is missing or under-explored
        - WHY IT MATTERS: why filling this gap would be valuable
        - DIRECTION: a concrete suggestion for how it could be studied

        Base the gaps strictly on what is actually missing from these papers.

        RESEARCH PAPERS:
        {papers_content}

        List the research gaps now in the format above:
        """
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"error identifying gaps: {e}"

    # WP6 - Hypothesis Generator
    def generate_hypotheses(self):
        if not self.paper_full_text:
            return "no papers loaded - please upload papers first"

        papers_content = ""
        for name, full_text in self.paper_full_text.items():
            papers_content += f"\n\n=== PAPER: {name} ===\n{full_text[:5000]}\n"

        prompt = f"""
        You are a research scientist. Based on the following papers, propose
        5 novel, testable research hypotheses that could form the basis of
        future studies.

        For each hypothesis provide:
        - HYPOTHESIS: a clear, testable statement
        - RATIONALE: the reasoning from the papers that motivates it
        - HOW TO TEST: a brief suggestion of how it could be investigated

        Make the hypotheses specific and grounded in the papers, not generic.

        RESEARCH PAPERS:
        {papers_content}

        List the hypotheses now in the format above:
        """
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"error generating hypotheses: {e}"

    # FEATURE 1 - Paper Summary Generator (full paper)
    def generate_summaries(self):
        if not self.paper_full_text:
            return {}
        summaries = {}
        for paper_name, full_text in self.paper_full_text.items():
            try:
                prompt = f"""
                Summarize this research paper in exactly 5 clear bullet points.
                Each bullet point should capture a key contribution or finding.
                Read the whole paper carefully before summarizing.
                Paper content: {full_text[:12000]}
                Give exactly 5 bullet points:
                """
                response = self.llm.invoke(prompt)
                summaries[paper_name] = response.content
            except Exception as e:
                summaries[paper_name] = f"could not generate summary: {e}"
        return summaries

    # FEATURE 2 - Contradiction Detector (full papers)
    def detect_contradictions(self):
        if len(self.loaded_papers) < 2:
            return "need at least 2 papers to detect contradictions"
        papers_text = ""
        for name, full_text in self.paper_full_text.items():
            papers_text += f"\n\nPaper: {name}\nContent: {full_text[:8000]}\n"
        prompt = f"""
        Analyze these research papers carefully and find where they
        CONTRADICT each other. Read the full content of each paper.
        For each contradiction:
        - State what Paper A says
        - State what Paper B says
        - Explain why this is a contradiction
        Papers: {papers_text}
        List all contradictions found:
        """
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"error detecting contradictions: {e}"

    # FEATURE 3 - Key Findings Extractor (full papers)
    def extract_key_findings(self):
        if not self.paper_full_text:
            return "no papers loaded"
        all_content = ""
        for name, full_text in self.paper_full_text.items():
            all_content += f"\n\nFrom {name}: {full_text[:8000]}\n"
        prompt = f"""
        Extract the TOP 10 most important findings from ALL papers combined.
        Read the complete content of each paper carefully.
        Format as a numbered list.
        For each finding mention which paper it came from.
        Papers content: {all_content}
        Top 10 Key Findings:
        """
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"error extracting findings: {e}"

    # FEATURE 4 - Paper Similarity Score (full-paper cosine similarity)
    def calculate_similarity(self):
        if len(self.loaded_papers) < 2:
            return {}

        import numpy as np
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=0
        )

        # build one averaged vector representing the WHOLE paper
        names = list(self.paper_full_text.keys())
        vectors = {}
        for name in names:
            full_text = self.paper_full_text[name]
            chunks = splitter.split_text(full_text)
            if not chunks:
                continue
            # embed every chunk of the paper
            chunk_vectors = self.embeddings.embed_documents(chunks)
            # average them into one vector representing the whole paper
            vectors[name] = np.mean(np.array(chunk_vectors), axis=0)

        def cosine(a, b):
            dot = np.dot(a, b)
            norm = np.linalg.norm(a) * np.linalg.norm(b)
            if norm == 0:
                return 0.0
            return dot / norm

        scores = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                name1, name2 = names[i], names[j]
                if name1 not in vectors or name2 not in vectors:
                    continue
                sim = cosine(vectors[name1], vectors[name2])
                score = int(round(sim * 100))
                score = min(100, max(0, score))
                short1 = name1[:20] + "..." if len(name1) > 20 else name1
                short2 = name2[:20] + "..." if len(name2) > 20 else name2
                scores[f"{short1} vs {short2}"] = score
        return scores

    # FEATURE 5 - Mind Map Generator (fixed overlapping)
    def generate_mindmap_data(self):
        if not self.paper_contents:
            return []
        all_content = ""
        for name, content in self.paper_contents.items():
            all_content += f"\nFrom {name}: {content[:1000]}\n"
        prompt = f"""
        Extract main concepts and connections from these papers.
        Format EXACTLY like this - one connection per line:
        ConceptA -> ConceptB
        Use SHORT names (max 3 words each).
        No numbers, no special characters in concept names.
        Extract exactly 12 connections:
        Papers: {all_content}
        """
        try:
            response = self.llm.invoke(prompt)
            connections = []
            for line in response.content.strip().split('\n'):
                if '->' in line:
                    parts = line.split('->')
                    if len(parts) == 2:
                        source = parts[0].strip()[:20]
                        target = parts[1].strip()[:20]
                        if source and target and source != target:
                            connections.append((source, target))
            return connections[:12]
        except Exception as e:
            print(f"mind map error: {e}")
            return []

    # FEATURE 6 - Timeline Builder (full paper + arXiv year)
    def build_timeline(self):
        if not self.paper_full_text:
            return []
        import re
        timeline = []
        for paper_name, full_text in self.paper_full_text.items():
            year = "Unknown"
            # arXiv filenames encode the date as YYMM.number
            match = re.match(r"(\d{2})(\d{2})\.\d+", paper_name)
            if match:
                yy = int(match.group(1))
                year = str(2000 + yy)
            else:
                # for PubMed/CrossRef files, read the "Year:" line from the text
                for line in full_text.split("\n"):
                    if line.strip().startswith("Year:"):
                        y = line.replace("Year:", "").strip()
                        if y and y != "Unknown":
                            year = y
                        break
                # fallback: find a 4-digit year in the filename
                if year == "Unknown":
                    ymatch = re.search(r"(19|20)\d{2}", paper_name)
                    if ymatch:
                        year = ymatch.group(0)

            prompt = f"""
            Give a one sentence description of what this paper contributed
            to research. Read the full paper. Max 20 words.
            Paper: {paper_name}
            Content: {full_text[:6000]}
            Reply with just the sentence:
            """
            try:
                response = self.llm.invoke(prompt)
                contribution = response.content.strip()
            except Exception as e:
                contribution = "No description available"

            timeline.append({
                "paper": paper_name,
                "year": year,
                "contribution": contribution
            })
        timeline.sort(key=lambda x: x["year"])
        return timeline
    
    # FEATURE 7 - Export Results (fixed to return content for download)
    def prepare_export(self, content, title="Results"):
        export_text = f"""RAG Research Assistant - {title}
Pranay Teja Chintakunta - 25079476
MSc AI & ML - University of Limerick
{'='*50}

{content}
"""
        return export_text

    # WP7 - Voice Input (speech to text via Whisper)
    def transcribe_audio(self, audio_bytes):
        import tempfile
        import os
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            from openai import OpenAI
            client = OpenAI()
            with open(tmp_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            os.remove(tmp_path)
            return transcript.text
        except Exception as e:
            return f"transcription error: {e}"

    # WP7 - Voice Output (text to speech)
    def text_to_speech(self, text):
        import tempfile
        import os
        try:
            from openai import OpenAI
            client = OpenAI()
            speech_text = text[:2000]

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp_path = tmp.name

            with client.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="alloy",
                input=speech_text
            ) as response:
                response.stream_to_file(tmp_path)

            with open(tmp_path, "rb") as f:
                audio_bytes = f.read()
            os.remove(tmp_path)
            return audio_bytes
        except Exception as e:
            print(f"TTS error: {e}")
            return None

    # WP8 - Multilingual Support
    def translate_text(self, text, target_language):
        # translate any text into the target language
        if target_language == "English":
            return text  # no translation needed
        try:
            prompt = f"""
            Translate the following text into {target_language}.
            Keep any citations, author names, and technical terms intact.
            Only return the translation, nothing else.

            Text: {text}

            Translation in {target_language}:
            """
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"translation error: {e}\n\n(original) {text}"

    def get_loaded_papers(self):
        return self.loaded_papers


if __name__ == "__main__":
    rag = MultiDocRAG()
    print("all bugs fixed - ready!")