# Pranay Teja Chintakunta - 25079476
# MSc AI & ML - University of Limerick
# Main App - Bug Fixed Version

import streamlit as st
from arxiv_search import ArxivSearcher, PubMedSearcher, CrossRefSearcher, ResearchAlertSystem
from multi_doc import MultiDocRAG
import os
import plotly.graph_objects as go
import networkx as nx

st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="🔬",
    layout="wide"
)

st.title("RAG Research Assistant")
st.caption("Pranay Teja Chintakunta | MSc AI & ML | University of Limerick")
st.divider()

# fix 1 - proper session state management
if "rag" not in st.session_state:
    st.session_state.rag = MultiDocRAG()
    # start each fresh launch with a clean slate
    st.session_state.rag.clear_all()
if "arxiv" not in st.session_state:
    st.session_state.arxiv = ArxivSearcher()
if "pubmed" not in st.session_state:
    st.session_state.pubmed = PubMedSearcher()
if "crossref" not in st.session_state:
    st.session_state.crossref = CrossRefSearcher()
if "papers_processed" not in st.session_state:
    st.session_state.papers_processed = False
if "alerts" not in st.session_state:
    st.session_state.alerts = ResearchAlertSystem()

rag = st.session_state.rag
arxiv_searcher = st.session_state.arxiv
pubmed_searcher = st.session_state.pubmed
crossref_searcher = st.session_state.crossref
alert_system = st.session_state.alerts

# sidebar
with st.sidebar:
    st.header("Upload Papers")

    # file size warning
    st.caption("Max 200MB per file • PDF only")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    # New Session button - fresh start like ChatGPT
    if st.button("➕  New Session", type="primary", key="new_session"):
        rag.clear_all()
        st.session_state.papers_processed = False
        st.session_state.rag = MultiDocRAG()
        # clear arxiv search results too
        if "arxiv_results" in st.session_state:
            del st.session_state.arxiv_results
        st.success("Started a fresh session! Upload papers to begin.")
        st.rerun()

    if uploaded_files:
        # fix 3 - file size check
        oversized = [
            f.name for f in uploaded_files
            if f.size > 200 * 1024 * 1024
        ]
        if oversized:
            st.error(f"Files too large: {oversized}")
        else:
            if st.button("Process Documents", type="primary"):
                with st.spinner("Processing papers..."):
                    pdf_paths = []
                    # use a dedicated uploads folder that we control
                    upload_dir = os.path.join(os.getcwd(), "uploaded_papers")
                    os.makedirs(upload_dir, exist_ok=True)

                    for file in uploaded_files:
                         # clean filename - remove any (1), (2) duplicated suffixes
                        clean_name = file.name
                        save_path = os.path.join(upload_dir, clean_name)
                        with open(save_path, "wb") as f:
                            f.write(file.read())
                        pdf_paths.append(save_path)
                        rag.temp_files.append(save_path)

                    pages = rag.load_multiple_pdfs(pdf_paths)
                    if pages:
                        rag.build_multi_index(pages)
                        st.session_state.papers_processed = True
                        st.success(
                            f"Loaded {len(rag.get_loaded_papers())} papers!"
                        )
                    else:
                        st.error("Could not load any pages from PDFs!")

    # show loaded papers
    if rag.get_loaded_papers():
        st.subheader("Loaded Papers:")
        for paper in rag.get_loaded_papers():
            st.write(f"📄 {paper}")

# helper function for export button
def show_export(content, filename, title):
    export_text = rag.prepare_export(content, title)
    st.download_button(
        label="Download Results",
        data=export_text,
        file_name=filename,
        mime="text/plain"
    )

# check if papers are loaded before showing features
def check_papers():
    if not rag.get_loaded_papers():
        st.warning("Please upload and process papers first!")
        return False
    return True

# reusable trust badge display for verified features
def show_trust_badge(verification):
    score = verification["score"]
    level = verification["level"]

    if level == "high":
        st.success(f"🟢  Trust Score: {score}%  —  High Confidence")
    elif level == "medium":
        st.warning(f"🟡  Trust Score: {score}%  —  Medium Confidence")
    elif level == "low":
        st.error(f"🔴  Trust Score: {score}%  —  Low Confidence, verify manually")
    else:
        st.info(f"⚪  Trust Score: {score}%  —  {verification['assessment']}")

    with st.expander("🛡️  Verification Details"):
        st.write(f"**Claims checked:** {verification['claims_total']}")
        st.write(f"**Claims supported:** {verification['claims_supported']}")
        st.write(f"**Assessment:** {verification['assessment']}")
        st.caption("Automatically verified against your source papers.")

# 7 feature tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13 = st.tabs([
    "Ask Questions",
    "Compare Papers",
    "Summaries",
    "Contradictions",
    "Key Findings",
    "Similarity",
    "Mind Map & Timeline",
    "Live ArXiv Search",
    "PubMed Search",
    "CrossRef Search",
    "Literature Review",
    "Research Gaps",
    "Research Alerts"
])

# TAB 1
with tab1:
    st.header("Ask Across All Papers")

    # WP8 - Language selector
    answer_language = st.selectbox(
        "🌍 Answer language:",
        ["English", "Spanish", "French", "German", "Hindi", "Telugu",
         "Chinese", "Arabic", "Portuguese", "Japanese", "Italian", "Russian"],
        key="lang_select"
    )

    st.caption("🎤 Speak your question or type it below")
    audio_value = st.audio_input("Record your question")

    voice_question = ""
    if audio_value:
        with st.spinner("Transcribing your voice..."):
            audio_bytes = audio_value.read()
            voice_question = rag.transcribe_audio(audio_bytes)
        if voice_question and not voice_question.startswith("transcription error"):
            st.success(f"You said: {voice_question}")

    typed_question = st.text_input("Or type your question:")
    question = voice_question if voice_question else typed_question

    if st.button("Get Answer", key="ask_btn"):
        if not check_papers():
            pass
        elif not question:
            st.warning("Please type or record a question first!")
        else:
            with st.spinner("Searching all papers..."):
                result = rag.ask_across_papers(question)
            # store result in session state so it survives reruns
            st.session_state.last_answer = result

    # show the answer if we have one (outside the button block)
    if "last_answer" in st.session_state:
        result = st.session_state.last_answer

        st.subheader("Answer:")
        # WP8 - translate answer if needed
        display_answer = result["answer"]
        if answer_language != "English":
            with st.spinner(f"Translating to {answer_language}..."):
                display_answer = rag.translate_text(result["answer"], answer_language)
        st.write(display_answer)

        # WP7 - Voice output
        if st.button("🔊 Listen to Answer", key="tts_btn"):
            with st.spinner("Generating audio..."):
                audio = rag.text_to_speech(display_answer)
            if audio:
                st.audio(audio, format="audio/mp3")

        # verification badge
        with st.spinner("Verifying answer against sources..."):
            verification = rag.verify_answer(
                question if question else "",
                result["answer"],
                result.get("source_docs", [])
            )
        show_trust_badge(verification)

        # papers used and citations
        if result["papers_used"]:
            st.subheader("Papers Used:")
            for paper in result["papers_used"]:
                st.success(f"📄 {paper}")
        if result["citations"]:
            st.subheader("Citations:")
            for cite in result["citations"]:
                st.info(cite)
        show_export(result["answer"], "answer.txt", "Question Answer")

            # MODULE 4 - verify the answer against sources
        with st.spinner("Verifying answer against sources..."):
                verification = rag.verify_answer(
                    question,
                    result["answer"],
                    result.get("source_docs", [])
                )

            # show the trust badge with color
        score = verification["score"]
        level = verification["level"]

        if level == "high":
            st.success(f"🟢  Trust Score: {score}%  —  High Confidence")
        elif level == "medium":
            st.warning(f"🟡  Trust Score: {score}%  —  Medium Confidence")
        elif level == "low":
            st.error(f"🔴  Trust Score: {score}%  —  Low Confidence, verify manually")
        else:
            st.info(f"⚪  Trust Score: {score}%  —  {verification['assessment']}")

            # expandable detailed verification report
            with st.expander("🛡️  Verification Details"):
                st.write(f"**Claims checked:** {verification['claims_total']}")
                st.write(f"**Claims supported by sources:** {verification['claims_supported']}")
                st.write(f"**Assessment:** {verification['assessment']}")
                st.caption(
                    "This answer was automatically verified against the "
                    "retrieved source documents to detect possible hallucination."
                )

            if result["papers_used"]:
                st.subheader("Papers Used:")
                for paper in result["papers_used"]:
                    st.success(f"📄 {paper}")
            if result["citations"]:
                st.subheader("Citations:")
                for cite in result["citations"]:
                    st.info(cite)
            show_export(result["answer"], "answer.txt", "Question Answer")

# TAB 2
with tab2:
    st.header("Compare Papers On A Topic")
    topic = st.text_input("Enter topic to compare:", key="compare_input")
    if st.button("Compare", key="compare_btn"):
        if not check_papers():
            pass
        elif not topic:
            st.warning("Please enter a topic first!")
        else:
            with st.spinner("Comparing papers..."):
                result = rag.compare_papers(topic)
            st.subheader("Comparison:")
            st.write(result["answer"])
            # verify the comparison
            with st.spinner("Verifying against sources..."):
                v = rag.verify_answer(
                    topic, result["answer"], result.get("source_docs", [])
                )
            show_trust_badge(v)
            if result["citations"]:
                st.subheader("Sources:")
                for cite in result["citations"]:
                    st.info(cite)
            show_export(result["answer"], "comparison.txt", "Paper Comparison")

# TAB 3
with tab3:
    st.header("Paper Summaries")
    if st.button("Generate Summaries", key="summary_btn"):
        if not check_papers():
            pass
        else:
            with st.spinner("Summarising all papers..."):
                summaries = rag.generate_summaries()
            all_summaries = ""
            for paper, summary in summaries.items():
                with st.expander(f"📄 {paper}"):
                    st.write(summary)
                all_summaries += f"\n{paper}:\n{summary}\n"
            show_export(all_summaries, "summaries.txt", "Paper Summaries")

# TAB 4
with tab4:
    st.header("Contradiction Detector")
    st.caption("Finds where papers disagree with each other")
    if st.button("Find Contradictions", key="contra_btn"):
        if not check_papers():
            pass
        elif len(rag.get_loaded_papers()) < 2:
            st.warning("Need at least 2 papers to detect contradictions!")
        else:
            with st.spinner("Analysing contradictions..."):
                contradictions = rag.detect_contradictions()
            st.subheader("Contradictions Found:")
            st.write(contradictions)
            show_export(
                contradictions, "contradictions.txt", "Contradictions"
            )

# TAB 5
with tab5:
    st.header("Key Findings Extractor")
    if st.button("Extract Key Findings", key="findings_btn"):
        if not check_papers():
            pass
        else:
            with st.spinner("Extracting findings..."):
                findings = rag.extract_key_findings()
            st.subheader("Top 10 Key Findings:")
            st.write(findings)
            # verify the findings
            with st.spinner("Verifying against sources..."):
                v = rag.verify_against_papers(findings)
            show_trust_badge(v)
            show_export(findings, "key_findings.txt", "Key Findings")

# TAB 6
with tab6:
    st.header("Paper Similarity Scores")
    if st.button("Calculate Similarity", key="sim_btn"):
        if not check_papers():
            pass
        elif len(rag.get_loaded_papers()) < 2:
            st.warning("Need at least 2 papers for similarity!")
        else:
            with st.spinner("Calculating similarity..."):
                scores = rag.calculate_similarity()
            if scores:
                st.subheader("Similarity Scores:")
                for pair, score in scores.items():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.progress(score / 100)
                    with col2:
                        st.metric(label=pair, value=f"{score}%")

# TAB 7
with tab7:
    col1, col2 = st.columns(2)

    with col1:
        st.header("Mind Map")
        if st.button("Generate Mind Map", key="mindmap_btn"):
            if not check_papers():
                pass
            else:
                with st.spinner("Building mind map..."):
                    connections = rag.generate_mindmap_data()
                if connections:
                    G = nx.DiGraph()
                    for src, tgt in connections:
                        G.add_edge(src, tgt)

                    # fix 6 - better layout to prevent overlap
                    pos = nx.kamada_kawai_layout(G)

                    edge_x, edge_y = [], []
                    for edge in G.edges():
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])

                    node_x = [pos[n][0] for n in G.nodes()]
                    node_y = [pos[n][1] for n in G.nodes()]
                    node_text = list(G.nodes())

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=edge_x, y=edge_y,
                        mode='lines',
                        line=dict(width=1.5, color='#888'),
                        hoverinfo='none'
                    ))
                    fig.add_trace(go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers+text',
                        text=node_text,
                        textposition="top center",
                        textfont=dict(size=10),
                        marker=dict(
                            size=20,
                            color='#2196F3',
                            line=dict(width=2, color='white')
                        ),
                        hoverinfo='text'
                    ))
                    fig.update_layout(
                        showlegend=False,
                        height=600,
                        title="Concept Mind Map",
                        xaxis=dict(showgrid=False, zeroline=False,
                                   showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False,
                                   showticklabels=False),
                        margin=dict(l=40, r=40, t=60, b=40)
                    )
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.warning("Could not generate mind map. Try again!")

    with col2:
        st.header("Research Timeline")
        if st.button("Build Timeline", key="timeline_btn"):
            if not check_papers():
                pass
            else:
                with st.spinner("Building timeline..."):
                    timeline = rag.build_timeline()
                timeline_text = ""
                for item in timeline:
                    st.markdown(f"""
**📅 {item['year']}**

📄 `{item['paper']}`

💡 {item['contribution']}

---""")
                    timeline_text += (
                        f"{item['year']}: {item['paper']}\n"
                        f"{item['contribution']}\n\n"
                    )
                show_export(
                    timeline_text,
                    "timeline.txt",
                    "Research Timeline"
                )
# TAB 8 - Live ArXiv Search
with tab8:
    st.header("Search Latest Papers on ArXiv")
    st.caption("Find and load the newest research papers automatically")

    search_query = st.text_input(
        "Enter a research topic:",
        placeholder="e.g. retrieval augmented generation",
        key="arxiv_query"
    )

    num_results = st.slider("Number of papers", 1, 10, 5)

    if st.button("Search ArXiv", key="arxiv_search_btn"):
        if not search_query:
            st.warning("Please enter a search topic!")
        else:
            with st.spinner("Searching ArXiv for latest papers..."):
                papers = arxiv_searcher.search_papers(search_query, num_results)
            if papers:
                st.session_state.arxiv_results = papers
                st.success(f"Found {len(papers)} papers!")
            else:
                st.error("No papers found. Try a different topic.")

    # show search results
    if "arxiv_results" in st.session_state:
        st.subheader("Search Results:")
        for i, paper in enumerate(st.session_state.arxiv_results):
            with st.expander(f"{paper['title']} ({paper['published']})"):
                st.write(f"**Authors:** {', '.join(paper['authors'][:5])}")
                st.write(f"**Published:** {paper['published']}")
                st.write(f"**ArXiv ID:** {paper['arxiv_id']}")
                st.write(f"**Categories:** {', '.join(paper['categories'][:3])}")
                st.write(f"**Abstract:** {paper['summary'][:400]}...")

                if st.button(f"Load this paper into system", key=f"load_{i}"):
                    with st.spinner("Downloading and processing..."):
                        pdf_path = arxiv_searcher.download_paper(
                            paper['pdf_url'], paper['arxiv_id']
                        )
                        if pdf_path:
                            pages = rag.load_multiple_pdfs([pdf_path])
                            if pages:
                                rag.build_multi_index(pages)
                                st.success(f"Loaded '{paper['title'][:50]}...' into your library!")
                            else:
                                st.error("Could not process the paper.")
                        else:
                            st.error("Could not download the paper.")
# TAB 9 - PubMed Medical Paper Search
with tab9:
    st.header("Search Medical Research on PubMed")
    st.caption("Find and load the latest medical and biology papers")

    pubmed_query = st.text_input(
        "Enter a medical topic:",
        placeholder="e.g. cancer immunotherapy",
        key="pubmed_query"
    )

    pubmed_num = st.slider("Number of papers", 1, 10, 5, key="pubmed_slider")

    if st.button("Search PubMed", key="pubmed_search_btn"):
        if not pubmed_query:
            st.warning("Please enter a medical topic!")
        else:
            with st.spinner("Searching PubMed for latest research..."):
                papers = pubmed_searcher.search_papers(pubmed_query, pubmed_num)
            if papers:
                st.session_state.pubmed_results = papers
                st.success(f"Found {len(papers)} papers!")
            else:
                st.error("No papers found. Try a different topic.")

    # show search results
    if "pubmed_results" in st.session_state:
        st.subheader("Search Results:")
        for i, paper in enumerate(st.session_state.pubmed_results):
            with st.expander(f"{paper['title']} ({paper['published']})"):
                st.write(f"**Authors:** {', '.join(paper['authors'][:5])}")
                st.write(f"**Year:** {paper['published']}")
                st.write(f"**PubMed:** {paper['pubmed_url']}")
                st.write(f"**Abstract:** {paper['summary'][:500]}...")

                if st.button(f"Load this paper into system", key=f"pmload_{i}"):
                    with st.spinner("Loading abstract into system..."):
                        txt_path = pubmed_searcher.save_abstract_as_text(paper)
                        if txt_path:
                            pages = rag.load_multiple_pdfs([txt_path])
                            if pages:
                                rag.build_multi_index(pages)
                                st.success(f"Loaded '{paper['title'][:50]}...' into your library!")
                            else:
                                st.error("Could not process the abstract.")
                        else:
                            st.error("Could not save the abstract.")

# TAB 10 - CrossRef Journal Search
with tab10:
    st.header("Search Academic Journals on CrossRef")
    st.caption("Find papers across all academic journals with full citations")

    crossref_query = st.text_input(
        "Enter a research topic:",
        placeholder="e.g. machine learning healthcare",
        key="crossref_query"
    )

    crossref_num = st.slider("Number of papers", 1, 10, 5, key="crossref_slider")

    if st.button("Search CrossRef", key="crossref_search_btn"):
        if not crossref_query:
            st.warning("Please enter a research topic!")
        else:
            with st.spinner("Searching CrossRef journals..."):
                papers = crossref_searcher.search_papers(crossref_query, crossref_num)
            if papers:
                st.session_state.crossref_results = papers
                st.success(f"Found {len(papers)} papers!")
            else:
                st.error("No papers found. Try a different topic.")

    # show results
    if "crossref_results" in st.session_state:
        st.subheader("Search Results:")
        for i, paper in enumerate(st.session_state.crossref_results):
            with st.expander(f"{paper['title']} ({paper['published']})"):
                st.write(f"**Authors:** {', '.join(paper['authors'][:5])}")
                st.write(f"**Year:** {paper['published']}")
                st.write(f"**Journal:** {paper['journal']}")
                st.write(f"**DOI:** {paper['doi_url']}")
                st.write(f"**Abstract:** {paper['summary'][:500]}...")

                if st.button(f"Load this paper into system", key=f"crload_{i}"):
                    with st.spinner("Loading metadata into system..."):
                        txt_path = crossref_searcher.save_metadata_as_text(paper)
                        if txt_path:
                            pages = rag.load_multiple_pdfs([txt_path])
                            if pages:
                                rag.build_multi_index(pages)
                                st.success(f"Loaded '{paper['title'][:50]}...' into your library!")
                            else:
                                st.error("Could not process the metadata.")
                        else:
                            st.error("Could not save the metadata.")

# TAB 11 - Auto Literature Review with Citations
with tab11:
    st.header("Automatic Literature Review Generator")
    st.caption("Generate a fully cited academic literature review from all your papers")

    review_length = st.radio(
        "Choose review length:",
        ["short", "medium", "detailed", "comprehensive"],
        index=1,
        horizontal=True,
        key="review_length"
    )

    length_info = {
        "short": "Concise — about 5 paragraphs total",
        "medium": "Balanced — about 8-10 paragraphs",
        "detailed": "In-depth — about 12-15 paragraphs",
        "comprehensive": "Thesis-style — 15-20 paragraphs (takes longer)"
    }
    st.info(f"📝 {length_info[review_length]}")

    if review_length == "comprehensive":
        st.warning("⏳ Comprehensive mode generates each section separately and may take 30-60 seconds.")

    if st.button("Generate Literature Review", key="litreview_btn"):
        if not check_papers():
            pass
        else:
            with st.spinner(f"Writing your {review_length} literature review with citations..."):
                review = rag.generate_literature_review(review_length)

            st.subheader("Literature Review:")
            st.write(review)
            # verify the literature review
            with st.spinner("Verifying against sources..."):
                v = rag.verify_against_papers(review)
            show_trust_badge(v)

            show_export(review, "literature_review.txt", "Literature Review")

# TAB 12 - Research Gaps & Hypotheses
with tab12:
    st.header("Research Gap Identifier & Hypothesis Generator")
    st.caption("Discover unexplored directions and generate testable hypotheses")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Research Gaps")
        if st.button("Identify Research Gaps", key="gaps_btn"):
            if not check_papers():
                pass
            else:
                with st.spinner("Analysing papers for research gaps..."):
                    gaps = rag.identify_research_gaps()
                st.write(gaps)
                show_export(gaps, "research_gaps.txt", "Research Gaps")

    with col2:
        st.subheader("Hypotheses")
        if st.button("Generate Hypotheses", key="hypo_btn"):
            if not check_papers():
                pass
            else:
                with st.spinner("Generating research hypotheses..."):
                    hypotheses = rag.generate_hypotheses()
                st.write(hypotheses)
                show_export(hypotheses, "hypotheses.txt", "Research Hypotheses")

# TAB 13 - Research Alert System
with tab13:
    st.header("Research Alert System")
    st.caption("Save topics and check for the latest papers published on them")

    # add a new topic
    new_topic = st.text_input(
        "Add a research topic to track:",
        placeholder="e.g. retrieval augmented generation",
        key="alert_topic"
    )
    if st.button("Add Topic", key="add_topic_btn"):
        if new_topic:
            if alert_system.add_topic(new_topic):
                st.success(f"Now tracking: {new_topic}")
            else:
                st.info("Topic already tracked or empty.")

    # show saved topics
    topics = alert_system.get_topics()
    if topics:
        st.subheader("Your Tracked Topics:")
        for topic in topics:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"🔔 {topic}")
            with col2:
                if st.button("Check", key=f"check_{topic}"):
                    with st.spinner(f"Checking for new papers on '{topic}'..."):
                        result = alert_system.check_for_new_papers(topic)
                    if result["new"]:
                        st.success(f"Found {len(result['new'])} new papers!")
                        for paper in result["new"]:
                            with st.expander(f"🆕 {paper['title']} ({paper['published']})"):
                                st.write(f"**Authors:** {', '.join(paper['authors'])}")
                                st.write(f"**ArXiv ID:** {paper['arxiv_id']}")
                                st.write(f"**Summary:** {paper['summary']}...")
                    else:
                        st.info("No new papers since last check.")
    else:
        st.info("Add a topic above to start tracking new research.")