
import json
import logging
import time

import streamlit as st

st.set_page_config(
    page_title="Multi-Modal Document Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');

:root {
    --bg: #fff5fb;
    --card: rgba(255, 255, 255, 0.9);
    --card-strong: rgba(255, 255, 255, 0.97);
    --text: #471f32;
    --muted: #69475a;
    --accent: #e85da2;
    --line: rgba(232, 93, 162, 0.16);
}

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
    color: var(--text);
}

h1, h2, h3, h4, h5, h6,
p,
label,
span,
div,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] *,
[data-testid="stForm"] label,
.stSelectbox label,
.stTextInput label,
.stFileUploader label,
.stTextInput label p,
.stSelectbox label p,
.stFileUploader label p {
    color: var(--text);
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(255, 178, 214, 0.55), transparent 28%),
        radial-gradient(circle at top right, rgba(255, 215, 231, 0.7), transparent 22%),
        linear-gradient(180deg, #fff7fb 0%, #ffeef7 44%, #fff8fc 100%);
}

section[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"] {
    display: none !important;
}

.block-container {
    padding-top: 2.2rem;
    padding-bottom: 3rem;
    max-width: 1180px;
}

.hero {
    background: linear-gradient(135deg, rgba(255,255,255,0.94), rgba(255,239,247,0.92));
    border: 1px solid var(--line);
    border-radius: 28px;
    padding: 2rem 2.2rem;
    box-shadow: 0 18px 50px rgba(232, 93, 162, 0.08);
    margin-bottom: 1.35rem;
}

.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    line-height: 1.1;
    color: #9f2f67;
    letter-spacing: -0.03em;
    margin-bottom: 0.6rem;
    max-width: 980px;
}

.hero-sub {
    color: #69475a;
    font-size: 1.05rem;
    max-width: 820px;
}

.chip {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.5rem 0.85rem;
    border-radius: 999px;
    background: rgba(255, 202, 225, 0.36);
    color: #9b4271;
    border: 1px solid rgba(232, 93, 162, 0.14);
    font-size: 0.86rem;
    font-weight: 600;
    margin-right: 0.3rem;
    margin-bottom: 0.3rem;
}

.panel-title {
    font-size: 1.18rem;
    font-weight: 800;
    color: #b13e77;
    margin-bottom: 0.2rem;
}

.panel-sub {
    color: var(--muted);
    font-size: 0.92rem;
    margin-bottom: 1rem;
}

.step-grid {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 0.8rem;
    margin-bottom: 0.9rem;
}

.step-card {
    background: rgba(255,255,255,0.88);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 0.95rem;
    min-height: 112px;
}

.step-card.current {
    border-color: rgba(232, 93, 162, 0.42);
    box-shadow: 0 10px 24px rgba(232, 93, 162, 0.12);
    background: linear-gradient(180deg, rgba(255, 234, 244, 0.95), rgba(255,255,255,0.95));
}

.step-card.done {
    border-color: rgba(126, 204, 157, 0.32);
    background: linear-gradient(180deg, rgba(239, 252, 244, 0.95), rgba(255,255,255,0.95));
}

.step-label {
    font-size: 0.74rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    margin-bottom: 0.45rem;
}

.step-name {
    font-size: 0.98rem;
    font-weight: 800;
    color: var(--text);
    margin-bottom: 0.35rem;
}

.step-progress {
    font-size: 0.83rem;
    color: #9a4a74;
}

.doc-card {
    background: var(--card-strong);
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 1rem 1.1rem;
}

.doc-title {
    font-weight: 700;
    color: var(--text);
    margin-bottom: 0.35rem;
}

.doc-meta {
    color: #69475a;
    font-size: 0.9rem;
}

.answer-box {
    background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(255, 243, 249, 0.98));
    border: 1px solid rgba(232, 93, 162, 0.2);
    border-left: 5px solid var(--accent);
    border-radius: 18px;
    padding: 1.2rem 1.25rem;
    line-height: 1.7;
    color: var(--text);
}

.answer-box,
.answer-box p,
.answer-box li,
.answer-box span,
.answer-box strong,
.answer-box em,
.answer-box h1,
.answer-box h2,
.answer-box h3,
.answer-box h4,
.answer-box code {
    color: var(--text) !important;
}

.mini-note {
    color: #69475a;
    font-size: 0.9rem;
}

.source-pill {
    display: inline-flex;
    margin: 0.2rem 0.35rem 0.2rem 0;
    padding: 0.42rem 0.75rem;
    border-radius: 999px;
    background: rgba(255, 220, 236, 0.7);
    border: 1px solid rgba(232, 93, 162, 0.18);
    color: #9a416f;
    font-size: 0.82rem;
    font-weight: 600;
}

.stButton > button {
    background: linear-gradient(135deg, #f062a6, #ff9dc8) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    font-weight: 700 !important;
    min-height: 2.9rem;
    box-shadow: 0 12px 24px rgba(232, 93, 162, 0.16);
}

.stTextInput > div > div > input,
[data-baseweb="select"] > div,
.stFileUploader {
    background: rgba(255,255,255,0.95) !important;
    border: 1px solid rgba(232, 93, 162, 0.18) !important;
    border-radius: 14px !important;
    color: var(--text) !important;
}

.stTextInput input::placeholder,
textarea::placeholder {
    color: #b18a9e !important;
    opacity: 1 !important;
}

[data-baseweb="select"] *,
[data-baseweb="select"] span,
[data-baseweb="select"] div {
    color: var(--text) !important;
}

.stTextInput input,
.stSelectbox div,
.stSelectbox span,
.stFileUploader div,
.stAlert,
.stAlert * {
    color: var(--text) !important;
}

.stCaption,
.stCaption *,
[data-testid="stExpander"],
[data-testid="stExpander"] *,
.stMarkdown,
.stMarkdown * {
    color: var(--text) !important;
}

[data-testid="stTabsContainer"] h3,
[data-testid="stTabsContainer"] h2,
[data-testid="stTabsContainer"] label,
[data-testid="stTabsContainer"] p,
[data-testid="stTabsContainer"] div {
    color: var(--text);
}

.stTabs [data-baseweb="tab-list"] {
    background: rgba(255, 246, 251, 0.9);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 0.35rem;
}

.stTabs [data-baseweb="tab"] {
    color: var(--muted);
    font-weight: 700;
    border-radius: 12px;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(240, 98, 166, 0.14), rgba(255, 157, 200, 0.22)) !important;
    color: #b13e77 !important;
}

@media (max-width: 900px) {
    .step-grid {
        grid-template-columns: 1fr;
    }

    .hero-title {
        font-size: 2.4rem;
    }
}
</style>
""",
    unsafe_allow_html=True,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)

try:
    from config import cfg, resolve_colpali_device
    from evaluation import Evaluator
    from pipeline import RAGPipeline

    IMPORTS_OK = True
except Exception as exc:
    IMPORTS_OK = False
    IMPORT_ERROR = str(exc)


PROCESS_STEPS = {
    1: "Convert PDF",
    2: "Load Model",
    3: "Embed Pages",
    4: "Store Vectors",
    5: "Ready",
}


def init_state():
    defaults = {
        "chat_history": [],
        "active_doc_id": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource(show_spinner=False)
def get_pipeline():
    return RAGPipeline()


def build_steps_markup(step_state: dict) -> str:
    cards = []
    current_step = step_state["current_step"]
    progress_map = step_state["progress"]

    for step_number, label in PROCESS_STEPS.items():
        if step_number < current_step:
            status_class = "done"
            progress_text = "Completed"
        elif step_number == current_step:
            status_class = "current"
            progress_text = f"{int(progress_map.get(step_number, 0.0) * 100)}%"
        else:
            status_class = ""
            progress_text = "Waiting"

        cards.append(
            f"<div class='step-card {status_class}'><div class='step-label'>Step {step_number}</div><div class='step-name'>{label}</div><div class='step-progress'>{progress_text}</div></div>"
        )

    return "<div class='step-grid'>" + "".join(cards) + "</div>"


def render_sources(citations: list[dict]) -> str:
    return "".join(
        f"<span class='source-pill'>Page {citation['page']} · {citation['doc_name']} · {citation.get('section_title', 'Section')} · {citation.get('chunk_type', 'text')} · {citation['score']}</span>"
        for citation in citations
    )


init_state()

if IMPORTS_OK:
    cfg.colpali.device = resolve_colpali_device(cfg.colpali.model_name)
    cfg.top_k = 3

st.markdown(
    "<div class='hero'><div class='hero-title'>Multi-Modal Document Intelligence (RAG-Based QA System)</div><div class='hero-sub'>Upload a PDF, index its content, and ask grounded questions with clear evidence from the document.</div></div>",
    unsafe_allow_html=True,
)

if not IMPORTS_OK:
    st.error(f"Import error: {IMPORT_ERROR}")
    st.stop()

pipeline = get_pipeline()
docs = pipeline.list_documents()

header_left, header_right = st.columns([2.5, 1.2])
with header_left:
    st.markdown("<div class='panel-title'>Simple flow</div><div class='panel-sub'>Step 1 uploads and indexes the PDF. Step 2 lets you ask questions about the indexed file.</div>", unsafe_allow_html=True)
with header_right:
    with st.expander("Answer settings", expanded=False):
        openai_api_key = st.text_input(
            "OpenAI API Key",
            value="",
            type="password",
            placeholder="sk-...",
            help="Optional. Leave this blank to use the key loaded from your local .env file. If you enter a value here, it is only used for the current session.",
        )
        session_openai_key = openai_api_key.strip()
        if session_openai_key:
            cfg.openai.api_key = session_openai_key
        st.markdown(
            f"<div class='panel-sub'>Answer mode: {'OpenAI multimodal grounding' if cfg.openai.api_key else 'Local grounded extraction'}<br>Pool factor: {cfg.colpali.pool_factor}</div>",
            unsafe_allow_html=True,
        )

if docs:
    doc_columns = st.columns(min(len(docs), 3))
    for column, doc in zip(doc_columns, docs):
        with column:
            st.markdown(
                f"<div class='doc-card'><div class='doc-title'>{doc['doc_name']}</div><div class='doc-meta'>{doc['total_pages']} pages · {doc.get('chunk_count', 0)} chunks · {doc.get('table_count', 0)} table chunks · {doc.get('chart_count', 0)} chart chunks · {doc.get('image_count', 0)} images</div></div>",
                unsafe_allow_html=True,
            )

tab_upload, tab_chat, tab_eval = st.tabs(["Step 1 · Upload PDF", "Step 2 · Ask Questions", "Step 3 · Evaluate"])

with tab_upload:
    st.markdown("<div class='panel-title'>Upload and index your PDF</div><div class='panel-sub'>The app will show which stage it is in: converting, loading the model, embedding pages, storing vectors, then ready.</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload the PDF you want to search and ask questions about.",
    )

    if uploaded_file is not None:
        st.markdown(
            f"<span class='chip'>{uploaded_file.name}</span><span class='chip'>{uploaded_file.size / 1024:.1f} KB</span>",
            unsafe_allow_html=True,
        )

    steps_placeholder = st.empty()
    overall_progress = st.empty()
    status_box = st.empty()

    if st.button("Start indexing", use_container_width=True, disabled=uploaded_file is None):
        step_state = {
            "current_step": 1,
            "progress": {step: 0.0 for step in PROCESS_STEPS},
            "message": "Preparing upload...",
            "overall": 0.0,
        }

        def redraw_progress():
            steps_placeholder.markdown(build_steps_markup(step_state), unsafe_allow_html=True)
            overall_progress.progress(min(max(step_state["overall"], 0.0), 1.0))
            status_box.info(step_state["message"])

        def progress_cb(step: int, total: int, message: str, progress: float | None = None):
            step_state["current_step"] = step
            step_state["message"] = message
            for previous_step in range(1, step):
                step_state["progress"][previous_step] = 1.0
            if progress is not None:
                step_state["progress"][step] = progress
            step_state["overall"] = ((step - 1) + step_state["progress"].get(step, 0.0)) / total
            redraw_progress()

        redraw_progress()

        try:
            pdf_bytes = uploaded_file.read()
            result = pipeline.ingest_pdf(
                source=pdf_bytes,
                doc_name=uploaded_file.name,
                progress_callback=progress_cb,
            )
            st.session_state.active_doc_id = result["doc_id"]
            st.success(
                f"Indexed {result['doc_name']} successfully. {result['num_pages']} pages, {result.get('num_chunks', 0)} chunks, {result.get('num_tables', 0)} table chunks, and {result.get('num_charts', 0)} chart chunks are ready for questions."
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Ingestion failed: {exc}")

with tab_chat:
    st.markdown("<div class='panel-title'>Ask questions about your PDF</div><div class='panel-sub'>Once a PDF is indexed, ask a question and the app will retrieve relevant pages and generate an answer.</div>", unsafe_allow_html=True)

    docs = pipeline.list_documents()
    if not docs:
        st.info("No indexed document yet. Finish Step 1 first.")
    else:
        doc_options = {doc["doc_name"]: doc["doc_id"] for doc in docs}
        default_index = 0
        if st.session_state.active_doc_id:
            for index, doc in enumerate(docs):
                if doc["doc_id"] == st.session_state.active_doc_id:
                    default_index = index
                    break

        selected_doc_name = st.selectbox(
            "Choose the indexed PDF",
            options=list(doc_options.keys()),
            index=default_index,
        )
        selected_doc_id = doc_options[selected_doc_name]

        st.markdown(
            "<div class='mini-note'>Example questions: What are the main ideas? Summarize the report. What tables or figures matter most?</div>",
            unsafe_allow_html=True,
        )

        with st.form("question-form", clear_on_submit=True):
            question = st.text_input(
                "Your question",
                placeholder="Ask about the uploaded PDF...",
            )
            submitted = st.form_submit_button("Get answer", use_container_width=True)

        if submitted and question:
            with st.spinner("Searching the PDF and preparing the answer..."):
                started_at = time.time()
                result = pipeline.query(
                    question=question,
                    top_k=cfg.top_k,
                    doc_id=selected_doc_id,
                )
                latency = time.time() - started_at

            st.session_state.chat_history.append(
                {
                    "question": question,
                    "answer": result.get("answer", ""),
                    "citations": result.get("citations", []),
                    "evidence": result.get("evidence", []),
                    "hits": result.get("hits", []),
                    "latency": latency,
                    "model": result.get("model", ""),
                }
            )

        if st.session_state.chat_history:
            latest = st.session_state.chat_history[-1]
            st.markdown(f"### Question\n{latest['question']}")
            st.markdown(f"<div class='answer-box'>{latest['answer']}</div>", unsafe_allow_html=True)
            st.markdown(
                f"<span class='chip'>Answer time: {latest['latency']:.2f}s</span><span class='chip'>Model: {latest['model']}</span>",
                unsafe_allow_html=True,
            )

            if latest["citations"]:
                st.markdown("**Sources**")
                st.markdown(render_sources(latest["citations"]), unsafe_allow_html=True)

            if latest.get("evidence"):
                with st.expander("Show extracted evidence", expanded=False):
                    for item in latest["evidence"]:
                        st.markdown(
                            f"- {item['text']}  \nPage {item['page_number']} · {item['doc_name']} · {item['section_title']} · {item['chunk_type']}"
                        )

            if latest["hits"]:
                with st.expander("Show retrieved pages", expanded=False):
                    page_columns = st.columns(min(len(latest["hits"]), 3))
                    for column, hit in zip(page_columns, latest["hits"]):
                        with column:
                            if hit.get("image") is not None:
                                st.image(hit["image"], use_container_width=True)
                                st.caption(f"Page {hit['page_number']} · {hit['doc_name']}")

            if st.button("Clear last answer", use_container_width=False):
                st.session_state.chat_history = []
                st.rerun()

with tab_eval:
    st.markdown(
        "<div class='panel-title'>Benchmark evaluation</div>"
        "<div class='panel-sub'>Runs the built-in benchmark queries against the indexed document and reports "
        "retrieval quality metrics: latency, citation coverage, modality hit rate, and top retrieval score.</div>",
        unsafe_allow_html=True,
    )

    eval_docs = pipeline.list_documents()
    if not eval_docs:
        st.info("No indexed document yet. Finish Step 1 first.")
    else:
        eval_doc_options = {doc["doc_name"]: doc["doc_id"] for doc in eval_docs}
        eval_selected_name = st.selectbox(
            "Document to evaluate",
            options=list(eval_doc_options.keys()),
            key="eval_doc_select",
        )
        eval_selected_id = eval_doc_options[eval_selected_name]

        if st.button("Run evaluation", use_container_width=True):
            with st.spinner("Running benchmark queries — this may take a moment..."):
                evaluator = Evaluator(pipeline)
                eval_results = evaluator.run(top_k=cfg.top_k, doc_id=eval_selected_id)
                st.session_state["eval_results"] = eval_results

        if "eval_results" in st.session_state and st.session_state["eval_results"]:
            eval_results = st.session_state["eval_results"]
            summary = Evaluator.summary(eval_results)

            st.markdown("#### Summary metrics")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Avg latency (s)", summary.get("avg_latency_s", "—"))
            m2.metric("Citation coverage", f"{summary.get('citation_coverage_rate', 0):.1f}%")
            m3.metric("Modality hit rate", f"{summary.get('modality_hit_rate', 0):.1f}%")
            m4.metric("Avg top score", summary.get("avg_top_retrieval_score", "—"))

            st.markdown("#### Per-query results")
            rows = [
                {
                    "Query": r.query,
                    "Modality": r.modality,
                    "Pages hit": ", ".join(str(p) for p in r.top_pages),
                    "Top score": r.scores[0] if r.scores else 0.0,
                    "Latency (s)": r.latency_s,
                    "Citations": r.citation_count,
                    "Keyword hit": r.keyword_hit,
                    "Modality hit": r.modality_hit,
                }
                for r in eval_results
            ]
            st.dataframe(rows, use_container_width=True)

            export_payload = {
                "summary": summary,
                "results": [
                    {
                        "query": r.query,
                        "modality": r.modality,
                        "top_pages": r.top_pages,
                        "top_docs": r.top_docs,
                        "scores": r.scores,
                        "top_sections": r.top_sections,
                        "latency_s": r.latency_s,
                        "keyword_hit": r.keyword_hit,
                        "citation_count": r.citation_count,
                        "modality_hit": r.modality_hit,
                        "evidence_count": r.evidence_count,
                    }
                    for r in eval_results
                ],
            }
            st.download_button(
                label="Export results as JSON",
                data=json.dumps(export_payload, indent=2),
                file_name="evaluation_results.json",
                mime="application/json",
                use_container_width=False,
            )
