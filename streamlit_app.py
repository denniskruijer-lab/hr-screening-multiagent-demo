"""A thin chat UI over the supervisor pipeline.

This is presentation only: it wraps build_supervisor() exactly as main.py
does. The agents, tools, and trace are identical either way -- this file
just gives you something to type into and watch respond, alongside the
CLI's plain-text output.

Run with: streamlit run streamlit_app.py
"""
from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from src.agent import MaxIterationsExceeded
from src.agents.supervisor import build_supervisor
from src.logging_setup import configure_logging, log_trace
from src.providers.anthropic_client import build_anthropic_client
from src.providers.gemini_adapter import GeminiMessagesClient, build_gemini_sdk_client
from src.rag.store import InMemoryVectorStore, build_corpus

load_dotenv()  # loads .env if present; real environment variables always take priority

DATASETS = {
    "Fictional sample data": {
        "cv_file": "kandidaat_cv.txt",
        "vacancy_file": "vacature.txt",
        "candidate_name": "Jamie Visser",
    },
    "Real data (Dennis vs. PAQT)": {
        "cv_file": "cv_dennis_kruijer.docx",
        "vacancy_file": "vacature_paqt.txt",
        "candidate_name": "Dennis Kruijer",
    },
}

st.set_page_config(page_title="HR Screening Multi-Agent Demo", page_icon=":compass:")
st.title("HR Screening Multi-Agent Demo")
st.caption(
    "A supervisor dispatches to a screening agent (Claude) and an independent "
    "calibration agent (Gemini), grounded with RAG over the vacancy, CV, and "
    "talent matrix."
)

dataset_label = st.sidebar.radio("Dataset", list(DATASETS.keys()))
dataset = DATASETS[dataset_label]
if dataset_label.startswith("Real data"):
    st.sidebar.caption("Reads from data_local/, which is gitignored and never committed.")


@st.cache_resource
def get_supervisor(cv_file: str, vacancy_file: str):
    logger = configure_logging()
    anthropic_client = build_anthropic_client()
    gemini_sdk = build_gemini_sdk_client()
    gemini_client = GeminiMessagesClient(gemini_sdk)

    corpus = build_corpus([cv_file, vacancy_file, "talent_matrix.json"])
    vector_store = InMemoryVectorStore(corpus, gemini_sdk)

    supervisor = build_supervisor(
        supervisor_client=gemini_client,
        screening_client=anthropic_client,
        calibration_client=gemini_client,
        vector_store=vector_store,
    )
    return supervisor, logger


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

default_prompt = (
    f"Screen the candidate named '{dataset['candidate_name']}' using the CV in "
    f"'{dataset['cv_file']}' against the vacancy in '{dataset['vacancy_file']}'."
)
prompt = st.chat_input(default_prompt)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        try:
            supervisor, logger = get_supervisor(dataset["cv_file"], dataset["vacancy_file"])
        except RuntimeError as exc:
            st.error(str(exc))
        else:
            with st.spinner("Running the supervisor and worker agents..."):
                try:
                    result = supervisor.run(prompt)
                except MaxIterationsExceeded as exc:
                    logger.critical("Supervisor aborted: %s", exc)
                    st.error(f"Supervisor aborted: {exc}")
                else:
                    log_trace(logger, result.trace)
                    st.write(result.final_text)
                    st.session_state.messages.append({"role": "assistant", "content": result.final_text})

                    with st.expander("Tool-call trace (what actually happened under the hood)"):
                        for step in result.trace:
                            status = "ok" if not step["is_error"] else "ERROR"
                            st.write(f"[{step['iteration']}] `{step['tool']}` -> {status} -- input: {step['input']}")
