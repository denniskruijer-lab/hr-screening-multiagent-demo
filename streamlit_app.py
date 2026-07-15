"""A thin chat UI over the supervisor pipeline.

This is presentation only: it wraps build_supervisor() exactly as main.py
does. The agents, tools, and trace are identical either way -- this file
just gives you something to type into and watch respond, alongside the
CLI's plain-text output.

Which CV/vacancy files to use is always decided by this file (the sidebar
selection or an upload), never guessed by the model from whatever the user
happens to type -- that guessing is what used to make the app "crash" on
typos/paraphrases of a candidate's name. The optional text field is for
extra instructions layered on top of a fixed, correct instruction; a bare
click of "Run screening" (no text needed) runs the default behaviour.

Run with: streamlit run streamlit_app.py
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.agent import MaxIterationsExceeded
from src.agents.supervisor import build_supervisor
from src.logging_setup import configure_logging, log_trace
from src.providers.anthropic_client import build_anthropic_client
from src.providers.gemini_adapter import GeminiMessagesClient, build_gemini_sdk_client
from src.rag.store import InMemoryVectorStore, build_corpus
from src.tools.documents import DATA_LOCAL_DIR

load_dotenv()  # loads .env if present; real environment variables always take priority

PRESET_DATASETS = {
    "Fictional sample data": {
        "cv_file": "kandidaat_cv.txt",
        "vacancy_file": "vacature.txt",
        "candidate_name": "Jamie Visser",
    },
}
UPLOAD_OPTION = "Upload your own documents"

st.set_page_config(page_title="HR Screening Multi-Agent Demo", page_icon=":compass:")
st.title("HR Screening Multi-Agent Demo")
st.caption(
    "A supervisor dispatches to a screening agent (Claude) and an independent "
    "calibration agent (Gemini), grounded with RAG over the vacancy, CV, and "
    "talent matrix."
)


def _save_upload(uploaded_file) -> tuple[str, str]:
    """Write an uploaded file into data_local/ and return (filename, content_hash).

    Only the basename is used (Path(...).name strips any path components the
    browser might send), so this can't be used to write outside data_local/.
    The content hash is used to bust the supervisor cache when a re-upload
    under the same filename actually has different content.
    """
    filename = Path(uploaded_file.name).name
    file_bytes = uploaded_file.getvalue()
    (DATA_LOCAL_DIR / filename).write_bytes(file_bytes)
    content_hash = hashlib.md5(file_bytes).hexdigest()[:12]
    return filename, content_hash


def _get_dataset() -> dict | None:
    """Returns {cv_file, vacancy_file, candidate_name, cache_key} or None if
    the user picked "upload" but hasn't supplied everything yet."""
    dataset_label = st.sidebar.radio("Dataset", [*PRESET_DATASETS, UPLOAD_OPTION])

    if dataset_label != UPLOAD_OPTION:
        preset = PRESET_DATASETS[dataset_label]
        return {**preset, "cache_key": dataset_label}

    st.sidebar.caption("Saved into data_local/, which is gitignored and never committed.")
    cv_upload = st.sidebar.file_uploader("CV", type=["txt", "docx", "pdf"])
    vacancy_upload = st.sidebar.file_uploader("Vacancy", type=["txt", "pdf"])
    candidate_name = st.sidebar.text_input("Candidate name")

    if not (cv_upload and vacancy_upload and candidate_name):
        st.sidebar.info("Upload a CV, a vacancy, and enter the candidate's name to continue.")
        return None

    cv_file, cv_hash = _save_upload(cv_upload)
    vacancy_file, vacancy_hash = _save_upload(vacancy_upload)
    return {
        "cv_file": cv_file,
        "vacancy_file": vacancy_file,
        "candidate_name": candidate_name,
        "cache_key": f"{cv_hash}-{vacancy_hash}",
    }


@st.cache_resource
def get_supervisor(cv_file: str, vacancy_file: str, cache_key: str):
    """`cache_key` deliberately has no leading underscore -- Streamlit excludes
    underscore-prefixed args from the cache key hash entirely, which would
    defeat its whole purpose here: busting the cache when uploaded content
    changes under the same filename. The preset dataset passes its own
    label as a stable key instead."""
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


dataset = _get_dataset()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_note = st.text_input(
    "Additional context (optional)",
    placeholder="e.g. focus especially on leadership experience",
    disabled=dataset is None,
)
run_clicked = st.button("Run screening", disabled=dataset is None)

if not dataset:
    st.info("Choose or upload a dataset in the sidebar to enable this.")

if run_clicked and dataset is not None:
    display_text = f"Screen {dataset['candidate_name']}." + (f" {user_note}" if user_note else "")
    st.session_state.messages.append({"role": "user", "content": display_text})
    with st.chat_message("user"):
        st.write(display_text)

    with st.chat_message("assistant"):
        try:
            supervisor, logger = get_supervisor(dataset["cv_file"], dataset["vacancy_file"], dataset["cache_key"])
        except RuntimeError as exc:
            st.error(str(exc))
        else:
            with st.spinner("Running the supervisor and worker agents..."):
                prompt = (
                    f"Screen the candidate named '{dataset['candidate_name']}' using the CV in "
                    f"'{dataset['cv_file']}' against the vacancy in '{dataset['vacancy_file']}'."
                )
                if user_note:
                    prompt += f" Additional instructions from the user: {user_note}"

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
