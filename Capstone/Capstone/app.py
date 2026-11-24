"""Grounded Streamlit study-tip app using a compact GPT-2 last-block checkpoint."""

import re

from local_paths import ROOT
import pandas as pd
import streamlit as st
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "openai-community/gpt2"
DATA_PATH = ROOT / "dataset" / "study_tips.csv"
CHECKPOINT = ROOT / "lab_outputs" / "last_block.safetensors"
MAX_TOPIC_LENGTH = 120
ACTION_WORDS = {
    "calculate", "compare", "draw", "explain", "implement", "mark",
    "practice", "predict", "solve", "test", "trace", "write",
}
STOP_WORDS = {"a", "an", "and", "for", "in", "of", "on", "the", "to", "with"}
PRACTICE_CUES = (
    "small example", "on paper", "step by step", "trace", "predict",
    "practice", "calculate", "compare", "draw", "write", "solve",
    "test one", "work through", "explain each",
)

st.set_page_config(page_title="AI Study Tip Generator", page_icon=":material/school:", layout="centered")


def normalize(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return " ".join(word[:-1] if word.endswith("s") and len(word) > 4 else word for word in words)


def validate_topic(topic):
    topic = " ".join(topic.split())
    if not topic:
        raise ValueError("Enter a study topic before generating a tip.")
    if len(topic) > MAX_TOPIC_LENGTH:
        raise ValueError(f"Keep the topic to {MAX_TOPIC_LENGTH} characters or fewer.")
    if any(ord(char) < 32 for char in topic):
        raise ValueError("The topic contains unsupported control characters.")
    return topic


@st.cache_data
def load_tips():
    return pd.read_csv(DATA_PATH).fillna("")


@st.cache_resource(max_entries=1)
def load_model():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Fine-tuned block missing: {CHECKPOINT}. Run `python train.py`.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.transformer.h[-1].load_state_dict(load_file(str(CHECKPOINT), device="cpu"), strict=True)
    model.to(device).eval()
    return model, tokenizer, str(device)


def draft_with_gpt2(model, tokenizer, topic, device):
    prompt = f"Topic: {topic}\nStudy tip:"
    tokens = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.generate(
            **tokens,
            max_new_tokens=40,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output[0, tokens["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip().splitlines()[0].strip()


def find_curated_tip(topic, tips):
    query = normalize(topic)
    query_words = set(query.split()) - STOP_WORDS
    best = None
    best_score = 0.0
    for row in tips.to_dict(orient="records"):
        full_topic = normalize(row["topic"])
        focused_topic = normalize(row["topic"].split(":")[-1])
        if query in {full_topic, focused_topic}:
            return row, True
        words = set(focused_topic.split()) - STOP_WORDS
        overlap = query_words & words
        score = len(overlap) / max(1, len(words))
        if len(overlap) >= 2 and score > best_score:
            best, best_score = row, score
    if best is not None and best_score >= 0.66:
        return best, False
    return None, False


def candidate_is_actionable(topic, candidate):
    normalized_candidate = normalize(candidate)
    words = set(normalized_candidate.split())
    topic_words = set(normalize(topic).split()) - STOP_WORDS
    meaningful = {word for word in topic_words if len(word) > 2}
    rejected_phrases = (
        "learn more", "check out", "report button", "use the following",
        "runtime environment", "tab of the", "section of the",
    )
    return (
        not any(phrase in normalized_candidate for phrase in rejected_phrases)
        and any(cue in normalized_candidate for cue in PRACTICE_CUES)
        and bool(words & ACTION_WORDS)
        and bool(words & meaningful)
        and 6 <= len(candidate.split()) <= 26
    )


def topic_specific_tip(topic):
    text = normalize(topic)
    if any(word in text for word in ("deep learning", "neural", "attention", "backprop", "transformer", "cnn")):
        return (
            f"Draw a tiny {topic} example and track one input through each step. "
            "Work out the values on paper before checking your code.",
            "Deep-learning practice prompt",
        )
    if any(word in text for word in ("machine learning", "clustering", "regression", "classification", "overfitting", "feature")):
        return (
            f"Use a tiny dataset for {topic}: predict one unseen example, compare it with the known result, "
            "and write down why the model was right or wrong.",
            "Machine-learning practice prompt",
        )
    if any(word in text for word in ("algorithm", "binary search", "sorting", "complexity", "recursion", "dynamic programming", "big o")):
        if text in {"algorithm", "algorithms"}:
            return (
                "Solve one small problem two ways. Count the steps in each solution, then compare how the work grows as the input gets bigger.",
                "Algorithm practice prompt",
            )
        return (
            f"Trace {topic} on five sample values. Record every step, then compare the final result "
            "with your hand-worked answer.",
            "Algorithm practice prompt",
        )
    if any(word in text for word in ("data structure", "stack", "queue", "tree", "graph", "hash", "heap", "array", "linked list")):
        if text in {"data structure", "data structures"}:
            return (
                "Use the same three values in a stack and a queue. Trace insert and remove operations, then compare which value comes out first.",
                "Data-structure practice prompt",
            )
        return (
            f"Draw a small {topic} containing three values. Apply a short sequence of operations "
            "and sketch its state after each operation.",
            "Data-structure practice prompt",
        )
    if any(word in text for word in ("math", "algebra", "calculus", "derivative", "matrix", "probability", "eigenvalue", "equation", "statistics", "integral")):
        return (
            f"Work one small {topic} problem on paper, show every step, then verify the answer "
            "by substitution or a second method.",
            "Mathematics practice prompt",
        )
    if any(word in text for word in ("database", "dbms", "sql", "query", "transaction", "normalization", "join")):
        if "transaction" in text:
            return (
                f"Trace two updates in a {topic} example. Record the data before commit, then check "
                "what remains after rollback.",
                "Database practice prompt",
            )
        return (
            f"Make two tiny tables for {topic}. Predict the query result by hand, run it, "
            "and explain any difference.",
            "Database practice prompt",
        )
    if any(word in text for word in ("network", "tcp", "dns", "subnet", "routing", "http", "osi", "protocol", "packet")):
        if text in {"computer network", "computer networks"}:
            return (
                "Trace opening a webpage: write what DNS, TCP, and HTTP each do, then follow the response back to the browser.",
                "Networking practice prompt",
            )
        return (
            f"Trace one {topic} message from sender to receiver. Label what happens at each step "
            "and predict the response.",
            "Networking practice prompt",
        )
    if any(word in text for word in ("operating system", "process", "thread", "scheduling", "deadlock", "semaphore", "mutex", "memory management")):
        if text in {"operating system", "operating systems"}:
            return (
                "Draw a process moving through ready, running, and blocked states. Trace one context switch and label the event that caused each transition.",
                "Operating-systems practice prompt",
            )
        return (
            f"Draw the states or resources involved in {topic}. Trace one process through each "
            "transition and mark what could block it.",
            "Operating-systems practice prompt",
        )
    if any(word in text for word in ("python", "java", "programming", "code", "function", "loop", "class", "object", "exception", "syntax", "debug")):
        return (
            f"Write a tiny {topic} example. Predict its output before running it, then test one "
            "edge case and explain the result.",
            "Programming practice prompt",
        )
    return (
        f"Make a one-question self-test about {topic}: explain or solve a small example from memory, "
        "check it against your notes, and rewrite the step you missed.",
        "Guided study prompt",
    )


def choose_useful_tip(topic, draft, tips):
    matched, exact = find_curated_tip(topic, tips)
    if matched is not None:
        source = f"Curated example: {matched['topic']}" if exact else f"Related course example: {matched['topic']}"
        return matched["study_tip"], source
    if candidate_is_actionable(topic, draft):
        return draft, "GPT-2 draft passed the topic and action check"
    return topic_specific_tip(topic)


st.title("AI Study Tip Generator", icon=":material/school:")
st.write("Enter a course topic for one concise, practical study action.")

with st.container(border=True):
    with st.form("study_tip_form", border=False):
        topic_input = st.text_input("Study topic", placeholder="For example: Python Loops", max_chars=160)
        submitted = st.form_submit_button("Generate study tip", type="primary", icon=":material/auto_awesome:")

if submitted:
    try:
        topic = validate_topic(topic_input)
    except ValueError as error:
        st.warning(str(error), icon=":material/info:")
    else:
        try:
            tips = load_tips()
            with st.spinner("Checking course examples and preparing a study action..."):
                model, tokenizer, device = load_model()
                draft = draft_with_gpt2(model, tokenizer, topic, device)
                tip, source = choose_useful_tip(topic, draft, tips)
        except (OSError, RuntimeError, ValueError) as error:
            st.error(f"Could not generate a study tip: {error}", icon=":material/error:")
        else:
            st.subheader("Your study tip")
            with st.container(border=True):
                st.write(tip)
            st.caption(f"{source} · GPT-2 fine-tuned last block · {device}")

st.subheader("How it works")
st.write(
    "GPT-2 drafts a suggestion, but the app shows it only when it includes a topic match and a "
    "clear practice action. Otherwise, it uses a curated tip or a subject-specific practice prompt."
)
st.caption(
    "Generated suggestions may still be incomplete or inaccurate. Check them against course "
    "materials and ask an instructor when unsure."
)
