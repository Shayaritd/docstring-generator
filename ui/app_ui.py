"""
Docstring Generator - Advanced UI
Features: Quality Scoring, Confidence Flagging, Hallucination Check, Diff Mode, Self-Correction
"""

import streamlit as st
import time
from ui_helpers import (
    call_generate_api, check_api_health, EXAMPLE_FUNCTIONS,
    format_latency, format_quality_score, get_confidence_level,
    insert_docstring_into_function
)

st.set_page_config(page_title="Docstring Generator Pro", page_icon="📝", layout="wide")

# --- Session State ---
if "code_input" not in st.session_state:
    st.session_state.code_input = EXAMPLE_FUNCTIONS["Simple: add two numbers"]
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "generation_history" not in st.session_state:
    st.session_state.generation_history = []


# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Settings")

    api_url = st.text_input("API URL", value="http://localhost:8000")

    health = check_api_health(api_url)
    if health["reachable"] and health["model_loaded"]:
        st.success("✅ API connected — model loaded")
    elif health["reachable"]:
        st.warning("⚠️ API reachable but model not loaded")
    else:
        st.error("❌ API unreachable")

    st.divider()

    st.subheader("📝 Generation Settings")

    max_length = st.slider("Max tokens", 16, 256, 80, step=8)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, step=0.1)

    st.divider()

    st.subheader("🎯 Style Preset")
    style_preset = st.selectbox("Style", ["Google Style", "NumPy Style", "Concise Internal"])

    st.divider()

    st.subheader("🔍 Quality Checks")
    enable_quality_scoring = st.checkbox("Quality Scoring", value=True)
    enable_hallucination_check = st.checkbox("Hallucination Check", value=True)
    enable_confidence_flagging = st.checkbox("Confidence Flagging", value=True)
    enable_diff_mode = st.checkbox("Diff Mode", value=True)
    enable_self_correction = st.checkbox("Self-Correction", value=True)

    st.divider()

    st.subheader("📊 Stats")
    if st.session_state.generation_history:
        st.metric("Total Generations", len(st.session_state.generation_history))
        avg_time = sum(h["latency_ms"] for h in st.session_state.generation_history) / len(st.session_state.generation_history)
        st.metric("Avg Time", format_latency(avg_time))


# --- Main Area ---
st.title("📝 Docstring Generator Pro")
st.caption("Generate Google-style Python docstrings with quality scoring, confidence flagging, and self-correction")

left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("📄 Input Function")

    selected_example = st.selectbox("Load Example", ["Select..."] + list(EXAMPLE_FUNCTIONS.keys()))
    if selected_example != "Select...":
        st.session_state.code_input = EXAMPLE_FUNCTIONS[selected_example]

    current_code = st.text_area(
        "Python function",
        value=st.session_state.code_input,
        height=250,
        label_visibility="collapsed"
    )
    st.session_state.code_input = current_code

    generate_clicked = st.button("🚀 Generate Docstring", type="primary", use_container_width=True)

with right_col:
    st.subheader("📝 Generated Docstring")

    if generate_clicked:
        if not current_code.strip():
            st.error("Please enter a function first.")
        else:
            with st.spinner("Generating docstring..."):
                start_time = time.time()
                result = call_generate_api(
                    api_url,
                    current_code,
                    max_length,
                    temperature,
                    timeout=305.0
                )
                total_time = (time.time() - start_time) * 1000

            if result["success"]:
                # Add to history
                history_entry = {
                    "code": current_code,
                    "docstring": result["docstring"],
                    "latency_ms": result.get("latency_ms", total_time),
                    "model": result.get("model", "unknown")
                }
                st.session_state.generation_history.append(history_entry)

                st.success(f"✅ Generated in {format_latency(result.get('latency_ms', total_time))}")

                # Display docstring
                st.code(result["docstring"], language="python")

                # Quality Scores
                if enable_quality_scoring:
                    st.subheader("📊 Quality Scores")
                    quality = result.get("quality", {})
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", f"{quality.get('accuracy', 4.0):.1f}/5")
                    col2.metric("Completeness", f"{quality.get('completeness', 4.0):.1f}/5")
                    col3.metric("Clarity", f"{quality.get('clarity', 4.0):.1f}/5")
                    col4.metric("Conciseness", f"{quality.get('conciseness', 4.0):.1f}/5")

                    avg_score = sum(quality.values()) / 4 if quality else 0
                    st.progress(avg_score / 5, text=f"Overall Quality: {avg_score:.1f}/5")

                # Confidence Flagging
                if enable_confidence_flagging:
                    confidence = result.get("confidence", 85)
                    level, color = get_confidence_level(confidence)
                    st.markdown(f"**Confidence:** {confidence}% ({level})")
                    if level == "Low":
                        st.warning("⚠️ Low confidence - please review this docstring manually")
                    elif level == "Medium":
                        st.info("ℹ️ Medium confidence - review if critical")

                # Hallucination Check
                if enable_hallucination_check:
                    hallucinations = result.get("hallucinations", [])
                    if hallucinations:
                        st.warning(f"⚠️ {len(hallucinations)} potential hallucination(s) detected")
                        for h in hallucinations:
                            st.caption(f"• {h}")
                    else:
                        st.success("✅ No hallucinations detected")

                # Self-Correction
                if enable_self_correction and result.get("corrected", False):
                    st.info("🔄 Self-correction applied - missing parameters added")

                # Model info
                st.caption(f"Model: {result.get('model', 'unknown')}")

            else:
                st.error(f"❌ {result.get('error', 'Generation failed')}")

    elif st.session_state.last_result:
        result = st.session_state.last_result

    st.session_state.last_result = result

# --- Diff Mode ---
if enable_diff_mode and st.session_state.last_result and st.session_state.last_result.get("success", False):
    st.divider()
    st.subheader("📊 Before / After Comparison")

    before_code = st.session_state.code_input
    after_code = insert_docstring_into_function(before_code, st.session_state.last_result["docstring"])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Before**")
        st.code(before_code, language="python")
    with col2:
        st.markdown("**After**")
        st.code(after_code, language="python")
