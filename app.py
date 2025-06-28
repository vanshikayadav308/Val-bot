
import streamlit as st
import openai
import anthropic
from dotenv import load_dotenv
import os
from textblob import TextBlob
import plotly.graph_objects as go
import pandas as pd
import io

# === Load API keys === #
load_dotenv()
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# === Page Config === #
st.set_page_config(page_title="valBot - LLM Comparison Dashboard", layout="wide", page_icon="🤖")
st.markdown("""
    <style>
        .main {background-color: #0f1117; color: white;}
        .stTextArea textarea {font-size: 16px;}
        .stButton button {background-color: #00c2ff; color: white; border-radius: 8px;}
    </style>
""", unsafe_allow_html=True)

st.title("🤖 valBot: LLM Comparison Dashboard")
st.markdown("Enter a prompt (e.g., resume summary, candidate review, etc.) and compare how different LLMs respond.")

prompt = st.text_area("Enter your prompt here:", height=180, placeholder="E.g., Summarize this resume for a data analyst role...")

def evaluate_response(text, keywords):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    tone_score = max(min((polarity + 1) * 2.5, 10), 0)
    word_count = len(text.split())
    conciseness = max(0, min(10, 1000 / word_count)) if word_count > 0 else 0
    clarity = 10 - min(blob.sentiment.subjectivity * 10, 10)
    keyword_hits = sum(1 for k in keywords if k.lower() in text.lower())
    relevance = round(min(10, (keyword_hits / len(keywords)) * 10), 2) if keywords else 0
    insight = 6 + (clarity + tone_score) / 8
    return {
        "Clarity": round(clarity, 2),
        "Relevance": round(relevance, 2),
        "Tone": round(tone_score, 2),
        "Insight": round(insight, 2),
        "Conciseness": round(conciseness, 2)
    }

def render_radar_chart(scores, model_name):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=list(scores.values()),
        theta=list(scores.keys()),
        fill='toself',
        name=model_name
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="#0f1117",
        font_color="white",
        title=f"{model_name} Evaluation"
    )
    st.plotly_chart(fig, use_container_width=True)

if st.button("Generate Responses"):
    if not prompt.strip():
        st.warning("⚠️ Please enter a valid prompt.")
    else:
        with st.spinner("Talking to the LLMs..."):
            keywords = [w.strip('.,!?').lower() for w in prompt.split() if len(w) > 3]
            try:
                gpt_response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=1000
                )
                gpt_output = gpt_response.choices[0].message.content
                gpt_scores = evaluate_response(gpt_output, keywords)
            except Exception as e:
                gpt_output = f"❌ GPT-4 Error: {e}"
                gpt_scores = {}

            try:
                claude_response = claude_client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=1000,
                    temperature=0.7,
                    messages=[{"role": "user", "content": prompt}]
                )
                claude_output = claude_response.content[0].text
                claude_scores = evaluate_response(claude_output, keywords)
            except Exception as e:
                claude_output = f"❌ Claude Error: {e}"
                claude_scores = {}

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🧠 GPT-4")
            st.markdown(f"""
            <div style='background-color:#1a1a1a; padding:15px; border-radius:10px;'>
                {gpt_output.replace('\n', '<br>')}
            </div>
            """, unsafe_allow_html=True)
            if gpt_scores:
                render_radar_chart(gpt_scores, "GPT-4")

        with col2:
            st.markdown("### 🤖 Claude 3")
            st.markdown(f"""
            <div style='background-color:#1a1a1a; padding:15px; border-radius:10px;'>
                {claude_output.replace('\n', '<br>')}
            </div>
            """, unsafe_allow_html=True)
            if claude_scores:
                render_radar_chart(claude_scores, "Claude 3")

        # === Voting System === #
        st.markdown("---")
        st.subheader("🗳️ Vote: Which model gave the better response?")
        vote = st.radio("Pick one:", ["GPT-4", "Claude 3"])
        if st.button("Submit Vote"):
            st.success(f"✅ You voted for {vote}!")

        # === Export to CSV === #
        if gpt_scores and claude_scores:
            data = [
                {"Prompt": prompt, "Model": "GPT-4", "Response": gpt_output, **gpt_scores},
                {"Prompt": prompt, "Model": "Claude 3", "Response": claude_output, **claude_scores},
            ]
            df = pd.DataFrame(data)
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download CSV of Results",
                data=csv_buffer.getvalue(),
                file_name="valbot_comparison.csv",
                mime="text/csv"
            )