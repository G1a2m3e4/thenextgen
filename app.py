# app.py
import streamlit as st
import requests
import pandas as pd
import json
import time

# -----------------------
# CONFIG - replace tokens/creds here
# -----------------------
# Hugging Face Inference API token (preferred)
HF_TOKEN = "replace this text with API key"  # <-- replace with your token or leave empty

# Optional: IBM Watson Assistant credentials (if you want to use Watson instead)
IBM_API_KEY = ""        # <-- fill if you want to use IBM Watson
IBM_URL = ""            # <-- Watson base url
IBM_ASSISTANT_ID = ""   # <-- watson assistant id

# Model choices
GRANITE_MODEL = "ibm-granite/granite-13b-instruct"   # often restricted
FALLBACK_MODEL = "distilgpt2"                       # open small model for testing

# -----------------------
# Utilities
# -----------------------
def hf_query(model: str, prompt: str, max_tokens: int = 200):
    """Call Hugging Face Inference API for a given model. Returns string or raises."""
    if not HF_TOKEN:
        raise RuntimeError("No Hugging Face token configured.")
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_tokens}}
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if resp.status_code == 200:
        data = resp.json()
        # Many models return list with generated_text; handle a few shapes
        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        # sometimes API returns a dict or string
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
        # fallback: convert to string
        return json.dumps(data)
    else:
        raise RuntimeError(f"Hugging Face API error {resp.status_code}: {resp.text}")

# Simple intent matcher (very lightweight)
def detect_intent(text: str):
    t = text.lower()
    if any(w in t for w in ["budget", "income", "expenses", "spend", "savings plan"]):
        return "budget"
    if any(w in t for w in ["tax", "taxes", "filing", "deduction"]):
        return "tax"
    if any(w in t for w in ["invest", "investment", "stocks", "mutual", "etf", "portfolio"]):
        return "investment"
    if any(w in t for w in ["insight", "spending", "optimize", "save more"]):
        return "insight"
    return "general"

# Local rule-based budget summary (structured)
def generate_budget_summary_from_input(raw: str):
    """
    Expect the user to paste or type something like:
    'income: 50000; rent:15000; food:5000; transport:2000; entertainment:2000; savings:5000'
    We'll parse simple key:value pairs.
    """
    items = {}
    parts = [p.strip() for p in raw.replace(",", ";").split(";") if p.strip()]
    for p in parts:
        if ":" in p:
            k, v = p.split(":", 1)
            try:
                items[k.strip()] = float(v.strip())
            except:
                # ignore non-numeric
                pass
    if not items:
        return None  # couldn't parse

    df = pd.DataFrame(list(items.items()), columns=["category", "amount"])
    income = df[df["category"].str.contains("income", case=False)]
    total_income = float(income["amount"].sum()) if not income.empty else 0.0
    expenses = df[~df["category"].str.contains("income", case=False)]
    total_expenses = float(expenses["amount"].sum()) if not expenses.empty else 0.0
    savings = total_income - total_expenses
    summary = {
        "df": df,
        "total_income": total_income,
        "total_expenses": total_expenses,
        "savings": savings
    }
    return summary

# Spending insights (simple heuristics)
def spending_insights_from_df(df: pd.DataFrame):
    insights = []
    total = df["amount"].sum()
    if total == 0:
        return ["No numeric spending data provided."]
    # find largest categories
    top = df.sort_values("amount", ascending=False).head(3)
    for _, row in top.iterrows():
        pct = row["amount"] / total * 100
        insights.append(f"{row['category']}: ₹{row['amount']:.2f} ({pct:.0f}% of provided items)")
    # heuristic tips
    if any(df["category"].str.contains("entertain", case=False)):
        insights.append("Consider reducing entertainment by 10–20% to increase monthly savings.")
    if any(df["category"].str.contains("food", case=False)) and total>0:
        insights.append("Meal planning and cooking at home can reduce food costs significantly.")
    return insights

# Build prompt for LLM
def build_prompt(user_type: str, user_question: str, mode: str="qa"):
    tone = "casual and simple" if user_type=="student" else "professional and concise"
    preamble = (
        f"You are a helpful personal finance assistant. Use a {tone} tone. "
        "Tasks you should perform when relevant: provide personalized advice on savings, taxes, investments; "
        "generate a simple budget summary when the user provides income/expenses; give practical spending optimization tips. "
    )
    if mode == "qa":
        prompt = f"{preamble}\nUser ({user_type}) asks: {user_question}\nAnswer concisely with actionable steps."
    else:
        prompt = f"{preamble}\n{user_question}"
    return prompt

# Safe generate function with fallback
def generate_reply(user_type: str, user_text: str):
    intent = detect_intent(user_text)
    # If user provided a budget-like input, try local structured parsing first
    if intent == "budget":
        parsed = generate_budget_summary_from_input(user_text)
        if parsed:
            df = parsed["df"]
            tot_inc = parsed["total_income"]
            tot_exp = parsed["total_expenses"]
            sav = parsed["savings"]
            table = df.copy()
            # show structured table later; produce a textual summary now
            text = f"Budget summary — Income: ₹{tot_inc:.2f}, Expenses: ₹{tot_exp:.2f}, Net savings: ₹{sav:.2f}."
            insights = spending_insights_from_df(table)
            ins_text = " Insights: " + " | ".join(insights) if insights else ""
            return {"text": text + ins_text, "table": table}
    # else use LLM (Hugging Face preferred)
    prompt = build_prompt(user_type, user_text)
    # Try Granite first (if token present), else fallback
    if HF_TOKEN:
        try:
            # try restricted granite model first
            try:
                resp = hf_query(GRANITE_MODEL, prompt)
            except Exception as e_gran:
                # if granite fails, try fallback model
                resp = hf_query(FALLBACK_MODEL, prompt)
            return {"text": resp, "table": None}
        except Exception as e:
            # API failed; fall back to a simple rule-based reply
            return {"text": f"(API failed: {str(e)})\nHere's a basic answer: I recommend tracking income/expenses, saving at least 10% of income, and using low-cost index funds for long-term investing.", "table": None}
    else:
        # No HF token — local fallback answer
        fallback_text = ("I don't have an AI token configured. Quick guidance: "
                         "Track your monthly income and expenses, aim to save 10-20% if possible, "
                         "use an emergency fund of 3-6 months, and invest in diversified low-cost funds.")
        return {"text": fallback_text, "table": None}

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Personal Finance Chatbot", layout="wide")
st.title("💰 Personal Finance Chatbot")

# Left column: controls / chat
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("Ask & Profile")
    user_type = st.radio("Select your profile:", ["student", "professional"], key="profile")
    user_input = st.text_input("Ask me about savings, investments, taxes, or paste simple budget (like income:50000; rent:15000; food:5000):", key="chat_input")
    send = st.button("Send", key="send_btn")

    # show some helpful instructions
    with st.expander("Tips for better answers"):
        st.markdown(
            "- For budget summaries paste `income:..., rent:..., food:...` etc.\n"
            "- Ask direct questions like: `How much should I save monthly on 50k income?`\n"
            "- For investment advice, ask for timeframe and risk tolerance."
        )

with col2:
    st.subheader("Conversation")
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # when user clicks send -> generate reply
    if send:
        if user_input and user_input.strip():
            st.session_state["messages"].append(("user", user_input))
            with st.spinner("Generating answer..."):
                result = generate_reply(user_type, user_input.strip())
                # result = {"text":..., "table":DataFrame | None}
                text = result.get("text")
                table = result.get("table")
                # Append bot reply
                st.session_state["messages"].append(("bot", text))
                if table is not None:
                    # store table as JSON string so session_state is serializable
                    st.session_state.setdefault("tables", []).append(table.to_dict(orient="records"))
        else:
            st.warning("Please type a question or paste budget details.")

    # Display messages and any saved tables
    for idx, (role, msg) in enumerate(st.session_state.get("messages", [])):
        if role == "user":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Bot:** {msg}")

    # If we saved tables, show the latest one
    if st.session_state.get("tables"):
        last_table = st.session_state["tables"][-1]
        df_show = pd.DataFrame(last_table)
        st.markdown("**Budget details (parsed):**")
        st.dataframe(df_show)

# -----------------------
# Footer: API tests & debug (collapsed)
# -----------------------
with st.expander("Diagnostics & API Test (advanced)", expanded=False):
    st.markdown("Use this area to test your Hugging Face token quickly.")
    if st.button("Test Hugging Face token"):
        if not HF_TOKEN:
            st.error("HF_TOKEN not set in the script.")
        else:
            test_prompt = "Hello! This is a token test."
            try:
                out = hf_query(FALLBACK_MODEL, test_prompt)
                st.success("Token works. Sample response (truncated):")
                st.write(out[:500])
            except Exception as e:
                st.error(f"Token test failed: {e}")
    st.markdown("Tip: If your HF token returns 401 or 403, re-generate a token at https://huggingface.co/settings/tokens (Read + Inference scopes).")

