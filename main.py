import streamlit as st
import anthropic
from prediction_helper import predict

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="RiskRadar Finance: Credit Risk Modelling", page_icon="📊")
st.title("RiskRadar Finance: Credit Risk Modelling")

# ── Input form ───────────────────────────────────────────────────────────────
row1 = st.columns(3)
row2 = st.columns(3)
row3 = st.columns(3)
row4 = st.columns(3)

with row1[0]:
    age = st.number_input('Age', min_value=18, step=1, max_value=100, value=28)
with row1[1]:
    income = st.number_input('Income', min_value=0, value=1200000)
with row1[2]:
    loan_amount = st.number_input('Loan Amount', min_value=0, value=2560000)

loan_to_income_ratio = loan_amount / income if income > 0 else 0
with row2[0]:
    st.text("Loan to Income Ratio:")
    st.text(f"{loan_to_income_ratio:.2f}")
with row2[1]:
    loan_tenure_months = st.number_input('Loan Tenure (months)', min_value=0, step=1, value=36)
with row2[2]:
    avg_dpd_per_delinquency = st.number_input('Avg DPD', min_value=0, value=20)

with row3[0]:
    delinquency_ratio = st.number_input('Delinquency Ratio', min_value=0, max_value=100, step=1, value=30)
with row3[1]:
    credit_utilization_ratio = st.number_input('Credit Utilization Ratio', min_value=0, max_value=100, step=1, value=30)
with row3[2]:
    num_open_accounts = st.number_input('Open Loan Accounts', min_value=1, max_value=4, step=1, value=2)

with row4[0]:
    residence_type = st.selectbox('Residence Type', ['Owned', 'Rented', 'Mortgage'])
with row4[1]:
    loan_purpose = st.selectbox('Loan Purpose', ['Education', 'Home', 'Auto', 'Personal'])
with row4[2]:
    loan_type = st.selectbox('Loan Type', ['Unsecured', 'Secured'])

# ── Helper: build AI explanation using Claude ─────────────────────────────────
def get_ai_explanation(probability, credit_score, rating,
                        age, income, loan_amount, loan_tenure_months,
                        avg_dpd_per_delinquency, delinquency_ratio,
                        credit_utilization_ratio, num_open_accounts,
                        residence_type, loan_purpose, loan_type,
                        loan_to_income_ratio):

    decision = "APPROVED" if rating in ["Good", "Excellent"] else "REJECTED / HIGH RISK"

    prompt = f"""
You are a credit risk analyst at RiskRadar Finance. A loan application was evaluated by our ML model.
Explain the result clearly to a bank officer in 4–6 bullet points. Be specific — mention actual numbers.
Use plain English, no jargon.

=== APPLICATION SUMMARY ===
Age: {age}
Income: ₹{income:,}
Loan Amount: ₹{loan_amount:,}
Loan to Income Ratio: {loan_to_income_ratio:.2f}
Loan Tenure: {loan_tenure_months} months
Loan Purpose: {loan_purpose}
Loan Type: {loan_type}
Residence Type: {residence_type}
Avg Days Past Due (DPD): {avg_dpd_per_delinquency}
Delinquency Ratio: {delinquency_ratio}%
Credit Utilization Ratio: {credit_utilization_ratio}%
Open Loan Accounts: {num_open_accounts}

=== MODEL OUTPUT ===
Default Probability: {probability:.2%}
Credit Score: {credit_score}
Rating: {rating}
Decision: {decision}

Respond with:
1. One sentence stating the decision and overall risk level.
2. 3–5 bullet points explaining the KEY factors that drove this decision (good and bad).
3. One sentence of actionable advice for the applicant (what they could improve or maintain).

Keep total response under 200 words.
"""

    client = anthropic.Anthropic()

    with st.spinner("🤖 Generating AI explanation..."):
        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            explanation_placeholder = st.empty()
            full_text = ""
            for text in stream.text_stream:
                full_text += text
                explanation_placeholder.markdown(full_text)

    return full_text


# ── Calculate Risk button ─────────────────────────────────────────────────────
if st.button('Calculate Risk'):
    probability, credit_score, rating = predict(
        age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
        delinquency_ratio, credit_utilization_ratio, num_open_accounts,
        residence_type, loan_purpose, loan_type
    )

    # ── Result display ────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Default Probability", f"{probability:.2%}")
    with col2:
        st.metric("Credit Score", credit_score)
    with col3:
        color_map = {"Poor": "🔴", "Average": "🟡", "Good": "🟢", "Excellent": "🟢"}
        icon = color_map.get(rating, "⚪")
        st.metric("Rating", f"{icon} {rating}")

    # ── Decision banner ───────────────────────────────────────────────────────
    if rating in ["Good", "Excellent"]:
        st.success("✅ **LOAN APPROVED** — Applicant has a good credit profile.")
    elif rating == "Average":
        st.warning("⚠️ **CONDITIONAL** — Moderate risk. Manual review recommended.")
    else:
        st.error("❌ **HIGH RISK** — Loan not recommended based on current profile.")

    # ── AI Explanation ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🤖 AI Explanation")
    st.caption("Why did the model give this result?")

    get_ai_explanation(
        probability, credit_score, rating,
        age, income, loan_amount, loan_tenure_months,
        avg_dpd_per_delinquency, delinquency_ratio,
        credit_utilization_ratio, num_open_accounts,
        residence_type, loan_purpose, loan_type,
        loan_to_income_ratio
    )
