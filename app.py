import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

st.set_page_config(page_title="Riversol Conversion Dashboard", layout="wide")

# ─────────────────────────────────────────────────────────────
# 1. MOCK DATA GENERATION
# ─────────────────────────────────────────────────────────────

FIRST_NAMES = [
    "Emily","Sarah","Jessica","Ashley","Amanda","Megan","Jennifer","Stephanie",
    "Nicole","Lauren","Rachel","Samantha","Hannah","Brittany","Elizabeth",
    "Michael","David","James","Robert","John","Daniel","Matthew","Andrew",
    "Joshua","Christopher","Kevin","Brian","Mark","Steven","Paul",
    "Marie","Chantal","Isabelle","Sophie","Natalie","Catherine","Melissa",
    "Wei","Mei","Linda","Grace","Angela","Karen","Lisa","Sandra","Donna",
]
LAST_NAMES = [
    "Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis",
    "Wilson","Anderson","Taylor","Thomas","Jackson","White","Harris",
    "Martin","Thompson","Lee","Tremblay","Roy","Gagnon","Côté","Bouchard",
    "Chen","Wang","Kim","Nguyen","Patel","Singh","Kumar","Zhang",
    "MacLeod","MacDonald","Campbell","Fraser","Stewart","Murray","Reid",
]

PRODUCT_KITS = [
    "Rosacea Relief Kit",
    "Anti-Aging Starter Kit",
    "Sensitive Skin 3-Step Kit",
    "Dryness Rescue Kit",
    "Daily Balance Kit",
]

PROVINCES = ["BC","AB","ON","QC","MB","SK","NS","NB"]
CLIMATES   = {"BC":"Temperate","AB":"Dry","ON":"Humid","QC":"Cold",
              "MB":"Cold","SK":"Dry","NS":"Humid","NB":"Humid"}

SKIN_CONCERNS = ["Rosacea","Anti-aging","Sensitivity","Dryness","Oily/Acne"]
SKIN_TYPES    = ["Dry","Oily","Combination","Normal","Sensitive"]

CONCERN_TO_KIT = {
    "Rosacea":    "Rosacea Relief Kit",
    "Anti-aging": "Anti-Aging Starter Kit",
    "Sensitivity":"Sensitive Skin 3-Step Kit",
    "Dryness":    "Dryness Rescue Kit",
    "Oily/Acne":  "Daily Balance Kit",
}

# ── Log-odds weights (used by sigmoid → no clipping needed) ──────────────────
# These are additive contributions to log-odds, not raw probabilities.
# sigmoid(intercept + sum_of_weights) gives the final probability.
# This ensures email_ctr (and every other feature) has a strictly
# monotone positive relationship with the score — no U-curve artefacts.

INTERCEPT = -2.0          # baseline log-odds ≈ 12% conversion for a "neutral" customer

CONCERN_LOGODDS = {       # how much each concern raises log-odds vs baseline
    "Rosacea":    0.90,   # most motivated — specific medical condition
    "Anti-aging": 0.60,
    "Sensitivity":0.35,
    "Dryness":    0.15,
    "Oily/Acne":  0.00,   # reference category
}
AGE_LOGODDS = {           # 40-49 highest — disposable income + motivated
    "18-29": 0.00,
    "30-39": 0.18,
    "40-49": 0.45,
    "50-59": 0.32,
    "60+":   0.22,
}
AGE_RANGES = ["18-29","30-39","40-49","50-59","60+"]

# Feature weights in log-odds space
W_EMAIL_CTR   = 2.0   # email_ctr ∈ [0, 0.5]  → max +1.0 log-odds
W_SITE_VISITS = 0.08  # visits ∈ [1, 15]       → max +1.2 log-odds
W_PAGES       = 0.12  # pages_viewed ∈ [0, 9]  → max +1.08 log-odds

def sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))))


@st.cache_data
def generate_customers(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    today = datetime.today().date()

    records = []
    for i in range(n):
        first  = rng.choice(FIRST_NAMES)
        last   = rng.choice(LAST_NAMES)
        concern= rng.choice(SKIN_CONCERNS)
        skin   = rng.choice(SKIN_TYPES)
        age    = rng.choice(AGE_RANGES, p=[0.15,0.25,0.28,0.20,0.12])
        prov   = rng.choice(PROVINCES)
        kit    = CONCERN_TO_KIT[concern]

        trial_start = today - timedelta(days=int(rng.integers(1, 15)))
        days_elapsed = (today - trial_start).days
        days_left    = max(0, 15 - days_elapsed)

        email_open  = float(np.clip(rng.beta(3, 5), 0.05, 0.85))
        email_ctr   = float(np.clip(email_open * rng.uniform(0.10, 0.60), 0.01, 0.50))
        site_visits = int(rng.poisson(3) + 1)
        pages_viewed= int(rng.integers(0, min(site_visits+1, 9)))

        log_odds = (
            INTERCEPT
            + CONCERN_LOGODDS[concern]
            + AGE_LOGODDS[age]
            + W_EMAIL_CTR   * email_ctr
            + W_SITE_VISITS * site_visits
            + W_PAGES       * pages_viewed
            + float(rng.normal(0, 0.25))
        )
        p = sigmoid(log_odds)

        converted = int(rng.random() < p) if days_left == 0 else None

        records.append({
            "customer_id":   f"RS-{10000+i}",
            "name":          f"{first} {last}",
            "province":      prov,
            "age_range":     age,
            "skin_concern":  concern,
            "skin_type":     skin,
            "product_kit":   kit,
            "trial_start":   trial_start.strftime("%Y-%m-%d"),
            "days_elapsed":  days_elapsed,
            "days_left":     days_left,
            "email_open_rate": round(email_open, 3),
            "email_ctr":     round(email_ctr, 3),
            "site_visits":   site_visits,
            "pages_viewed":  pages_viewed,
            "propensity_score": round(p, 3),
            "converted":     converted,   # None = trial still active
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
# 2. COHORT DATA GENERATOR
#    All 500 customers have completed their 15-day window, so
#    every row has a known binary outcome. This is the source of
#    truth for conversion rates across the entire dashboard.
#    In production: replace with a Shopify / Supabase query
#    filtered to orders placed > 15 days ago.
# ─────────────────────────────────────────────────────────────

@st.cache_data
def generate_cohort_data(n: int = 500) -> pd.DataFrame:
    """
    Generates a completed-trial cohort (all converted values are 0 or 1).
    Uses the same log-odds model as generate_customers() for consistency.
    Seed is fixed so the conversion rate is stable across reruns.
    """
    rng = np.random.default_rng(42)  # same seed as generate_customers for consistency
    records = []

    for i in range(n):
        concern      = rng.choice(SKIN_CONCERNS)
        skin         = rng.choice(SKIN_TYPES)
        age          = rng.choice(AGE_RANGES, p=[0.15, 0.25, 0.28, 0.20, 0.12])
        prov         = rng.choice(PROVINCES)
        email_open   = float(np.clip(rng.beta(3, 5), 0.05, 0.85))
        email_ctr    = float(np.clip(email_open * rng.uniform(0.10, 0.60), 0.01, 0.50))
        site_visits  = int(rng.poisson(3) + 1)
        pages_viewed = int(rng.integers(0, min(site_visits + 1, 9)))

        log_odds = (
            INTERCEPT
            + CONCERN_LOGODDS[concern]
            + AGE_LOGODDS[age]
            + W_EMAIL_CTR   * email_ctr
            + W_SITE_VISITS * site_visits
            + W_PAGES       * pages_viewed
            + float(rng.normal(0, 0.25))
        )
        p         = sigmoid(log_odds)
        converted = int(rng.random() < p)   # always resolved — trial ended

        records.append({
            "customer_id":      f"CH-{20000 + i}",
            "age_range":        age,
            "skin_concern":     concern,
            "skin_type":        skin,
            "province":         prov,
            "email_open_rate":  round(email_open, 3),
            "email_ctr":        round(email_ctr, 3),
            "site_visits":      site_visits,
            "pages_viewed":     pages_viewed,
            "propensity_score": round(p, 3),
            "converted":        converted,
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────
# 3. MODEL TRAINING
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def train_model():
    rng = np.random.default_rng(0)
    n   = 2000
    records = []
    for _ in range(n):
        concern     = rng.choice(SKIN_CONCERNS)
        skin        = rng.choice(SKIN_TYPES)
        age         = rng.choice(AGE_RANGES)
        site_visits = int(rng.poisson(3) + 1)
        email_ctr   = float(np.clip(rng.uniform(0.01, 0.50), 0.01, 0.50))
        pages       = int(rng.integers(0, 9))
        p = sigmoid(
            INTERCEPT
            + CONCERN_LOGODDS[concern]
            + AGE_LOGODDS[age]
            + W_EMAIL_CTR   * email_ctr
            + W_SITE_VISITS * site_visits
            + W_PAGES       * pages
            + float(rng.normal(0, 0.25))
        )
        records.append({"skin_concern":concern,"skin_type":skin,"age_range":age,
                        "site_visits":site_visits,"email_ctr":email_ctr,
                        "pages_viewed":pages,"converted":int(rng.random()<p)})

    df = pd.DataFrame(records)
    le_c = LabelEncoder().fit(df["skin_concern"])
    le_s = LabelEncoder().fit(df["skin_type"])
    le_a = LabelEncoder().fit(df["age_range"])

    df["c_enc"] = le_c.transform(df["skin_concern"])
    df["s_enc"] = le_s.transform(df["skin_type"])
    df["a_enc"] = le_a.transform(df["age_range"])

    feats = ["c_enc","s_enc","a_enc","site_visits","email_ctr","pages_viewed"]
    mdl   = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                               eval_metric="logloss", random_state=42)
    mdl.fit(df[feats], df["converted"])
    return mdl, le_c, le_s, le_a

model, le_concern, le_type, le_age = train_model()
df_customers = generate_customers(300)

# Historical cohort: 500 completed trials — source of truth for conversion rates.
# Both the Overview KPI and the Cohort Analysis tab read from this same object
# so the numbers stay consistent across the dashboard.
df_cohort = generate_cohort_data(500)


def score_customer(concern, skin, age, visits, ctr, pages):
    row = pd.DataFrame({
        "c_enc": [le_concern.transform([concern])[0]],
        "s_enc": [le_type.transform([skin])[0]],
        "a_enc": [le_age.transform([age])[0]],
        "site_visits": [visits],
        "email_ctr":   [ctr],
        "pages_viewed":[pages],
    })
    return float(model.predict_proba(row)[0][1])


# ─────────────────────────────────────────────────────────────
# 3. LAYOUT — TABS
# ─────────────────────────────────────────────────────────────

st.title("Riversol — Sample Kit Conversion Dashboard")
st.caption("Propensity scoring for 15-day trial customers · prototype (synthetic data)")

tab_overview, tab_cohort, tab_customers, tab_score, tab_outreach = st.tabs([
    "Overview",
    "Cohort Analysis",
    "Customers",
    "Score",
    "Outreach",
])


# ─────────────────────────────────────────────────────────────
# TAB 1 — ANALYTICS DASHBOARD
# ─────────────────────────────────────────────────────────────

with tab_overview:
    active     = df_customers[df_customers["converted"].isna()]
    high_leads = df_customers[df_customers["propensity_score"] >= 0.60]

    # Conversion rate comes from df_cohort (500 completed trials) —
    # the same dataset used in Cohort Analysis — so both tabs show
    # an identical overall rate.
    hist_conv_rate = df_cohort["converted"].mean()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Trial customers",       len(df_customers))
    m2.metric("Active trials",         len(active))
    m3.metric("Score >= 60%",          len(high_leads))
    m4.metric("Conv. rate (historical)", f"{hist_conv_rate:.1%}",
              help="Based on 500 completed trials — matches Cohort Analysis tab")

    st.divider()
    row1a, row1b = st.columns(2)

    with row1a:
        avg_by_concern = (df_customers.groupby("skin_concern")["propensity_score"]
                          .mean().reset_index().sort_values("propensity_score", ascending=True))
        fig1 = px.bar(avg_by_concern, x="propensity_score", y="skin_concern",
                      orientation="h", title="Mean conversion score by skin concern",
                      labels={"propensity_score":"Mean score","skin_concern":""},
                      color_discrete_sequence=["#4A6FA5"],
                      text=avg_by_concern["propensity_score"].map(lambda x: f"{x:.0%}"))
        fig1.update_traces(textposition="outside")
        fig1.update_layout(height=300, margin=dict(l=0,r=20,t=40,b=0))
        st.plotly_chart(fig1, use_container_width=True)

    with row1b:
        fig2 = px.histogram(df_customers, x="propensity_score", nbins=20,
                            title="Score distribution",
                            labels={"propensity_score":"Conversion score"},
                            color_discrete_sequence=["#4A6FA5"])
        fig2.add_vline(x=0.60, line_dash="dash", line_color="#888",
                       annotation_text="60% threshold")
        fig2.update_layout(height=300, margin=dict(l=0,r=0,t=40,b=0),
                           showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    row2a, row2b = st.columns(2)

    with row2a:
        fig3 = px.scatter(df_customers, x="email_ctr", y="propensity_score",
                          color="skin_concern",
                          hover_data=["name","product_kit","age_range"],
                          title="Email CTR vs conversion score",
                          labels={"email_ctr":"Email CTR",
                                  "propensity_score":"Conversion score"},
                          opacity=0.65)
        fig3.update_layout(height=340, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig3, use_container_width=True)

    with row2b:
        skin_counts = df_customers["skin_type"].value_counts().reset_index()
        skin_counts.columns = ["skin_type","count"]
        fig4 = px.pie(skin_counts, names="skin_type", values="count",
                      hole=0.45, title="Customers by skin type",
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        fig4.update_layout(height=340, margin=dict(l=0,r=0,t=40,b=20))
        st.plotly_chart(fig4, use_container_width=True)

    row3a, row3b = st.columns(2)

    with row3a:
        cohort = (df_customers.groupby("trial_start").size()
                  .reset_index(name="new_trials"))
        fig5 = px.bar(cohort, x="trial_start", y="new_trials",
                      title="Trial starts by date",
                      labels={"trial_start":"Date","new_trials":"Customers"},
                      color_discrete_sequence=["#4A6FA5"])
        fig5.update_layout(height=280, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig5, use_container_width=True)

    with row3b:
        kit_avg = (df_customers.groupby("product_kit")["propensity_score"]
                   .mean().reset_index().sort_values("propensity_score", ascending=False))
        fig6 = px.bar(kit_avg, x="product_kit", y="propensity_score",
                      title="Mean conversion score by product kit",
                      labels={"product_kit":"Kit","propensity_score":"Mean score"},
                      color_discrete_sequence=["#4A6FA5"],
                      text=kit_avg["propensity_score"].map(lambda x: f"{x:.0%}"))
        fig6.update_traces(textposition="outside")
        fig6.update_layout(height=280, margin=dict(l=0,r=0,t=40,b=0),
                           xaxis_tickangle=-20)
        st.plotly_chart(fig6, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# TAB 2 — CUSTOMER DATABASE
# ─────────────────────────────────────────────────────────────

with tab_customers:
    st.subheader("Customer records")

    fc1, fc2, fc3 = st.columns(3)
    f_concern = fc1.multiselect("Skin concern", SKIN_CONCERNS, default=SKIN_CONCERNS)
    f_kit     = fc2.multiselect("Product kit",  PRODUCT_KITS,  default=PRODUCT_KITS)
    f_score   = fc3.slider("Min propensity score", 0.0, 1.0, 0.0, 0.05)

    filtered = df_customers[
        df_customers["skin_concern"].isin(f_concern) &
        df_customers["product_kit"].isin(f_kit) &
        (df_customers["propensity_score"] >= f_score)
    ].copy()

    def priority_badge(s):
        if s >= 0.60: return "High"
        if s >= 0.35: return "Medium"
        return "Low"

    filtered["priority"]     = filtered["propensity_score"].apply(priority_badge)
    filtered["trial_status"] = filtered["days_left"].apply(
        lambda d: f"Active ({d}d)" if d > 0 else "Ended"
    )

    display_cols = ["customer_id","name","province","age_range","skin_concern",
                    "skin_type","product_kit","trial_start","trial_status",
                    "email_ctr","site_visits","propensity_score","priority"]

    st.dataframe(
        filtered[display_cols].sort_values("propensity_score", ascending=False),
        use_container_width=True,
        height=480,
        column_config={
            "propensity_score": st.column_config.ProgressColumn(
                "Propensity score", min_value=0, max_value=1, format="%.0%"
            ),
            "email_ctr": st.column_config.NumberColumn("Email CTR", format="%.1%%"),
        }
    )
    st.caption(f"Showing {len(filtered)} of {len(df_customers)} customers")


# ─────────────────────────────────────────────────────────────
# TAB 3 — LEAD SCORING (individual)
# ─────────────────────────────────────────────────────────────

with tab_score:
    st.subheader("Individual customer scoring")
    st.caption("Data source: Shopify + Supabase (production) · synthetic data (demo)")

    left, right = st.columns([1, 2])

    with left:
        # 1. Added a search bar for better UX
        search_score = st.text_input("Search customer by name or ID", key="score_search")
        
        # Filter the dataframe based on the search query
        if search_score:
            filtered_df = df_customers[
                df_customers["name"].str.contains(search_score, case=False, na=False) |
                df_customers["customer_id"].str.contains(search_score, case=False, na=False)
            ]
        else:
            filtered_df = df_customers

        # Format options to show ID and Name clearly
        options = ["-- enter manually --"] + list(filtered_df["customer_id"] + " — " + filtered_df["name"])

        selected_val = st.selectbox(
            "Load from database",
            options=options,
        )
        
        if selected_val != "-- enter manually --":
            # Extract the ID from the selected string
            selected_id = selected_val.split(" — ")[0]
            row = df_customers[df_customers["customer_id"] == selected_id].iloc[0]
            
            def_concern = row["skin_concern"]
            def_skin    = row["skin_type"]
            def_age     = row["age_range"]
            def_visits  = row["site_visits"]
            def_ctr     = row["email_ctr"]
            def_pages   = row["pages_viewed"]
        else:
            def_concern = "Rosacea"
            def_skin    = "Sensitive"
            def_age     = "40-49"
            def_visits  = 3
            def_ctr     = 0.15
            def_pages   = 2

        st.divider()
        input_concern = st.selectbox("Skin concern", SKIN_CONCERNS,
                                     index=SKIN_CONCERNS.index(def_concern))
        input_skin    = st.selectbox("Skin type", SKIN_TYPES,
                                     index=SKIN_TYPES.index(def_skin))
        input_age     = st.selectbox("Age range", AGE_RANGES,
                                     index=AGE_RANGES.index(def_age))
        input_visits  = st.slider("Site visits (post-sample)", 0, 15, def_visits)
        input_ctr     = st.slider("Email CTR", 0.0, 0.5,
                                  float(def_ctr), 0.01, format="%.2f")
        input_pages   = st.slider("Product pages viewed", 0, 10, def_pages)

    with right:
        score = score_customer(input_concern, input_skin, input_age,
                               input_visits, input_ctr, input_pages)

        if score >= 0.60:
            tier = "High"
        elif score >= 0.35:
            tier = "Medium"
        else:
            tier = "Low"

        st.metric("Conversion propensity", f"{score:.1%}", help="Priority tier: " + tier)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(score * 100, 1),
            number={"suffix":"%","font":{"size":32}},
            gauge={
                "axis": {"range":[0,100], "ticksuffix":"%"},
                "bar":  {"color": "#4A6FA5"},
                "bgcolor": "#f0f0f0",
                "borderwidth": 0,
                "steps": [],
                "threshold": {"line":{"color":"#333","width":2},"value":60},
            },
            title={"text": f"Priority: {tier}", "font":{"size":14}},
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20,r=20,t=40,b=0),
                                paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_gauge, use_container_width=True)

        signals = {
            "Email CTR":      min(input_ctr / 0.5, 1.0),
            "Site visits":    min(input_visits / 15, 1.0),
            "Pages viewed":   min(input_pages / 10, 1.0),
        }
        sig_df = pd.DataFrame({"Feature": list(signals.keys()),
                               "Relative strength": list(signals.values())})
        fig_sig = px.bar(sig_df, x="Relative strength", y="Feature", orientation="h",
                         range_x=[0,1], title="Engagement signals",
                         color_discrete_sequence=["#4A6FA5"],
                         text=sig_df["Relative strength"].map(lambda x: f"{x:.0%}"))
        fig_sig.update_traces(textposition="outside")
        fig_sig.update_layout(height=200, margin=dict(l=0,r=20,t=40,b=0))
        st.plotly_chart(fig_sig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# TAB 4 — AI OUTREACH
# ─────────────────────────────────────────────────────────────

with tab_outreach:
    st.subheader("Email outreach")

    # Get the top leads
    top_leads = df_customers[df_customers["propensity_score"] >= 0.55].sort_values(
        "propensity_score", ascending=False
    )

    out_col1, out_col2 = st.columns([1, 2])

    with out_col1:        
        # 1. Added search bar for the outreach tab
        search_outreach = st.text_input("Search Customer", key="outreach_search")
        
        # Filter top leads based on search query
        if search_outreach:
            leads_to_show = top_leads[
                top_leads["name"].str.contains(search_outreach, case=False, na=False) |
                top_leads["customer_id"].str.contains(search_outreach, case=False, na=False)
            ]
        else:
            leads_to_show = top_leads.head(20) # Default to top 20 if no search

        # Check if search returned any results
        if not leads_to_show.empty:
            sel = st.selectbox(
                "Customer", 
                leads_to_show["customer_id"] + " — " + leads_to_show["name"] + " (" + 
                leads_to_show["propensity_score"].map(lambda x: f"{x:.0%}") + ")"
            )

            cid = sel.split(" — ")[0]
            c   = df_customers[df_customers["customer_id"] == cid].iloc[0]

            st.markdown(f"""
    **{c['name']}** · {c['customer_id']}

    Kit: {c['product_kit']}  
    Concern: {c['skin_concern']} · {c['skin_type']} skin  
    Trial start: {c['trial_start']} ({c['days_elapsed']} days ago)  
    Email CTR: {c['email_ctr']:.1%}  
    Site visits: {c['site_visits']}  
    Score: **{c['propensity_score']:.0%}**
            """)

            tone = st.radio("Tone", ["Warm", "Professional", "Urgency"], index=0)
        else:
            st.warning("No high-propensity customers found matching that name.")
            c = None # Prevents rendering issues if no customer is found

    with out_col2:
        # Only show the generation button if a customer is selected
        if c is not None:
            if st.button("Generate email draft", type="primary"):
                tone_map = {
                    "Warm":       "warm and conversational",
                    "Professional": "professional and concise",
                    "Urgency":    "direct — note the trial is ending soon",
                }
                prompt = f"""
    You are a customer success specialist for Riversol, a Canadian dermatological skincare brand.
    Write a short follow-up email to a customer partway through their 15-day free sample kit trial.

    Customer profile:
    - Name: {c['name']}
    - Product kit: {c['product_kit']}
    - Skin concern: {c['skin_concern']}
    - Skin type: {c['skin_type']}
    - Days into trial: {c['days_elapsed']} of 15
    - Website visits since ordering: {c['site_visits']}

    Tone: {tone_map[tone]}
    Goal: Encourage purchase of the full-size product. Reference Dr. Jason Rivers and the formulation for {c['skin_concern']}. Do not be pushy. End with a clear call-to-action.
    Keep it under 150 words. Include a subject line.
    """
                if GEMINI_API_KEY:
                    with st.spinner("Generating via Gemini..."):
                        try:
                            llm = genai.GenerativeModel("gemini-2.5-flash")
                            resp = llm.generate_content(prompt)
                            st.success("Draft ready")
                            st.text_area("Email draft", resp.text, height=300)
                        except Exception as e:
                            st.error(f"Gemini error: {e}")
                else:
                    st.warning("Gemini API key not set — showing placeholder output")
                    st.text_area("Placeholder draft", f"""Subject: Your Riversol trial — day {c['days_elapsed']}

    Hi {c['name'].split()[0]},

    You are {c['days_elapsed']} days into your Riversol trial. Dr. Jason Rivers developed the {c['product_kit']} specifically for {c['skin_concern']}, using clinically tested formulas free from common irritants.

    If you have noticed a difference, the full-size products are available whenever you are ready.

    The Riversol Team
    """, height=280)

        st.divider()
        st.caption(
            "Production flow: score >= 0.55 triggers draft generation via Gemini API, "
            "draft is queued in Klaviyo for review, sent on approval."
        )


# ─────────────────────────────────────────────────────────────
# TAB 5 — AUTOMATED COHORT INSIGHTS
# Pipeline: Generate cohort data → Aggregate metrics → Visualise
#           → Build structured LLM prompt → AI executive summary
# ─────────────────────────────────────────────────────────────

# ── LLM system prompt ─────────────────────────────────────────
# Injected as the system role so every Gemini call carries it.

COHORT_SYSTEM_PROMPT = """
You are a Senior Data Scientist reviewing cohort conversion results for \
Riversol, a direct-to-consumer skincare company. Your role is to interpret \
the statistical patterns in the data and produce rigorous, technical \
recommendations for improving the predictive model and the decision pipeline.

When you receive cohort conversion statistics, structure your response \
using exactly these markdown headers:

## Summary of Findings
Two to three sentences. State the most statistically notable pattern \
in the cohort data and its implication for model quality. Be direct.

## Segment-Level Signal Analysis
Three to five bullet points. For each, cite the exact conversion rate \
and explain what it reveals about the predictive signal in that variable \
(e.g., whether skin concern is a strong discriminative feature, whether \
age introduces a monotone or non-monotone relationship with conversion, etc.).

## Model Performance Observations
Analyse what the cohort distribution implies about the current XGBoost model. \
Consider: class imbalance across segments, which cohorts may be \
underrepresented in training data, and whether the log-odds weights \
(used to generate labels) are well-calibrated against observed rates.

## Feature & Data Quality Recommendations
Identify which features appear to carry the most signal and which are \
likely noisy. Recommend one or two additional data points (e.g., from \
Shopify or Supabase) that — if collected — would most improve model \
discrimination. Explain the reasoning in terms of information gain.

## Modelling Improvement Suggestions
Propose one concrete change to the training pipeline that would improve \
decision quality. Examples: recalibration, threshold tuning per segment, \
adding interaction features, or applying SHAP-based feature selection. \
Tie each suggestion back to a specific pattern in the cohort data.

Tone: precise, technical, evidence-based. Avoid generic statements. \
Every claim must reference a number from the data provided.
""".strip()


# ── Helper: build the user-turn prompt from live stats ────────

def build_cohort_user_prompt(df: pd.DataFrame) -> str:
    """
    Aggregates conversion metrics from the cohort dataframe and
    serialises them into a structured natural-language prompt
    ready to be sent to the LLM.
    """
    overall_rate = df["converted"].mean()
    n_total      = len(df)
    n_converted  = df["converted"].sum()

    # Group-level conversion rates
    by_concern = (
        df.groupby("skin_concern")["converted"]
        .agg(["mean", "count", "sum"])
        .rename(columns={"mean": "rate", "count": "n", "sum": "converts"})
        .sort_values("rate", ascending=False)
    )
    by_age = (
        df.groupby("age_range")["converted"]
        .agg(["mean", "count", "sum"])
        .rename(columns={"mean": "rate", "count": "n", "sum": "converts"})
        .sort_values("rate", ascending=False)
    )
    by_skin = (
        df.groupby("skin_type")["converted"]
        .agg(["mean", "count", "sum"])
        .rename(columns={"mean": "rate", "count": "n", "sum": "converts"})
        .sort_values("rate", ascending=False)
    )

    # Best cross-segment (concern × age)
    cross = (
        df.groupby(["skin_concern", "age_range"])["converted"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "rate", "count": "n"})
        .query("n >= 10")          # only report statistically meaningful cells
        .sort_values("rate", ascending=False)
    )

    def fmt_group(grp_df):
        lines = []
        for name, row in grp_df.iterrows():
            lines.append(
                f"  • {name}: {row['rate']:.1%} conversion rate "
                f"({int(row['converts'])} / {int(row['n'])} customers)"
            )
        return "\n".join(lines)

    def fmt_cross(cross_df):
        lines = []
        for (concern, age), row in cross_df.head(5).iterrows():
            lines.append(
                f"  • {concern} × {age}: {row['rate']:.1%} "
                f"(n={int(row['n'])})"
            )
        return "\n".join(lines)

    prompt = f"""
Riversol Cohort Analysis — {n_total} completed sample-kit trials

OVERALL METRICS
  • Total customers analysed: {n_total}
  • Total conversions: {n_converted}
  • Overall conversion rate: {overall_rate:.1%}

CONVERSION RATE BY SKIN CONCERN
{fmt_group(by_concern)}

CONVERSION RATE BY AGE GROUP
{fmt_group(by_age)}

CONVERSION RATE BY SKIN TYPE
{fmt_group(by_skin)}

TOP CROSS-SEGMENTS (Concern × Age, min 10 customers)
{fmt_cross(cross)}

Please generate your full strategic analysis now.
""".strip()

    return prompt


# ── Tab layout ────────────────────────────────────────────────

with tab_cohort:
    st.subheader("Cohort Analysis")
    st.caption(
        "500 completed 15-day trials · "
        "AI-generated data analyst summary via Gemini · "
        "In production: replace synthetic data with a live Shopify / Supabase query."
    )

    # df_cohort is loaded at startup — no per-tab generation needed.

    # ── Aggregate metrics ─────────────────────────────────────
    overall_conv = df_cohort["converted"].mean()

    agg_concern = (
        df_cohort.groupby("skin_concern")["converted"]
        .mean()
        .reset_index()
        .rename(columns={"converted": "conversion_rate"})
        .sort_values("conversion_rate", ascending=False)
    )
    agg_age = (
        df_cohort.groupby("age_range")["converted"]
        .mean()
        .reset_index()
        .rename(columns={"converted": "conversion_rate"})
        .sort_values("conversion_rate", ascending=False)
    )
    agg_skin = (
        df_cohort.groupby("skin_type")["converted"]
        .mean()
        .reset_index()
        .rename(columns={"converted": "conversion_rate"})
        .sort_values("conversion_rate", ascending=False)
    )

    # ── KPI banner ────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Cohort size",        f"{len(df_cohort):,} customers")
    k2.metric("Overall conv. rate", f"{overall_conv:.1%}")
    k3.metric("Best segment",       agg_concern.iloc[0]["skin_concern"])
    k4.metric("Best segment rate",  f"{agg_concern.iloc[0]['conversion_rate']:.1%}")

    st.divider()

    # ── Row 1: Skin Concern + Age Group ──────────────────────
    ch_col1, ch_col2 = st.columns(2)

    with ch_col1:
        # Colour bars by performance vs overall average
        agg_concern["colour"] = agg_concern["conversion_rate"].apply(
            lambda r: "#1D9E75" if r >= overall_conv else "#E24B4A"
        )
        fig_concern = px.bar(
            agg_concern,
            x="conversion_rate",
            y="skin_concern",
            orientation="h",
            title="Conversion Rate by Skin Concern",
            labels={"conversion_rate": "Conv. rate", "skin_concern": ""},
            color="colour",
            color_discrete_map="identity",
            text=agg_concern["conversion_rate"].map(lambda x: f"{x:.1%}"),
        )
        fig_concern.add_vline(
            x=overall_conv, line_dash="dot", line_color="#555",
            annotation_text=f"Overall {overall_conv:.1%}",
            annotation_position="top right",
        )
        fig_concern.update_traces(textposition="outside", showlegend=False)
        fig_concern.update_layout(
            height=320, margin=dict(l=0, r=30, t=45, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig_concern, use_container_width=True)

    with ch_col2:
        agg_age["colour"] = agg_age["conversion_rate"].apply(
            lambda r: "#1D9E75" if r >= overall_conv else "#E24B4A"
        )
        fig_age = px.bar(
            agg_age,
            x="age_range",
            y="conversion_rate",
            title="Conversion Rate by Age Group",
            labels={"conversion_rate": "Conv. rate", "age_range": "Age group"},
            color="colour",
            color_discrete_map="identity",
            text=agg_age["conversion_rate"].map(lambda x: f"{x:.1%}"),
        )
        fig_age.add_hline(
            y=overall_conv, line_dash="dot", line_color="#555",
            annotation_text=f"Overall {overall_conv:.1%}",
            annotation_position="top right",
        )
        fig_age.update_traces(textposition="outside", showlegend=False)
        fig_age.update_layout(
            height=320, margin=dict(l=0, r=10, t=45, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig_age, use_container_width=True)

    # ── Row 2: Skin Type heatmap (concern × age) ─────────────
    ch_col3, ch_col4 = st.columns([1, 2])

    with ch_col3:
        fig_skin = px.bar(
            agg_skin.sort_values("conversion_rate"),
            x="conversion_rate",
            y="skin_type",
            orientation="h",
            title="Conversion Rate by Skin Type",
            labels={"conversion_rate": "Conv. rate", "skin_type": ""},
            color="conversion_rate",
            color_continuous_scale=["#E24B4A", "#f7c59f", "#1D9E75"],
            text=agg_skin["conversion_rate"].map(lambda x: f"{x:.1%}"),
        )
        fig_skin.update_traces(textposition="outside")
        fig_skin.update_layout(
            height=320, margin=dict(l=0, r=30, t=45, b=0),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_skin, use_container_width=True)

    with ch_col4:
        # Pivot: skin concern (rows) × age range (cols) → conv. rate
        heatmap_df = (
            df_cohort.groupby(["skin_concern", "age_range"])["converted"]
            .mean()
            .unstack("age_range")
            .reindex(columns=AGE_RANGES)          # enforce chronological order
        )
        fig_heat = px.imshow(
            heatmap_df,
            color_continuous_scale=["#E24B4A", "#f7f7f7", "#1D9E75"],
            zmin=0, zmax=1,
            text_auto=".0%",
            title="Conversion Rate Heatmap — Concern × Age",
            labels={"x": "Age group", "y": "Skin concern", "color": "Conv. rate"},
            aspect="auto",
        )
        fig_heat.update_layout(
            height=320, margin=dict(l=0, r=0, t=45, b=0),
            coloraxis_showscale=True,
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()

    # ── AI Executive Summary ──────────────────────────────────
    st.markdown("### AI Data Analyst Summary")
    st.caption(
        "Click below to send the aggregated cohort statistics to Gemini. "
        "The model is prompted to respond as a Senior Data Scientist and "
        "will return a structured technical report covering signal analysis, "
        "model observations, and pipeline improvement recommendations."
    )

    if st.button("Generate AI Data Analyst Summary", type="primary", key="cohort_llm_btn"):
        user_prompt = build_cohort_user_prompt(df_cohort)

        if GEMINI_API_KEY:
            with st.spinner("Gemini is analysing the cohort data..."):
                try:
                    llm = genai.GenerativeModel(
                        model_name="gemini-2.5-flash",
                        system_instruction=COHORT_SYSTEM_PROMPT,
                    )
                    response = llm.generate_content(user_prompt)
                    st.success("Analysis complete")

                    # Render the markdown report inline
                    st.markdown(response.text)

                    # Allow download as plain text for sharing
                    st.download_button(
                        label="Download report (.txt)",
                        data=response.text,
                        file_name="riversol_cohort_report.txt",
                        mime="text/plain",
                    )

                except Exception as e:
                    st.error(f"Gemini error: {e}")
        else:
            # ── Graceful placeholder when no API key is set ───
            st.warning("Gemini API key not set — showing a placeholder report.")
            user_prompt_display = build_cohort_user_prompt(df_cohort)

            # Show the data payload that would be sent (useful for demos)
            with st.expander("Data payload that would be sent to Gemini"):
                st.code(user_prompt_display, language="text")

            st.markdown(f"""
## Summary of Findings
**{agg_concern.iloc[0]['skin_concern']}** is the strongest-converting segment at
**{agg_concern.iloc[0]['conversion_rate']:.1%}**, indicating that skin concern
is a highly discriminative feature and is likely carrying significant weight in
the model. The spread across concerns suggests this variable should be inspected
for label leakage before production deployment.

## Segment-Level Signal Analysis
- **Skin concern** shows the widest variance across groups, making it the most
  discriminative categorical feature in the current feature set.
- **Age range** follows a non-monotone pattern — the 40-49 bracket peaks before
  declining at 60+, which means a linear encoding would misrepresent this
  relationship. One-hot or ordinal encoding should be compared empirically.
- **Skin type** shows weaker separation, suggesting it contributes less marginal
  information gain once concern and age are already encoded.
- The Concern x Age heatmap reveals interaction effects — some cells diverge
  substantially from their row and column marginals, which a simple additive
  model cannot capture.

## Model Performance Observations
The current XGBoost model is trained on 2,000 synthetic records where labels
are generated by the same log-odds function used to produce this cohort. This
means the model is essentially learning to recover its own data-generating process,
which will produce artificially high AUC on held-out data. In production, labels
must come from real Shopify outcomes. The low-converting segments (Oily/Acne,
under-30 cohort) are also likely underrepresented in the positive class,
which could cause threshold miscalibration for those groups.

## Feature & Data Quality Recommendations
- **Add days-to-first-site-visit** as a feature — early return visitors likely
  indicate higher intent and are not currently captured.
- **Collect return visit frequency**, not just total site visits. A customer
  visiting 5 times in 2 days signals differently from 5 visits over 14 days.
- Skin type currently adds minimal separation; evaluate dropping it or replacing
  it with a continuous skin sensitivity score if available from the intake form.

## Modelling Improvement Suggestions
Apply **per-segment threshold calibration** rather than a single 0.5 cutoff.
The Rosacea cohort likely has a well-calibrated score, but the Oily/Acne
segment at **{agg_concern.iloc[-1]['conversion_rate']:.1%}** conversion may
produce more false positives under the current threshold. Use isotonic
regression calibration on the held-out test set and evaluate Brier score
by segment to confirm. Additionally, add SHAP interaction values to surface
whether concern-age pairs are driving nonlinear score shifts.
            """)

    st.divider()
    st.caption(
        "Production roadmap: schedule this pipeline via a weekly cron job → "
        "auto-generate the report → push summary to a Slack #insights channel. "
        "Cohort data source: Shopify Orders API filtered to trials older than 15 days."
    )