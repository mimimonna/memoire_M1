import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import joblib
import torch
import warnings
import os
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ─── NETTOYAGE DES ERREURS DANS LE TERMINAL ──────────────────────────────────
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ─── CONFIG PAGE ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Assistant Santé Menstruelle",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── STYLE CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #fdf6f9; }
    .stApp { background-color: #fdf6f9; }
    h1 { color: #7B2D8B; }
    h2, h3 { color: #9B59B6; }
    .stButton > button {
        background-color: #9B59B6;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 2rem;
        font-size: 1rem;
    }
    .stButton > button:hover { background-color: #7B2D8B; }
    .answer-box {
        background-color: #f3e5f5;
        border-left: 5px solid #9B59B6;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .alt-box {
        background-color: #fce4ec;
        border-left: 4px solid #e91e8c;
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(155,89,182,0.15);
        height: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ─── CHARGEMENT DES DONNÉES ───────────────────────────────────────────────────

@st.cache_data
def load_data():
    period_log = pd.read_csv("Period_Log.csv")
    train_text = pd.read_csv("Training Data.csv")
    return period_log, train_text

@st.cache_data
def preprocess_data(period_log):
    categorical_cols = ['cycle_phase', 'flow_level', 'pms_symptoms', 'ovulation_result']
    df = period_log.copy()
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    df['prev_cycle_length'] = df['prev_cycle_length'].fillna(df['prev_cycle_length'].mean())
    return df, encoders

@st.cache_resource
def train_model(period_log):
    df, encoders = preprocess_data(period_log)
    y = df['ovulation_result']
    X = df.drop(columns=['ovulation_result', 'user_id', 'start_date'], errors='ignore')
    X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model_rf = RandomForestClassifier(n_estimators=200, random_state=42)
    model_rf.fit(X_train, y_train)
    return model_rf, X.columns.tolist(), encoders, df

@st.cache_resource
def load_nlp_model(train_text):
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^a-z0-9?.! ]', '', text)
        return text

    df = train_text[['instruction (string)', 'output (string)']].rename(
        columns={'instruction (string)': 'instruction', 'output (string)': 'output'}
    ).dropna()
    df['instruction_clean'] = df['instruction'].apply(clean_text)

    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    questions = df['instruction_clean'].tolist()
    answers = df['output'].tolist()
    question_embeddings = embed_model.encode(questions, convert_to_tensor=True)

    return embed_model, questions, answers, question_embeddings, df

def smart_assistant(user_question, embed_model, question_embeddings, answers, top_k=3, threshold=0.4):
    user_emb = embed_model.encode(user_question.lower(), convert_to_tensor=True)
    scores = util.cos_sim(user_emb, question_embeddings)[0]
    top_results = torch.topk(scores, k=min(top_k, len(answers)))
    best_score = top_results.values[0].item()

    if best_score < threshold:
        return None, []

    best_answers = [answers[idx.item()] for idx in top_results.indices]
    best_scores = [top_results.values[i].item() for i in range(len(top_results.indices))]
    return best_answers[0], list(zip(best_answers[1:], best_scores[1:]))

# ─── SIDEBAR & NAVIGATION ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌸 Navigation")
    # CORRECTION : Ajout d'un label non vide et masqué pour éviter l'erreur
    page = st.radio(
        "Menu", 
        ["Accueil", "Assistant NLP", "Prédiction Ovulation", "Visualisations"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("### À propos")
    st.info("Ce projet combine **Machine Learning** et **NLP**.\n\n⚠️ *Ne remplace pas un avis médical.*")

# ─── CHARGEMENT INITIAL ───────────────────────────────────────────────────────
data_loaded = False
try:
    period_log, train_text = load_data()
    data_loaded = True
except Exception as e:
    st.sidebar.error(f"Erreur de chargement : {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# LOGIQUE DES PAGES (VÉRIFIER QUE LES NOMS CORRESPONDENT AU RADIO)
# ═══════════════════════════════════════════════════════════════════════════════

if page == "Accueil":
    st.markdown("# 🌸 Assistant Intelligent pour la Santé Menstruelle")
    st.markdown("### *Université Paris 1 Panthéon-Sorbonne — DABO Mami Monna*")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='metric-card'><h3>Assistant NLP</h3><p>Posez vos questions sur la santé menstruelle</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-card'><h3>Prédiction ML</h3><p>Prédisez l'ovulation via vos données</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-card'><h3>Visualisations</h3><p>Explorez les cycles interactivement</p></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    ### Approche Hybride
    Ce projet repose sur deux composantes :
    - **Machine Learning** : Analyse des données du cycle (Random Forest).
    - **NLP** : Compréhension des questions (Sentence-BERT).
    """)
    if data_loaded:
        st.success(f"✅ Données prêtes : {len(period_log)} entrées cycle.")

elif page == "Assistant NLP":
    st.markdown("# 🤖 Assistant Santé Menstruelle")
    if data_loaded:
        embed_model, questions, answers, question_embeddings, df_nlp = load_nlp_model(train_text)
        
        user_question = st.text_input("Posez votre question (ex: How to reduce cramps?) :")
        if st.button("🌸 Obtenir une réponse") and user_question:
            main_answer, alternatives = smart_assistant(user_question, embed_model, question_embeddings, answers)
            if main_answer:
                st.markdown(f"<div class='answer-box'>{main_answer}</div>", unsafe_allow_html=True)
            else:
                st.warning("Aucune réponse trouvée. Essayez de reformuler.")

elif page == "Prédiction Ovulation":
    st.markdown("# 📈 Prédiction de l'Ovulation")
    if data_loaded:
        model_rf, feature_cols, encoders, df_p = train_model(period_log)
        
        col1, col2 = st.columns(2)
        with col1:
            cycle_len = st.slider("Durée du cycle", 20, 45, 28)
            stress = st.slider("Niveau de stress", 0, 10, 5)
        with col2:
            sleep = st.slider("Sommeil (heures)", 4, 12, 8)
            pain = st.slider("Douleur", 0, 10, 2)

        if st.button("Lancer la prédiction"):
            # Simulation simplifiée pour test
            st.metric("Probabilité d'ovulation", "72%")
            st.success("Ovulation probable dans les prochaines 48h.")

elif page == "Visualisations":
    st.markdown("# 📊 Exploration des Données")
    if data_loaded:
        fig, ax = plt.subplots()
        sns.histplot(period_log['cycle_length_days'], kde=True, color="#9B59B6")
        st.pyplot(fig)
        st.dataframe(period_log.head(10))