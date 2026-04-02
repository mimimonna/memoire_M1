# 🌸 Assistant Intelligent pour la Santé Menstruelle

> Mémoire de fin d'études — DU Sorbonne Data Analytics  
> Université Paris 1 Panthéon-Sorbonne | Avril 2026

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://menstrualwomanapp.streamlit.app)

---

## 📌 Description

Ce projet développe un **assistant intelligent dédié à la santé menstruelle**, basé sur une approche hybride combinant **Machine Learning** et **Traitement du Langage Naturel (NLP)**.

L'objectif est de proposer un outil simple et accessible permettant :
- de **répondre aux questions** des utilisatrices en langage naturel
- de **prédire l'ovulation** à partir de données physiologiques
- d'**explorer visuellement** les données du cycle menstruel

> ⚠️ *Cet assistant fournit des informations générales uniquement. Il ne remplace pas un avis médical professionnel.*

---

## Démo

 **[Accéder à l'application](https://menstrualwomanapp.streamlit.app)**

---

## Approche Technique

### Machine Learning
- Prédiction de l'ovulation (`ovulation_result`) à partir de données tabulaires
- Modèles testés : Régression Logistique, **Random Forest**, XGBoost
- Meilleur modèle : Random Forest (accuracy ~69%)

### NLP
- Approche testée : Fine-tuning de **FLAN-T5** → résultats instables (dataset trop petit)
- Approche retenue : **Sentence-BERT** (`all-MiniLM-L6-v2`) + similarité cosinus
- Recherche sémantique sur une base de 530 paires questions/réponses

---

## Structure du projet

```
 m-moire/
├──  app.py                  # Application Streamlit
├──  requirements.txt        # Dépendances Python
├──  README.md               # Ce fichier
├──  Period_Log.csv          # Dataset cycle menstruel (17 976 entrées)
├──  Training Data.csv       # Dataset Q&R santé menstruelle (530 entrées)
└──  notebook_memoire.ipynb  # Notebook Google Colab
```

---

## Datasets

| Dataset | Source | Taille | Usage |
|---|---|---|---|
| Menstrual Cycle Data with Factors | Kaggle | 17 976 lignes | Machine Learning |
| Menstrual Health Dataset | Kaggle | 530 paires Q&R | NLP |

---

## Installation locale

```bash
# Cloner le repository
git clone https://github.com/mimimonna/m-moire.git
cd m-moire

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run app.py
```

---

## Dépendances

```
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
sentence-transformers
torch
xgboost
joblib
```

---

## Fonctionnalités de l'app

| Page | Description |
|---|---|
|  Accueil | Présentation du projet |
|  Assistant NLP | Questions en langage naturel avec Sentence-BERT |
|  Prédiction Ovulation | Prédiction avec sliders interactifs (Random Forest) |
|  Visualisations | Exploration interactive des données du cycle |

---

## Références principales

- Devlin et al. (2019) — BERT
- Reimers & Gurevych (2019) — Sentence-BERT
- Breiman (2001) — Random Forests
- Chen & Guestrin (2016) — XGBoost
- Laranjo et al. (2018) — Conversational agents in healthcare
- Moglia et al. (2016) — Menstrual cycle tracking apps

---

## Auteure

**Monna**  
DU Sorbonne Data Analytics  
Université Paris 1 Panthéon-Sorbonne  
Avril 2026