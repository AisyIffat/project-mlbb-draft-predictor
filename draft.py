import streamlit as st
import pandas as pd
import numpy as np
import scikit.learn
import joblib

hero = pd.read_csv("sample_101_Mlbb_Heroes.csv")
model = joblib.load("draft.joblib")

st.set_page_config(layout="wide")
st.title("MLBB Draft Win Predictor")

def team_score(hero_list):
    stats = hero[hero["Name"].isin(hero_list)]
    power = (
        stats["Phy_Damage"] +
        stats["Mag_Damage"] +
        stats["Phy_Defence"] +
        stats["Mag_Defence"] +
        stats["Mov_Speed"]
    )
    return power.mean()

def draft_features(allies, enemies):
    ally_score = team_score(allies)
    enemy_score = team_score(enemies)

    final_score = ally_score - enemy_score

    return np.array([[final_score]])

available = hero["Name"].tolist()

col_ally, col_enemy = st.columns(2)

with col_ally:
    st.subheader("ALLY TEAM")
    allies = []
    for i in range(5):
        pick = st.selectbox(
            f"Ally Hero {i+1}",
            available,
            key=f"a{i}"
        )
        allies.append(pick)
        available.remove(pick)

with col_enemy:
    st.subheader("ENEMY TEAM")
    enemies = []
    for i in range(5):
        pick = st.selectbox(
            f"Enemy Hero {i+1}",
            available,
            key=f"e{i}"
        )
        enemies.append(pick)
        available.remove(pick)

if st.button("Predict Result"):
    X = draft_features(allies, enemies)
    winrate = model.predict(X)[0]
    winrate = max(0, min(1, winrate))

    st.markdown("---")
    st.write(f"**Predicted Win Rate:** {winrate:.2f}")

    if winrate >= 0.5:
        st.success("Prediction: VICTORY")
    else:
        st.error("Prediction: DEFEAT")