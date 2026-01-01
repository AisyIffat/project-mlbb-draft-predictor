import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="MLBB Draft Predictor",
    layout="wide"
)

st.title("MLBB Draft Win Probability Predictor")
st.write("Select 5 heroes for each team to predict match outcome.")

model = joblib.load("draft.joblib")
hero = pd.read_csv('sample_101_Mlbb_Heroes.csv')

hero_names = sorted(hero["Name"].unique())

hero["Total_damage"] = hero["Phy_Damage"] + hero["Mag_Damage"]
hero["Total_defence"] = hero["Phy_Defence"] + hero["Mag_Defence"]

hero_stats = hero.set_index("Name")[[
    "Total_damage",
    "Total_defence",
    "Mov_Speed"
]]

def team_stats(hero_list):
    team = hero_stats.loc[hero_list]
    total_damage = team["Total_damage"].sum()
    total_defence = team["Total_defence"].sum()
    avg_speed = team["Mov_Speed"].mean()
    return total_damage, total_defence, avg_speed

def draft_features(allies, enemies):
    a_dmg, a_def, a_spd = team_stats(allies)
    e_dmg, e_def, e_spd = team_stats(enemies)

    return [[
        a_dmg - e_dmg,
        a_def - e_def,
        a_spd - e_spd
    ]]

available = hero_names.copy()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Allies")
    allies = []
    for i in range(5):
        pick = st.selectbox(
            f"Ally Hero {i+1}",
            available,
            key=f"a{i}"
        )
        allies.append(pick)
        available.remove(pick)

with col2:
    st.subheader("Enemies")
    enemies = []
    for i in range(5):
        pick = st.selectbox(
            f"Enemy Hero {i+1}",
            available,
            key=f"e{i}"
        )
        enemies.append(pick)
        available.remove(pick)

if st.button("Predict Match Outcome"):
    X_input = draft_features(allies, enemies)
    prediction = model.predict(X_input)[0]

    st.subheader("Prediction Result")
    st.write(f"Win Probability: **{prediction:.2f}**")

    if prediction >= 0.5:
        st.success("Victory")
    else:
        st.error("Defeat")