import streamlit as st
from utils import *
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import io
import zipfile
from datetime import datetime
import sys
print(sys.executable)

if "ji" not in st.session_state:
    st.session_state["ji"] = 5.5

if "jf" not in st.session_state:
    st.session_state["jf"] = 1.5


def round_to_half(x):
    return round(x * 2) / 2

def round_ji_and_reset_am():
    ji_val = round_to_half(st.session_state["ji"])
    st.session_state["ji"] = ji_val
    m_state = np.arange(-ji_val, ji_val + 1, 1)
    new_am = [1 / len(m_state)] * len(m_state)

    st.session_state["a_m"] = new_am
    for i, m in enumerate(m_state):
        st.session_state[f"am_{m}"] = float(new_am[i]) 

    st.session_state["polarisation"], st.session_state["alignement"] = calculate_polarisation(
        ji_val, new_am
    )

def reset_am():
    ji_val = st.session_state["ji"]
    m_state = np.arange(-ji_val, ji_val + 1, 1)
    new_am = [1 / len(m_state)] * len(m_state)

    st.session_state["a_m"] = new_am
    for i, m in enumerate(m_state):
        st.session_state[f"am_{m}"] = float(new_am[i]) 

    st.session_state["polarisation"], st.session_state["alignement"] = calculate_polarisation(
        ji_val, new_am
    )


st.set_page_config(layout="wide")

# === LEFT COLUMN ===
with st.sidebar:
    st.header("Input Parameters")

    ji = st.number_input("Jᵢ", step=0.5, format="%.1f", key="ji", on_change=round_ji_and_reset_am)
    jf = st.number_input("J𝒻", step=0.5, format="%.1f", key="jf")
    delta = st.number_input("δ", value=0.0, format="%.3f")

    if "am_initialized" not in st.session_state:
        reset_am()
        st.session_state["am_initialized"] = True

    m_state = np.arange(-ji, ji + 1, 1)

    st.markdown("---")
    if st.button("Reset aₘ"):
        reset_am()
        st.experimental_rerun()

    st.subheader("aₘ Inputs")

    if "a_m" not in st.session_state or len(st.session_state["a_m"]) != len(m_state):
        st.session_state["a_m"] = [1 / len(m_state)] * len(m_state)

    # synchroniser sliders si nécessaire
    if st.session_state.get("update_sliders_from_am", False):
        for i, m in enumerate(m_state):
            st.session_state[f"am_{m}"] = float(st.session_state["a_m"][i])
        st.session_state["update_sliders_from_am"] = False


    a_m_inputs = []
    for i, m in enumerate(m_state):
        a_m_inputs.append(
            st.slider(
                f"m = {m:+.1f}",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                key=f"am_{m}"
            )
        )



# === MAIN INTERFACE ===
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Distribution settings")
    variable = st.selectbox("Choose variable", ["Alignment", "Polarisation"])
    if variable == "Alignment":
        curve = "Symmetric Exponential"
    else:
        curve = "Exponential"

    # Affichage esthétique
    st.markdown(f"**Curve function selected:** `{curve}`")
    value_pct = st.number_input("Value [%]", value=0.0)

    st.subheader("Population aₘ")

    df_bar = pd.DataFrame(a_m_inputs, index=m_state)
    fig_bar = go.Figure([go.Bar(x=m_state, y=df_bar[0])])
    fig_bar.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        xaxis=dict(
            tickmode="array",
            tickvals=np.arange(min(m_state), max(m_state)+0.5, 1.0).tolist()
        )
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    polar, align = st.columns([1, 1])
    with polar:
        st.metric(label="Polarisation", value=f"{100*st.session_state.get('polarisation', 0):.2f}%")

    with align:
        st.metric(label="Alignement", value=f"{100*st.session_state.get('alignement', 0):.2f}%")

    generate_am_clicked = st.button("Generate aₘ")
    if generate_am_clicked:
        value = value_pct / 100

        a_m_generated, error_msg = generate_a_m_button(curve, variable, ji, value)
        if error_msg:
            st.error(error_msg)
        else:
            st.session_state["a_m"] = a_m_generated

            st.session_state["update_sliders_from_am"] = True 
            st.session_state["polarisation"], st.session_state["alignement"] = calculate_polarisation(ji, a_m_generated)
            st.experimental_rerun()

    st.markdown("---")
    st.subheader("Angular Distribution")

    fig = go.Figure()
    if "distribution" not in st.session_state:
        a_m = [1 / (2 * ji + 1)] * int(2 * ji + 1)
        st.session_state["even_distribution"] = calculate_distribution(ji, jf, delta, a_m)
        fig.add_trace(go.Scatterpolar(
            r=st.session_state["even_distribution"]["w"],
            theta=np.degrees(st.session_state["even_distribution"]["rads"]),
            mode='lines',
            name='Even Distribution',
            ))
        
        fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            polar=dict(
                domain=dict(x=[0, 1], y=[0, 1]),
                radialaxis=dict(visible=True, range=[0, max(st.session_state["even_distribution"]["w"])*1.1])
            ),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig.add_trace(go.Scatterpolar(
            r=st.session_state["distribution"]["w"],
            theta=np.degrees(st.session_state["distribution"]["rads"]),
            mode='lines',
            name='Angular Distribution'
        ))
    
        fig.add_trace(go.Scatterpolar(
            r=st.session_state["even_distribution"]["w"],
            theta=np.degrees(st.session_state["even_distribution"]["rads"]),
            mode='lines',
            name='Even Distribution',
            line=dict(color='black', dash='dash')
        ))

        fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            polar=dict(
                domain=dict(x=[0, 1], y=[0, 1]),
                radialaxis=dict(visible=True, range=[0, max(st.session_state["distribution"]["w"]) * 1.1])
            ),
            showlegend=True,
            legend=dict(
                x=1,
                y=1,
                xanchor="right",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.5)",  # Optionnel : fond semi-transparent

            )
        )
        st.plotly_chart(fig, use_container_width=True)

    calculate_clicked = st.button("Calculate")
    if calculate_clicked:
        try:
            st.session_state["distribution"] = calculate_distribution(ji, jf, delta, st.session_state["a_m"])
            st.experimental_rerun()

        except Exception as e:
            st.error(f"Erreur : {e}")

with col2:
    st.subheader("Geometry Settings")

    phantom_shape = st.selectbox("Choose phantom shape", ["Cylindrical", "Spherical"])

    st.markdown("**Phantom Dimensions**")

    if phantom_shape == "Spherical":
        Hauteur_P = st.number_input("Radius [cm]", value=3.2)
        Longueur_P = Hauteur_P
        # Ici on n'affiche que le rayon
    else:
        Hauteur_P = st.number_input("Radius [cm]", value=3.2)
        Longueur_P = st.number_input("Length [cm]", value=10.5)

    st.markdown("---")
    st.markdown("**Detector Dimensions**")
    Hauteur_D = st.number_input("H [cm]", value=0.7)
    Longueur_D = st.number_input("L [cm]", value=3.0)

    st.markdown("---")
    st.markdown("**Distances Dimensions**")
    d1 = st.number_input("d₁ [cm]", value=11.0)
    d2 = st.number_input("d₂ [cm]", value=14.0)

    geometry_fig = plot_geometry_plotly(phantom_shape, Longueur_P, Hauteur_P, Longueur_D, Hauteur_D, d1, d2)
    st.plotly_chart(geometry_fig, use_container_width=True)

st.markdown("---")
st.subheader("Simulation")
N = st.number_input("N", value=100000)
simulate_clicked = st.button("Simulate")
if simulate_clicked:
    try:
        if "a_m" not in st.session_state:
            st.warning("Please click Calculate first to generate distribution.")
        else:
            a_m = st.session_state["a_m"]
            distribution = calculate_distribution(ji, jf, delta, a_m)
            N1, N2 = simulation(N, distribution, phantom_shape, Longueur_P, Hauteur_P, Longueur_D, Hauteur_D, d1, d2)

            st.session_state["N1"] = N1
            st.session_state["N2"] = N2
            st.session_state["ProbD1"] = N1 / N * 100
            st.session_state["ProbD2"] = N2 / N * 100
            st.session_state["Asym"] = (N1 - N2) / (N1 + N2) * 100
    except Exception as e:
        st.error(f"Erreur pendant la simulation : {e}")


st.subheader("Results")
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.metric(label="N₁", value=st.session_state.get("N1", 0))
    st.metric(label="N₂", value=st.session_state.get("N2", 0))

with col2:
    st.metric(label="D1 hit probability", value=f"{st.session_state.get('ProbD1', 0):.2f}%")
    st.metric(label="D2 hit probability", value=f"{st.session_state.get('ProbD2', 0):.2f}%")

with col3:
    st.metric(label="Asymmetry", value=f"{st.session_state.get('Asym', 0):.2f}%")



try:
    # 1. Résumé de simulation
    result_data = {
        "Ji": [ji],
        "Jf": [jf],
        "delta": [delta],
        "N": [N],
        "N1": [st.session_state.get("N1", "")],
        "N2": [st.session_state.get("N2", "")],
        "Probability D1 [%]": [st.session_state.get("ProbD1", "")],
        "Probability D2 [%]": [st.session_state.get("ProbD2", "")],
        "Asymmetry [%]": [st.session_state.get("Asym", "")],
        "H_Phantom [cm]": [Hauteur_P],
        "L_Phantom [cm]": [Longueur_P],
        "H_Detector [cm]": [Hauteur_D],
        "L_Detector [cm]": [Longueur_D],
        "d1 [cm]": [d1],
        "d2 [cm]": [d2]
    }

    if "a_m" in st.session_state:
        for i, m in enumerate(np.arange(-ji, ji + 1, 1)):
            result_data[f"a (m={m:+.1f})"] = [st.session_state["a_m"][i]]

    df_results = pd.DataFrame(result_data)
    csv_results = df_results.to_csv(index=False)

    # 2. Distribution angulaire
    if "distribution" in st.session_state:
        distribution_save = pd.DataFrame({
            "theta_deg": np.degrees(st.session_state["distribution"]["rads"]),
            "w": st.session_state["distribution"]["w"]
        })
    else:
        distribution_save = pd.DataFrame({"theta_deg": [], "w": []})
    
    csv_dist = distribution_save.to_csv(index=False)

    # 3. Créer un fichier ZIP en mémoire
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        zip_file.writestr("simulation_summary.csv", csv_results)
        zip_file.writestr("angular_distribution.csv", csv_dist)

    zip_buffer.seek(0)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 4. Bouton de téléchargement
    st.download_button(
        label="Save",
        data=zip_buffer,
        file_name=f"simulation_data_{timestamp}.zip",
        mime="application/zip"
    )

except Exception as e:
    st.error(f"Erreur lors de la sauvegarde : {e}")


