import numpy as np
import matplotlib.pyplot as plt
from sympy.physics.quantum.cg import CG
from sympy.physics.wigner import racah
from sympy import symbols, solve
from scipy.special import eval_legendre
import pandas as pd
import math
import plotly.graph_objects as go
import random
from scipy.optimize import root_scalar

def a_m_exponentiel(ji, value, choix="Polarisation"):
    m_state = np.arange(-ji, ji + 1, 1)
    x = symbols("x", real=True)
    eq = 0
    i = 0

    if choix == "Polarisation":
        for m in m_state:
            eq += (value * ji - m) * x ** i
            i += 1
    elif choix == "Alignement":
        C = (1 / 3) * (ji + 1) / ji
        for m in m_state:
            eq += ((value + C) * ji ** 2 - m ** 2) * x ** i
            i += 1

    solutions = solve(eq, x, rational=True)
    tau = []
    for sol in solutions:
        try:
            s = float(sol)
            if s > 0:
                tau.append(-np.log(s))
        except:
            pass

    population = pd.DataFrame()
    for t in tau:
        a_m = []
        sum_exp = sum(np.exp(-t * m) for m in m_state)
        A = 1 / sum_exp
        for m in m_state:
            a_m.append(A * np.exp(-t * m))
        population[f"a_m0"] = a_m
        break  # on ne garde que la première solution

    return population

def a_m_exponentiel_symmetric(ji, value):
    m_state = np.arange(-ji, ji + 1, 1)
    C = (1 / 3) * (ji + 1) / ji

    # Fonction résidu à minimiser pour trouver t
    def residu(t):
        num = sum(m**2 * (np.exp(-t * m) + np.exp(t * m)) for m in m_state)
        den = sum(np.exp(-t * m) + np.exp(t * m) for m in m_state)
        align_calc = num / den
        return align_calc - (value + C) * ji**2

    # Recherche t numériquement
    sol = root_scalar(residu, bracket=[1e-6, 10], method='brentq')
    t_found = sol.root

    # Construire la population normalisée
    sum_exp = sum(np.exp(-t_found * m) + np.exp(t_found * m) for m in m_state)
    A = 1 / sum_exp
    a_m = [A * (np.exp(-t_found * m) + np.exp(t_found * m)) for m in m_state]

    population = pd.DataFrame()
    population["a_m0"] = a_m
    return population

def a_m_parabole(ji, L):
    m_state = np.arange(-ji, ji + 1, 1)
    C = (1 / 3) * (ji + 1) / ji

    sum_m2 = sum(m**2 for m in m_state)
    sum_m4 = sum(m**4 for m in m_state)

    B = (ji**2 * sum_m2) / (-12 * sum_m4 + sum_m2**2)
    D = (-sum_m4 + C * ji**2 * sum_m2) / (-12 * sum_m4 + sum_m2**2)

    L_limit_sup = (-D * ((-12 * 0.5**2) / sum_m2 + 1) - (0.5**2 / sum_m2)) / (B * (-12 * 0.5**2) / sum_m2 + B)
    L_limit_inf = (-D * ((-12 * 5.5**2) / sum_m2 + 1) - (5.5**2 / sum_m2)) / (B * (-12 * 5.5**2) / sum_m2 + B)

    population = pd.DataFrame()
    if L_limit_inf < L < L_limit_sup:
        h = (L * ji**2 * sum_m2 - sum_m4 + C * ji**2 * sum_m2) / (-12 * sum_m4 + sum_m2**2)
        a = (1 - 12 * h) / sum_m2
        a_m = [a * m**2 + h for m in m_state]
    else:
        a_m = [0] * len(m_state)

    population["a_m0"] = a_m
    return population

def f_k(k, jf, L1, L2, ji):
    cg = CG(L1, 1, L2, -1, k, 0).doit()
    W = racah(ji, ji, L1, L2, k, jf)
    return ((-1) ** (jf - ji - 1)) * (((2 * L1 + 1) * (2 * L2 + 1) * (2 * ji + 1)) ** (1 / 2)) * cg * W

def calculate_distribution(ji, jf, delta, a_m):
    L1 = abs(ji - jf)
    L2 = L1 + 1 if ji != jf else L1
    ordre_k = np.linspace(0, 2 * int(L1), int(L1 + 1), dtype=int)
    m_state = np.arange(-ji, ji + 1, 1)
    A_k = []
    angles = np.arange(0, 2 * math.pi + math.pi / 100, math.pi / 100)

    for k in ordre_k:
        somme = sum(((-1) ** (ji - m)) * CG(ji, m, ji, -m, k, 0).doit() * a_m[i] for i, m in enumerate(m_state))
        rho_k = (((2 * ji) + 1) ** 0.5) * somme
        a_k = rho_k * (1 / (1 + delta ** 2)) * (
            f_k(k, jf, L1, L1, ji) + 2 * delta * f_k(k, jf, L1, L2, ji) + (delta ** 2) * f_k(k, jf, L2, L2, ji))
        
        A_k.append(float(a_k))

    W = [sum(A_k[j] * eval_legendre(ordre_k[j], np.cos(theta)) for j in range(len(ordre_k))) for theta in angles]

    return pd.DataFrame({'rads': angles, 'w': W})

def calculate_polarisation(ji, a_m):
    m_state = np.arange(-ji, ji + 1, 1)
    polarisation = sum((m / ji) * a_m[i] for i, m in enumerate(m_state))
    alignment = sum(((m / ji) ** 2) * a_m[i] for i, m in enumerate(m_state))
    alignment -= (1 / 3) * ((ji + 1) / ji)
    return polarisation, alignment

def generate_figures(ji, jf, delta, a_m):
    dist_0 = calculate_distribution(ji, jf, delta, [1/(2*ji+1)] * int(2*ji+1))
    dist = calculate_distribution(ji, jf, delta, a_m)
    m_state = np.arange(-ji, ji + 1, 1)

    fig1, ax1 = plt.subplots(subplot_kw={'projection': 'polar'})
    ax1.plot(dist_0.rads, dist_0.w, 'k--', label='Isotropic')
    ax1.plot(dist.rads, dist.w, label='Oriented')
    ax1.set_rmin(0)
    ax1.legend(fontsize=6)

    polarisation, alignment = calculate_polarisation(ji, a_m)


    return fig1, polarisation, alignment

def generate_a_m_button(curve, variable, ji, value):
    """
    Génère les coefficients a_m en fonction des paramètres choisis.

    Paramètres :
        curve (str) : "Exponential" ou "Parabolic"
        variable (str) : "Polarisation" ou "Alignment"
        ji (float) : spin initial
        value (float) : valeur en proportion (ex: 0.5 pour 50%)

    Retourne :
        a_m_generated (np.ndarray) : les coefficients générés
        error_msg (str ou None) : message d'erreur s'il y a lieu
    """
    alignement_max_exp = 0.6060606060606060
    alignement_max_parab = 0.3151515151515151
    alignement_min_parab = -0.2005509641873278

    try:
        if ji is None or value is None:
            return None, "Please enter a value for Jᵢ and alignment/polarisation."

        if curve == "Exponential":
            if variable == "Alignment":
                if 0 < value < alignement_max_exp:
                    df = a_m_exponentiel(ji, value, choix="Alignement")
                    return df.iloc[:, 0].values, None
                else:
                    return None, f"Alignment must be between 0% and {round(alignement_max_exp * 100, 2)}% for exponential curve."

            elif variable == "Polarisation":
                if -1 < value < 1:
                    df = a_m_exponentiel(ji, value, choix="Polarisation")
                    return df.iloc[:, 0].values, None
                else:
                    return None, "Polarisation must be between -100% and 100% for exponential curve."

        elif curve == "Symmetric Exponential":
            if variable == "Alignment":
                if alignement_min_parab < value < alignement_max_parab:
                    df = a_m_exponentiel_symmetric(ji, value)
                    return df.iloc[:, 0].values, None
                else:
                    return None, f"Alignment must be between {round(alignement_min_parab * 100, 2)}% and {round(alignement_max_parab * 100, 2)}% for parabolic curve."
            elif variable == "Polarisation":
                return None, "Parabolic curve does not support Polarisation."

        return None, "Invalid combination of curve and variable."

    except Exception as e:
        return None, f"Unexpected error: {str(e)}"
    

def plot_geometry_plotly(phantom_shape, longueur_P, largeur_P, hauteur_D, diametre_D, distance_D1, distance_D2):
    fig = go.Figure()
    all_x = []
    all_y = []

    # Phantom shape
    if phantom_shape == "Cylindrical":
        phantom_x = [-longueur_P / 2, longueur_P / 2, longueur_P / 2, -longueur_P / 2, -longueur_P / 2]
        phantom_y = [largeur_P / 2, largeur_P / 2, -largeur_P / 2, -largeur_P / 2, largeur_P / 2]
    elif phantom_shape == "Spherical":
        theta = np.linspace(0, 2 * np.pi, 100)
        phantom_x = largeur_P * np.cos(theta)
        phantom_y = largeur_P * np.sin(theta)

    fig.add_trace(go.Scatter(x=phantom_x, y=phantom_y, mode='lines', name='Phantom', line=dict(color='blue')))
    all_x.extend(phantom_x)
    all_y.extend(phantom_y)

    # Detector 1 (horizontal)
    det1_x = [-hauteur_D / 2 + distance_D1, hauteur_D / 2 + distance_D1, hauteur_D / 2 + distance_D1,
              -hauteur_D / 2 + distance_D1, -hauteur_D / 2 + distance_D1]
    det1_y = [diametre_D / 2, diametre_D / 2, -diametre_D / 2, -diametre_D / 2, diametre_D / 2]
    fig.add_trace(go.Scatter(x=det1_x, y=det1_y, mode='lines', name='Detector 1', line=dict(color='red')))
    all_x.extend(det1_x)
    all_y.extend(det1_y)

    # Detector 2 (vertical)
    det2_y = [-hauteur_D / 2 + distance_D2, hauteur_D / 2 + distance_D2, hauteur_D / 2 + distance_D2,
              -hauteur_D / 2 + distance_D2, -hauteur_D / 2 + distance_D2]
    det2_x = [diametre_D / 2, diametre_D / 2, -diametre_D / 2, -diametre_D / 2, diametre_D / 2]
    fig.add_trace(go.Scatter(x=det2_x, y=det2_y, mode='lines', name='Detector 2', line=dict(color='green')))
    all_x.extend(det2_x)
    all_y.extend(det2_y)

    # Lines d1 et d2
    fig.add_trace(go.Scatter(x=[0, distance_D1], y=[0, 0], mode='lines', name='d₁', line=dict(color='black', dash='dash')))
    fig.add_trace(go.Scatter(x=[0, 0], y=[0, distance_D2], mode='lines', name='d₂', line=dict(color='black', dash='dash')))
    all_x.extend([0, distance_D1, 0])
    all_y.extend([0, 0, distance_D2])

    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)
    x_margin = (xmax - xmin) * 0.1 if xmax != xmin else 1
    y_margin = (ymax - ymin) * 0.1 if ymax != ymin else 1

    # Layout
    fig.update_layout(
        width=700,
        height=600,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(range=[-7, 17], visible=False),
        yaxis=dict(range=[-7, 17], visible=False),
        showlegend=True
    )

    fig.add_annotation(x=(xmin + xmax)/2, y=ymin - y_margin/2, text="d₁", showarrow=False, font=dict(size=16))
    fig.add_annotation(x=xmin - x_margin/2, y=(ymin + ymax)/2, text="d₂", showarrow=False, font=dict(size=16))

    return fig

def simulation(N, distribution, phantom_shape, longeur_P, largeur_P, hauteur_D, diametre_D, distance_D1, distance_D2):
    # droite du detecteur 1
    x1_D1 = distance_D1
    x2_D1 = distance_D1 + hauteur_D
    y1_D1 = -diametre_D / 2
    y2_D1 = diametre_D / 2

    # droite du detecteur 2
    y1_D2 = distance_D2
    y2_D2 = distance_D2 + hauteur_D
    x1_D2 = -diametre_D / 2
    x2_D2 = diametre_D / 2

    N1 = 0  # Nombre de photon détecté par le détecteur 1
    N2 = 0  # Nombre de photon détecté par le détecteur 2

    for coups in range(N):
        if phantom_shape == "Spherical":
            # Tir aléatoire dans un disque
            R = largeur_P
            r = R * np.sqrt(random.uniform(0, 1))
            theta = 2 * np.pi * random.uniform(0, 1)
            x0 = r * np.cos(theta)
            y0 = r * np.sin(theta)
        else:
            # Cylindrique
            x0 = random.randint(-int(longeur_P / 2), int(longeur_P / 2))
            y0 = random.randint(-int(largeur_P / 2), int(largeur_P / 2))
        random_rads = random.choices(distribution.rads, distribution.w)

        pente = np.tan(random_rads)
        #   DETECTION POUR D1
        I1_D1 = pente * (x1_D1 - x0) + y0
        I2_D1 = pente * (x2_D1 - x0) + y0
        if pente != 0:
            I3_D1 = (y1_D1 - y0) / pente + x0
            I4_D1 = (y2_D1 - y0) / pente + x0

        if y1_D1 < I1_D1 < y2_D1 or y1_D1 < I2_D1 < y2_D1 or x1_D1 < I3_D1 < x2_D1 or x1_D1 < I4_D1 < x2_D1:
            N1 += 1

        #   DETECTION POUR D2
        I3_D2 = pente * (x1_D2 - x0) + y0
        I4_D2 = pente * (x2_D2 - x0) + y0
        if pente != 0:
            I1_D2 = (y1_D2 - y0) / pente + x0
            I2_D2 = (y2_D2 - y0) / pente + x0

        if y1_D2 < I3_D2 < y2_D2 or y1_D2 < I4_D2 < y2_D2 or x1_D2 < I1_D2 < x2_D2 or x1_D2 < I2_D2 < x2_D2:
            N2 += 1

    return N1, N2