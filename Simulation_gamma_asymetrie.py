"""
Created on Wed Jun 22 16:08:37 2022

@author: Christian Doimo and Michael Da Silva Magalhaes

This work is licensed under a CC0 1.0 license (https://creativecommons.org/publicdomain/zero/1.0/).

"""

import numpy as np
import random
import math
import pandas as pd
from sympy import symbols, solve
from sympy.physics.quantum.cg import CG
from sympy.physics.wigner import racah
from scipy.special import eval_legendre  
import matplotlib.pyplot as plt

def a_m_exponentiel(ji, value, choix = "Polarisation"):   
    m_state = np.arange(-ji,ji+1,1)
    x = symbols("x", real = True)
    eq = 0
    i = 0
    if choix == "Polarisation":
        for m in m_state:
            eq += (value*ji-m)*x**i
            i += 1
            
    elif choix == "Alignement":
        C = 1/3*(ji+1)/ji
        for m in m_state:
            eq += ((value+C)*ji**2-m**2)*x**i
            i += 1
    
    solution = solve(eq, x, rational = True)
    tau = []
    for sol in solution:
        try:
            s = float(sol)
        except:
            pass
        if s > 0:
            tau.append(- np.log(float(s)))
            
    population = pd.DataFrame()
    for t in range(len(tau)):
        a_m = []
        sum_exp = 0
        for n in m_state:
            sum_exp += np.exp(-tau[t]*n)
        A = 1/sum_exp
        for q in m_state:
            a_m.append(A*np.exp(-tau[t]*q))
        population["a_m"+str(t)] = a_m
        
    return population

def a_m_parabole(ji, L): 
    m_state = np.arange(-ji,ji+1,1)
    C = 1/3*(ji+1)/ji 
    
    sum_m2 = 0
    sum_m4 = 0
    for m in m_state:
        sum_m2 += m**2
        sum_m4 += m**4
    B = (ji**2*sum_m2)/(-12*sum_m4+sum_m2**2)
    D = (-sum_m4+C*ji**2*sum_m2)/(-12*sum_m4+sum_m2**2)
    
    L_limit_sup = (-D*((-12*0.5**2)/sum_m2+1)-((0.5**2)/sum_m2))/(B*(-12*0.5**2)/sum_m2+B)
    L_limit_inf = (-D*((-12*5.5**2)/sum_m2+1)-((5.5**2)/sum_m2))/(B*(-12*5.5**2)/sum_m2+B)
    
    population = pd.DataFrame()
    if L_limit_inf < L < L_limit_sup:
        h = (L*ji**2*sum_m2-sum_m4+C*ji**2*sum_m2)/(-12*sum_m4+sum_m2**2)
        a = (1-12*h)/sum_m2
        
        a_m = []
        for n in m_state:
            a_m.append(a*n**2+h)
            
    else:
        print("Valeur d'alignement maximum : ", L_limit_sup)
        print("Valeur d'alignement minimum : ", L_limit_inf)
        a_m = [0,0,0,0,0,0,0,0,0,0,0,0]
    
    population["a_m0"] = a_m
    return population

def f_k(k,jf,L1,L2,ji):
    cg = CG(L1,1,L2,-1,k,0).doit()
    W = racah(ji,ji,L1,L2,k,jf)
    f_k = ((-1)**(jf-ji-1)) * (((2*L1+1)*(2*L2+1)*(2*ji+1))**(1/2)) * cg * W

    return f_k

def calculate_distribution(ji, jf, delta, a_m):
    L1 = abs(ji-jf)
    L2 = abs(ji+jf)
    if L1 == L2: L2 = L1
    elif L2 > L1: L2 = L1+1
    ordre_k = np.linspace(0,2*int(L1),int(L1+1), dtype=int)
    m_state = np.arange(-ji,ji+1,1)
    A_k = []
    W = []
    angles = np.arange(0, 2*math.pi+math.pi/100, math.pi/100)
    
    for k in ordre_k:
        somme = 0
        for i in range(len(m_state)):
            m = m_state[i]
            somme = somme + (((-1)**(ji-m)) * CG(ji,m,ji,-m,k,0).doit() * a_m[i])
        rho_k = (((2*ji)+1)**(1/2)) * somme
    
        a_k = np.array(rho_k) * (1/(1+delta**2)) * (f_k(k,jf,L1,L1,ji) + (2*delta*f_k(k,jf,L1,L2,ji)) + ((delta**2)*f_k(k,jf,L2,L2,ji)))
        A_k.append(float(a_k))
        
    for theta in angles:
        w_theta = 0
        for j in range(len(ordre_k)):
            w_theta += (A_k[j] * eval_legendre(ordre_k[j], np.cos(theta)))
            
        W.append(w_theta)
        
    distribution = pd.DataFrame()
    distribution["rads"] = angles
    distribution["w"] = W
    
    return distribution

def calculate_polarisation(ji, a_m):
    m_state = np.arange(-ji,ji+1,1)
    polarisation = 0.
    alignment = 0.
    for n in range(len(m_state)):
        polarisation += (m_state[n] / ji) * a_m[n]
        alignment += (m_state[n] / ji) ** 2 * a_m[n]
    alignment -= ((1 / 3) * ((ji + 1) / ji))

    return polarisation, alignment

def plot_geometry(longeur_P, largeur_P, hauteur_D, diametre_D, distance_D1, distance_D2):
    phantom_x = [-longeur_P / 2, longeur_P / 2, longeur_P / 2, -longeur_P / 2, -longeur_P / 2]
    phantom_y = [largeur_P / 2, largeur_P / 2, -largeur_P / 2, -largeur_P / 2, largeur_P / 2]

    det1_x = [-hauteur_D / 2 + distance_D1, hauteur_D / 2 + distance_D1, hauteur_D / 2 + distance_D1,
              -hauteur_D / 2 + distance_D1, -hauteur_D / 2 + distance_D1]
    det1_y = [diametre_D / 2, diametre_D / 2, -diametre_D / 2, -diametre_D / 2, diametre_D / 2]

    det2_y = [-hauteur_D / 2 + distance_D2, hauteur_D / 2 + distance_D2, hauteur_D / 2 + distance_D2,
              -hauteur_D / 2 + distance_D2, -hauteur_D / 2 + distance_D2]
    det2_x = [diametre_D / 2, diametre_D / 2, -diametre_D / 2, -diametre_D / 2, diametre_D / 2]

    fig, ax = plt.subplots(figsize=(6.4*1, 6.4*1))
    plt.plot(phantom_x, phantom_y, label='Phantom')
    plt.plot(det1_x, det1_y, 'r', label='Détecteur 1')
    plt.plot(det2_x, det2_y, 'g', label='Détecteur 2')
    plt.title("Géométrie")
    plt.legend()
    plt.xlabel("cm")
    plt.ylabel("cm")
    plt.xlim(-7, 17)
    plt.ylim(-7, 17)

    return fig, ax

def simulation(N, distribution, longeur_P, largeur_P, hauteur_D, diametre_D, distance_D1, distance_D2):
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

#-------------------------------Variables--------------------------------------
ji = 11/2
jf = 3/2
delta = 0

value = 0.18

population_0 = a_m_exponentiel(ji, 0, choix = "Alignement")   
population = a_m_exponentiel(ji, value, choix = "Polarisation")             # Génère une population selon une expondentiel
#population = a_m_parabole(ji, alignement)                                       # Génère une population selon une parabole

#--------------------------------Calculs---------------------------------------
dist_0 = calculate_distribution(ji, jf, delta, population_0.a_m0)
distribution = calculate_distribution(ji, jf, delta, population.a_m0)

#---------------------------------Plot-----------------------------------------

#Plot population
pop = [population]
m = np.arange(-ji,ji+1,1)
for p in pop:
    for a_m in p:
        fig_population, ax_population = plt.subplots()
        plt.bar(m, p[a_m])
        polarisation, alignment = calculate_polarisation(ji, p[a_m])
        plt.title('Polarisation = {:.2f}% | Alignement = {:.2f}%'.format(polarisation*100, alignment*100))
        plt.ylim(0,1.05)
        plt.xticks(m)

# Plot distribution
fig_distribution, ax_distribution = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8,6))
plt.plot(dist_0.rads, dist_0.w, 'k--', label="L = 0% | P = 0%")
plt.plot(distribution.rads, distribution.w, label="L = {:.2f}% | P = {:.2f}%".format(alignment*100, polarisation*100))
plt.legend(bbox_to_anchor=(1.325, 1))
plt.title("Distribution angulaire")

#%%-------------------------Setup géométrie------------------------------------

# Dimension phantom
longeur_P = 10.5 #cm
largeur_P = 3.2 #cm

# Dimension détecteur
diametre_D = 0.7 #cm
hauteur_D = 3 #cm
distance_D1 = 11 #cm
distance_D2 = 14 #cm

# Plot géométrie
fig_geometrie, ax_geometrie = plot_geometry(longeur_P, largeur_P, hauteur_D, diametre_D, distance_D1, distance_D2)
#%%---------------------------Simulation---------------------------------------

N = 100000
N1, N2 = simulation(N, distribution, longeur_P, largeur_P, hauteur_D, diametre_D, distance_D1, distance_D2)
print("N1 = ", N1, "+/- {:.2f}" .format(np.sqrt(N1)))
print("N2 = ", N2, "+/- {:.2f}" .format(np.sqrt(N2)))
print("Probabilité de toucher le détecteur 1 = {:.2f}" .format((N1/N)*100),"+/- {:.2f}" .format(np.sqrt(N1)/N*100),"%")
print("Probabilité de toucher le détecteur 2 = {:.2f}" .format((N2/N)*100),"+/- {:.2f}" .format(np.sqrt(N2)/N*100),"%")
difference = (N1-N2)/(N1+N2)*100
incertitude_diff = np.sqrt((2*N2/(N1+N2)**2)**2*N1+(-2*N1/(N1+N2)**2)**2*N2)*100
print("Différence = {:.2f}" .format(difference),"+/- {:.2f}" .format(incertitude_diff),"%")

# différence à 0% de polarisation et 0 % d'alignement = 5% donc :
asymetrie = difference - 5
print("Asymétrie = {:.2f}".format(asymetrie),"+/- {:.2f}" .format(incertitude_diff),"%")

#%%------------------------------Save------------------------------------------

fig_distribution.savefig("distribution.png")
fig_population.savefig("population.png")
fig_geometrie.savefig("geometry.png")
distribution.to_csv('data_distribution.csv')