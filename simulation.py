"""
Created on 2022

@author: Christian Doimo

This work is licensed under a CC0 1.0 license (https://creativecommons.org/publicdomain/zero/1.0/).

"""

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.dropdown import DropDown
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.progressbar import ProgressBar
from kivy.core.window import Window
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.properties import ObjectProperty
from kivy.clock import Clock
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import pandas as pd
from sympy import symbols, solve
from sympy.physics.quantum.cg import CG
from sympy.physics.wigner import racah
from scipy.special import eval_legendre

Window.size = (1100,600)

class Mypopup(Popup):
    pass

class MyProgressBar(Popup):
    def update(self, dt):
        if self.ids.progress_id.value < 100:
            self.ids.progress_id.value += 1
        else:
            self.dismiss()

    def start(self, N, distribution, longeur_P, largeur_P, hauteur_D, diametre_D, distance_D1, distance_D2):
        Clock.schedule_interval(self.update, 0.02)

class Simulation(Widget):
    exp = ObjectProperty(True)
    parab = ObjectProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        largeur_P = float(self.ids.Hauteur_P.text) # cm
        longeur_P = float(self.ids.Longueur_P.text) # cm
        hauteur_D = float(self.ids.Longueur_D.text) # cm
        diametre_D = float(self.ids.Hauteur_D.text) # cm
        distance_D1 = float(self.ids.d1_label.text) # cm
        distance_D2 = float(self.ids.d2_label.text) # cm

        self.creat_m_states()

        # Initialisation des graphiques
        self.fig0, ax0 = self.plot_geometry(longeur_P, largeur_P, hauteur_D, diametre_D, distance_D1, distance_D2)

        self.fig1, ax1 = plt.subplots(subplot_kw={'projection': 'polar'})
        ax1.plot()
        ax1.set_rmin(0)
        plt.yticks([0, 0.5, 1, 1.5, 2], fontsize=7)

        self.fig2, ax2 = plt.subplots()
        plt.bar(0, 0)
        plt.ylim(0,1.05)
        plt.yticks(fontsize=7)
        plt.xticks(fontsize=7)

        self.ids.geometrie.add_widget(FigureCanvasKivyAgg(self.fig0))
        self.ids.graph.add_widget(FigureCanvasKivyAgg(self.fig1))
        self.ids.population.add_widget(FigureCanvasKivyAgg(self.fig2))


    def a_m_exponentiel(self, ji, value, choix="Polarisation"):
        m_state = np.arange(-ji, ji + 1, 1)
        x = symbols("x", real=True)
        eq = 0
        i = 0
        if choix == "Polarisation":
            for m in m_state:
                eq += (value * ji - m) * x ** i
                i += 1

        elif choix == "Alignement":
            C = 1 / 3 * (ji + 1) / ji
            for m in m_state:
                eq += ((value + C) * ji ** 2 - m ** 2) * x ** i
                i += 1

        solution = solve(eq, x, rational=True)
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
                sum_exp += np.exp(-tau[t] * n)
            A = 1 / sum_exp
            for q in m_state:
                a_m.append(A * np.exp(-tau[t] * q))
            population["a_m" + str(t)] = a_m

        return population

    def a_m_parabole(self, ji, L):
        m_state = np.arange(-ji, ji + 1, 1)
        C = 1 / 3 * (ji + 1) / ji

        sum_m2 = 0
        sum_m4 = 0
        for m in m_state:
            sum_m2 += m ** 2
            sum_m4 += m ** 4
        B = (ji ** 2 * sum_m2) / (-12 * sum_m4 + sum_m2 ** 2)
        D = (-sum_m4 + C * ji ** 2 * sum_m2) / (-12 * sum_m4 + sum_m2 ** 2)

        L_limit_sup = (-D * ((-12 * 0.5 ** 2) / sum_m2 + 1) - ((0.5 ** 2) / sum_m2)) / (
                    B * (-12 * 0.5 ** 2) / sum_m2 + B)
        L_limit_inf = (-D * ((-12 * 5.5 ** 2) / sum_m2 + 1) - ((5.5 ** 2) / sum_m2)) / (
                    B * (-12 * 5.5 ** 2) / sum_m2 + B)

        population = pd.DataFrame()
        if L_limit_inf < L < L_limit_sup:
            h = (L * ji ** 2 * sum_m2 - sum_m4 + C * ji ** 2 * sum_m2) / (-12 * sum_m4 + sum_m2 ** 2)
            a = (1 - 12 * h) / sum_m2

            a_m = []
            for n in m_state:
                a_m.append(a * n ** 2 + h)

        else:
            print("Valeur d'alignement maximum : ", L_limit_sup)
            print("Valeur d'alignement minimum : ", L_limit_inf)
            a_m = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        population["a_m0"] = a_m
        return population

    def f_k(self, k, jf, L1, L2, ji):
        cg = CG(L1, 1, L2, -1, k, 0).doit()
        W = racah(ji, ji, L1, L2, k, jf)
        f_k = ((-1) ** (jf - ji - 1)) * (((2 * L1 + 1) * (2 * L2 + 1) * (2 * ji + 1)) ** (1 / 2)) * cg * W

        return f_k

    def calculate_distribution(self, ji, jf, delta, a_m):
        L1 = abs(ji - jf)
        L2 = abs(ji + jf)
        if L1 == L2:
            L2 = L1
        elif L2 > L1:
            L2 = L1 + 1
        ordre_k = np.linspace(0, 2 * int(L1), int(L1 + 1), dtype=int)
        m_state = np.arange(-ji, ji + 1, 1)
        A_k = []
        W = []
        angles = np.arange(0, 2 * math.pi + math.pi / 100, math.pi / 100)

        for k in ordre_k:
            somme = 0
            for i in range(len(m_state)):
                m = m_state[i]
                somme = somme + (((-1) ** (ji - m)) * CG(ji, m, ji, -m, k, 0).doit() * a_m[i])
            rho_k = (((2 * ji) + 1) ** (1 / 2)) * somme

            a_k = np.array(rho_k) * (1 / (1 + delta ** 2)) * (
                    self.f_k(k, jf, L1, L1, ji) + (2 * delta * self.f_k(k, jf, L1, L2, ji)) + (
                    (delta ** 2) * self.f_k(k, jf, L2, L2, ji)))
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

    def calculate_polarisation(self, ji, a_m):
        m_state = np.arange(-ji, ji + 1, 1)
        polarisation = 0.
        alignment = 0.
        for n in range(len(m_state)):
            polarisation += (m_state[n] / ji) * a_m[n]
            alignment += (m_state[n] / ji) ** 2 * a_m[n]
        alignment -= ((1 / 3) * ((ji + 1) / ji))

        return polarisation, alignment

    def plot_geometry(self, longeur_P, largeur_P, hauteur_D, diametre_D, distance_D1, distance_D2):
        phantom_x = [-longeur_P / 2, longeur_P / 2, longeur_P / 2, -longeur_P / 2, -longeur_P / 2]
        phantom_y = [largeur_P / 2, largeur_P / 2, -largeur_P / 2, -largeur_P / 2, largeur_P / 2]

        det1_x = [-hauteur_D / 2 + distance_D1, hauteur_D / 2 + distance_D1, hauteur_D / 2 + distance_D1,
                  -hauteur_D / 2 + distance_D1, -hauteur_D / 2 + distance_D1]
        det1_y = [diametre_D / 2, diametre_D / 2, -diametre_D / 2, -diametre_D / 2, diametre_D / 2]

        det2_y = [-hauteur_D / 2 + distance_D2, hauteur_D / 2 + distance_D2, hauteur_D / 2 + distance_D2,
                  -hauteur_D / 2 + distance_D2, -hauteur_D / 2 + distance_D2]
        det2_x = [diametre_D / 2, diametre_D / 2, -diametre_D / 2, -diametre_D / 2, diametre_D / 2]

        fig, ax = plt.subplots(figsize=(6.4 * 1, 6.4 * 1))
        plt.plot(phantom_x, phantom_y, label='Phantom')
        plt.plot(det1_x, det1_y, 'r', label='Detector 1')
        plt.plot(det2_x, det2_y, 'g', label='Detector 2')
        plt.plot([0, distance_D1], [0, 0], 'k--', linewidth=1.0)
        plt.plot([0, 0], [0, distance_D2], 'k--', linewidth=1.0)
        plt.legend()
        plt.axis("off")
        plt.xlim(-7, 17)
        plt.ylim(-7, 17)
        plt.text(7, -2, "$d_1$")
        plt.text(0.5, 7, "$d_2$")

        return fig, ax

    def simulation(self, N, distribution, longeur_P, largeur_P, hauteur_D, diametre_D, distance_D1, distance_D2):
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


    def creat_m_states(self):
        self.ids.layout.clear_widgets()

        self.ji = float(self.ids.ji.text)
        self.m_state = np.arange(-self.ji, self.ji + 1, 1)
        self.txtIn = []
        self.lbl = []

        for i in range(len(self.m_state)):
            self.lbl.append(Label(text='%.1f' % self.m_state[i], markup=True, color=(0, 0, 0), font_size=12,
                        size_hint_y=None, height=27,
                        size_hint_x=None, width=30,
                        pos_hint={"x": 0.1, "top": 0.99 - 0.075 * i}))
            self.ids.layout.add_widget(self.lbl[i])

            self.txtIn.append(TextInput(multiline=False, write_tab=False, input_filter='float',
                              size_hint_y=None, height=27,
                              size_hint_x=None, width=50,
                              pos_hint={"x": 0.4, "top": 0.99 - 0.075 * i}))
            self.ids.layout.add_widget(self.txtIn[i])


    def generate_am_button(self):
        alignement_max_exp = 0.6060606060606060
        alignement_max_parab = 0.3151515151515151
        alignement_min_parab = -0.2005509641873278
        try:
            self.creat_m_states()
            value = float(self.ids.value_id.text)/100
            try:

                if self.ids.choose_curve_id.text == 'Exponential':
                    if self.ids.choose_variable_id.text == 'Alignment':
                        if alignement_max_exp > value > 0:
                            population = self.a_m_exponentiel(self.ji, value, choix="Alignement")
                        else:
                            pop = Mypopup()
                            pop.title = f'Alignment can be only between {str(round(alignement_max_exp * 100, 2))}% and 0%'
                            pop.open()

                    elif self.ids.choose_variable_id.text == 'Polarisation':
                        if 1 > value > -1:
                            population = self.a_m_exponentiel(self.ji, value, choix="Polarisation")
                        else:
                            pop = Mypopup()
                            pop.title = f'Polarisation can be only between 100% and -100%'
                            pop.open()

                elif self.ids.choose_curve_id.text == 'Parabolic':
                    if self.ids.choose_variable_id.text == 'Alignment':
                        if alignement_max_parab > value > alignement_min_parab:
                            population = self.a_m_parabole(self.ji, value)
                        else:
                            pop = Mypopup()
                            pop.title = f'Alignment can be only between {str(round(alignement_max_parab * 100, 2))}% and {str(round(alignement_min_parab * 100, 2))}%'
                            pop.open()
                    elif self.ids.choose_variable_id.text == 'Polarisation':
                        pop = Mypopup()
                        pop.title = "There is no polarisation for parabolic curve"
                        pop.open()

                for p in range(len(population)):
                    self.txtIn[p].text = str(population.a_m0[p])

                self.fig2, ax2 = plt.subplots()
                plt.bar(self.m_state, population.a_m0)
                polarisation, alignment = self.calculate_polarisation(self.ji, population.a_m0)
                plt.title('Polarisation = {:.2f}% | Alignement = {:.2f}%'.format(polarisation * 100, alignment * 100),
                          fontsize=10)
                plt.yticks(fontsize=7)
                plt.xticks(self.m_state, fontsize=7)

                self.ids.population.add_widget(FigureCanvasKivyAgg(self.fig2))
                self.ids.ali.text = str(round(alignment * 100, 2))
                self.ids.pola.text = str(round(polarisation * 100, 2))
            except:
                pass

        except:
            pop = Mypopup()
            pop.title = "Enter J[size=20][sub]i[/sub][/size] and alignment value please"
            pop.open()

    def calculate_button(self):
        try:
            self.jf = float(self.ids.jf.text)
            self.delta = float(self.ids.delta.text)

            a_m = []
            for value in range(len(self.m_state)):
                a_m.append(float(self.txtIn[value].text))
            if round(sum(a_m),2) == 1:
                population_0 = self.a_m_exponentiel(self.ji, 0, choix="Alignement")

                dist_0 = self.calculate_distribution(self.ji, self.jf, self.delta, population_0.a_m0)
                self.distribution = self.calculate_distribution(self.ji, self.jf, self.delta, a_m)

                self.fig1, ax1 = plt.subplots(subplot_kw={'projection': 'polar'})
                ax1.plot(dist_0.rads, dist_0.w, 'k--')
                ax1.plot(self.distribution.rads, self.distribution.w)
                ax1.set_rmin(0)
                plt.yticks(fontsize=7)

                self.fig2, ax2 = plt.subplots()
                plt.bar(self.m_state, a_m)
                polarisation, alignment = self.calculate_polarisation(self.ji, a_m)
                plt.title('Polarisation = {:.2f}% | Alignement = {:.2f}%'.format(polarisation * 100, alignment * 100), fontsize=10)
                plt.yticks(fontsize=7)
                plt.xticks(self.m_state, fontsize=7)

                self.ids.graph.add_widget(FigureCanvasKivyAgg(self.fig1))
                self.ids.population.add_widget(FigureCanvasKivyAgg(self.fig2))
            else:
                pop = Mypopup()
                pop.title = "The sum of a_m must be equal to 1"
                pop.open()
        except:
            pop = Mypopup()
            pop.title = "Enter Ji, Jf, delta and a_m values please"
            pop.open()

    def simulation_button(self):
        try:
            N = int(self.ids.N_label.text)
            largeur_P = float(self.ids.Hauteur_P.text)  # cm
            longeur_P = float(self.ids.Longueur_P.text)  # cm
            hauteur_D = float(self.ids.Longueur_D.text)  # cm
            diametre_D = float(self.ids.Hauteur_D.text)  # cm
            distance_D1 = float(self.ids.d1_label.text)  # cm
            distance_D2 = float(self.ids.d2_label.text)  # cm

            self.fig0, ax0 = self.plot_geometry(longeur_P, largeur_P, hauteur_D, diametre_D, distance_D1, distance_D2)
            self.ids.geometrie.add_widget(FigureCanvasKivyAgg(self.fig0))

            N1, N2 = self.simulation(N, self.distribution, longeur_P, largeur_P, hauteur_D, diametre_D, distance_D1, distance_D2)

            incertitude_N1 = np.sqrt(N1)
            incertitude_N2 = np.sqrt(N2)

            self.ids.N1_label.text = f'{str(N1)} \u00B1 {str(round(incertitude_N1, 2))}'
            self.ids.N2_label.text = f'{str(N2)} \u00B1 {str(round(incertitude_N2, 2))}'

            prob_D1 = (N1 / N) * 100
            incertitude_prob_D1 = np.sqrt(N1) / N * 100
            prob_D2 = (N2 / N) * 100
            incertitude_prob_D2 = np.sqrt(N2) / N * 100
            self.ids.ProbD1_label.text = f'{str(round(prob_D1, 2))} \u00B1 {str(round(incertitude_prob_D1, 2))}%'
            self.ids.ProbD2_label.text = f'{str(round(prob_D2, 2))} \u00B1 {str(round(incertitude_prob_D2, 2))}%'

            asymetrie = (N1 - N2) / (N1 + N2) * 100
            incertitude_diff = np.sqrt((2 * N2 / (N1 + N2) ** 2) ** 2 * N1 + (-2 * N1 / (N1 + N2) ** 2) ** 2 * N2) * 100

            self.ids.Asymetrie.text = f'{str(round(asymetrie, 2))} \u00B1 {str(round(incertitude_diff, 2))}%'
        except:
            pop = Mypopup()
            pop.title = "Calculate distribution and enter N value please"
            pop.open()

    def save_button(self):
        # save figure
        self.fig0.savefig("geometry.png")
        self.fig1.savefig("distribution.png")
        self.fig2.savefig("population.png")
        self.distribution.to_csv('data_distribution.csv')

class SimulationApp(App):
    def build(selfs):
        return Simulation()

if __name__== '__main__':
    SimulationApp().run()