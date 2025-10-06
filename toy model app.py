import matplotlib.pyplot as plt
import streamlit as st
import time
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import itertools
from bokeh.plotting import figure, show
from bokeh.palettes import Category10
from bokeh.layouts import column
from bokeh.models import Legend
import concurrent.futures
from streamlit_bokeh import streamlit_bokeh


# ------------------- Page Config ------------------- #
st.set_page_config(page_title="Bacteria Growth Models", layout="wide")
st.markdown(
    '<h1 style="font-size:32px; text-align:center;">Bacteria Growth Models under Different Antibiotic Concentrations and Volumes</h1>',
    unsafe_allow_html=True
)

with st.expander("About this App", expanded=False):
    st.markdown("""
    This application simulates and visualizes the growth of bacterial populations under varying antibiotic concentrations and droplet volumes using three different mathematical models.
    Adjust the parameters in the sidebar to see how they affect the growth dynamics.
    """)

# ------------------- Debounce Setup ------------------- #
if "last_change_time" not in st.session_state:
    st.session_state.last_change_time = time.time()

DEBOUNCE_DELAY = 0.6  # seconds

def input_changed():
    st.session_state.last_change_time = time.time()

# ------------------- Sidebar ------------------- #
st.sidebar.header("Model Parameters")

with st.sidebar.expander("Growth and Lysis Parameters", expanded=False):
    mu_max = st.number_input("Maximum Growth Rate (mu_max)", 0.0, 2.0, 0.7, 0.05, format="%.2f", on_change=input_changed)
    Y = st.number_input("Yield Coefficient (Y)", 0.0001, 0.5, 0.001, 0.0001, format="%.4f", on_change=input_changed)
    S0 = st.number_input("Initial substrate (S0)", 0.1, 5.0, 1.0, 0.1, format="%.2f", on_change=input_changed)
    Ks = st.number_input("Half-saturation Constant (Ks)", 0.1, 10.0, 2.0, 0.1, format="%.2f", on_change=input_changed)

with st.sidebar.expander("Antibiotic Parameters", expanded=False):
    K_on = st.number_input("Antibiotic binding rate (K_on)", 0, 10000, 750, 10, on_change=input_changed)
    K_off = st.number_input("Antibiotic unbinding rate (K_off)", 0.0, 100.0, 0.01, 0.001, format="%.3f", on_change=input_changed)
    K_D = st.number_input("Dissociation constant (K_D)", 0, 50000, 12000, 1, on_change=input_changed)
    lambda_max = st.number_input("Maximum Lysis Rate (lambda_max)", 0.0, 10.0, 1.0, 0.1, format="%.2f", on_change=input_changed)
    K_A0 = st.number_input("Initial antibiotic constant (K_A0)", 0, 50, 10, 1, on_change=input_changed)
    n = st.number_input("Hill coefficient (n)", 1, 50, 20, 1, on_change=input_changed)
    a = st.number_input("Baseline Lysis (a)", 0.0, 10.0, 3.0, 0.1, format="%.2f", on_change=input_changed)
    b = st.number_input("Growth-dependent Lysis (b)", 0.0, 10.0, 0.1, 0.05, format="%.2f", on_change=input_changed)

# ------------------- Debounce Check ------------------- #
while time.time() - st.session_state.last_change_time < DEBOUNCE_DELAY:
    time.sleep(0.1)

def effective_concentration_model(y, t, mu_max, Ks, Y, K_on, K_off, lambda_max, V, K_D, n):
    """
    params:
    param y: vector of state variables [A_free, A_bound_live, B_live, B_dead, S]
    param t: time
    param mu_max: maximum growth rate of bacteria
    param Ks: half-saturation constant for substrate
    param Y: yield coefficient (biomass produced per unit substrate consumed)
    param K_on: rate constant for antibiotic binding to live bacteria
    param K_off: rate constant for antibiotic unbinding from live bacteria
    param lambda_max: maximum lysis rate due to antibiotic
    param V: volume of the droplet
    param K_D: dissociation constant for antibiotic effect
    param n: Hill coefficient for antibiotic effect
    variables:
    A_free: concentration of free antibiotic
    A_bound_live: concentration of antibiotic bound to live bacteria
    B_live: number of live bacteria
    B_dead: number of dead bacteria
    S: concentration of substrate
    """
    A_free, A_bound_live, B_live, B_dead, S = y
    density=B_live/V
    A_eff = A_bound_live / density
    mu = mu_max * S / (Ks + S)
    lambda_D = lambda_max * (A_eff ** n / (K_D ** n + A_eff ** n))
    dB_live_dt = (mu - lambda_D) * B_live
    dB_dead_dt = lambda_D * B_live
    dS_dt = - (1 / Y) * mu * density
    dA_free_dt = -K_on * A_free * density + K_off * A_bound_live + lambda_D * A_bound_live
    dA_bound_live_dt = K_on * A_free * density - K_off * A_bound_live - lambda_D * A_bound_live
    return np.array([dA_free_dt, dA_bound_live_dt, dB_live_dt, dB_dead_dt, dS_dt])

def linear_Lysis_rate_model(y, t, mu_max, Ks, Y, a, b, V,A0,K_A0,n):
    """
        params:
    param y: vector of state variables [B_live, B_dead, S]
    param t: time
    param mu_max: maximum growth rate of bacteria
    param Ks: half-saturation constant for substrate
    param Y: yield coefficient (biomass produced per unit substrate consumed)
    param a: baseline lysis rate
    param b: coefficient for growth-rate-dependent lysis
    param V: volume of the droplet
    param A0: initial antibiotic concentration
    param K_A0: dissociation constant for antibiotic effect
    param n: Hill coefficient for antibiotic effect
    variables:
    B_live: number of live bacteria
    B_dead: number of dead bacteria
    S: concentration of substrate
    """
    B_live, B_dead, S = y
    density=B_live/V
    mu = mu_max * S / (Ks + S)
    lambda_D = a*(A0**n / (K_A0**n + A0**n))*mu+b
    dB_live_dt = (mu - lambda_D) * B_live
    dB_dead_dt = lambda_D * B_live
    dS_dt = - (1 / Y) * mu * density
    return np.array([dB_live_dt, dB_dead_dt, dS_dt])

def combined_model (y, t, mu_max, Ks, Y, K_on, K_off, V, K_D, n, a, b):
    """
    params:
    param y: vector of state variables [A_free, A_bound_live, B_live, B_dead, S]
    param t: time
    param mu_max: maximum growth rate of bacteria
    param Ks: half-saturation constant for substrate
    param Y: yield coefficient (biomass produced per unit substrate consumed)
    param K_on: rate constant for antibiotic binding to live bacteria
    param K_off: rate constant for antibiotic unbinding from live bacteria
    param lambda_max: maximum lysis rate due to antibiotic
    param V: volume of the droplet
    param K_D: dissociation constant for antibiotic effect based on effective concentration
    param n: Hill coefficient for antibiotic effect based on effective concentration
    param a: baseline lysis rate
    param b: coefficient for growth-rate-dependent lysis
    variables:
    A_free: concentration of free antibiotic
    A_bound_live: concentration of antibiotic bound to live bacteria
    B_live: number of live bacteria
    B_dead: number of dead bacteria
    S: concentration of substrate
    """
    A_free, A_bound_live, B_live, B_dead, S = y
    density=B_live/V
    A_eff = A_bound_live / density
    mu = mu_max * S / (Ks + S)
    lambda_D = a*(A_eff ** n / (K_D ** n + A_eff ** n))*mu+b
    dB_live_dt = (mu - lambda_D) * B_live
    dB_dead_dt = lambda_D * B_live
    dS_dt = - (1 / Y) * mu * density
    dA_free_dt = -K_on * A_free * density + K_off * A_bound_live + lambda_D * A_bound_live
    dA_bound_live_dt = K_on * A_free * density - K_off * A_bound_live - lambda_D * A_bound_live
    return np.array([dA_free_dt, dA_bound_live_dt, dB_live_dt, dB_dead_dt, dS_dt])



def run_model_for_volumes(model_func, model_name, volumes, density, A_free0, t, **kwargs):
    all_results = []
    for col, V in enumerate(volumes):
        for A0 in A_free0:
            if model_func == effective_concentration_model:
                y0 = [A0, 0, V * density[col], 0, kwargs['S0']]
                sol = odeint(model_func, y0, t, args=(kwargs['mu_max'], kwargs['K_s'], kwargs['Y'], kwargs['k_bind'], kwargs['k_unbind'], kwargs['lambda_max'], V, kwargs['K_D'], kwargs['n']))
                data = pd.DataFrame(sol, columns=["A_free", "A_bound_live", "B_live", "B_dead", "S"])
            elif model_func == linear_Lysis_rate_model:
                y0 = [V * density[col], 0, kwargs['S0']]
                sol = odeint(model_func, y0, t, args=(kwargs['mu_max'], kwargs['K_s'], kwargs['Y'], kwargs['a'], kwargs['b'], V, A0, kwargs['K_A0'], kwargs['n']))
                data = pd.DataFrame(sol, columns=["B_live", "B_dead", "S"])
            elif model_func == combined_model:
                y0 = [A0, 0, V * density[col], 0, kwargs['S0']]
                sol = odeint(model_func, y0, t, args=(kwargs['mu_max'], kwargs['K_s'], kwargs['Y'], kwargs['k_bind'], kwargs['k_unbind'], V, kwargs['K_D'], kwargs['n'], kwargs['a'], kwargs['b']))
                data = pd.DataFrame(sol, columns=["A_free", "A_bound_live", "B_live", "B_dead", "S"])
            data["time"] = t
            data["A0"] = A0
            data["volume"] = V
            all_results.append(data)
    return pd.concat(all_results, ignore_index=True)

def int_to_subscript(n):
    subscript_digits = str.maketrans('0123456789-', '₀₁₂₃₄₅₆₇₈₉₋')
    return str(n).translate(subscript_digits)

def int_to_superscript(n):
    superscript_digits = str.maketrans('0123456789-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻')
    return str(n).translate(superscript_digits)

if __name__ == "__main__":
    volume = [1e3, 1e4, 1e5, 1e6, 1e7]
    density = [0.005742, 0.001325, 0.000619, 0.000499, 0.000402]
    t = np.linspace(0, 24, 250)
    A_free0 = [0, 3.3, 10, 30]

    model_funcs = [effective_concentration_model, linear_Lysis_rate_model, combined_model]
    model_names = ["Effective Concentration", "Linear Lysis Rate", "Combined Model"]


    plots = []
    for model_func, model_name in zip(model_funcs, model_names):
        results = run_model_for_volumes(
            model_func, model_name, volume, density, A_free0, t,
            mu_max=mu_max, K_s=Ks, Y=Y, k_bind=K_on, k_unbind=K_off,
            lambda_max=lambda_max, K_D=K_D, n=n, S0=S0, a=a, b=b, K_A0=K_A0
        )
        volume_list = sorted(results['volume'].unique())
        A0_list = sorted(results['A0'].unique())
        color_map = {V: c for V, c in zip(volume_list, Category10[10])}
        linestyles = ['solid', 'dashed', 'dotdash', 'dotted']
        linestyle_map = {A0: ls for A0, ls in zip(A0_list, itertools.cycle(linestyles))}

        p = figure(title=f'B_live over time - {model_name}', x_axis_label='Time', y_axis_label='B_live',
                   y_axis_type='log', width=1400, height=1000)
        legend_items = []
        for (V, A0), group in results.groupby(['volume', 'A0']):
            if 'B_live' in group.columns:
                exp = int(np.log10(V))
                legend_label = f"Droplet Size 10{int_to_superscript(exp)}, Conc {A0}"
                r = p.line(
                    group['time'], group['B_live'],
                    line_color=color_map[V],
                    line_dash=linestyle_map[A0],
                    line_width=2
                )
                legend_items.append((legend_label, [r]))
        legend = Legend(items=legend_items)
        p.add_layout(legend, 'right')
        legend.title = 'Volume/A0'
        legend.title_text_font_size = '8pt'
        legend.click_policy = 'hide'
        p.xgrid.visible = True
        p.ygrid.visible = True
        plots.append(p)

    for p in plots:
        streamlit_bokeh(p, use_container_width=True, theme="streamlit")




