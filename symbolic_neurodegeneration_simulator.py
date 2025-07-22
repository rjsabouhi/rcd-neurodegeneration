
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ------------------------------
# Symbolic Neurodegeneration Equations
# ------------------------------

def compute_coherence(theta, entropy_grad):
    return 1 / (1 + theta * entropy_grad)

def compute_kernel_integrity(gamma, mu):
    return gamma / (1 + mu)

def compute_fate(kernel):
    if kernel > 0.6:
        return "coherent"
    elif kernel > 0.3:
        return "fragmented"
    else:
        return "dissolved"

# ------------------------------
# Simulate Identity Over Time
# ------------------------------

def simulate_neurodegeneration(theta, mu, entropy_grad, steps=20):
    times = np.arange(steps)
    gammas = []
    kernels = []
    for t in times:
        # Entropy increases slightly over time
        local_entropy = entropy_grad + 0.05 * t
        gamma = compute_coherence(theta, local_entropy)
        kernel = compute_kernel_integrity(gamma, mu)
        gammas.append(gamma)
        kernels.append(kernel)
    return times, gammas, kernels

# ------------------------------
# Streamlit App
# ------------------------------

st.set_page_config(page_title="Symbolic Neurodegeneration Simulator", layout="wide")
st.title("Symbolic Neurodegeneration Simulator")
st.markdown("Model the collapse of symbolic identity coherence over time, inspired by Alzheimer's and related disorders.")

col1, col2 = st.columns(2)
with col1:
    theta = st.slider("Θ — Clinging (identity rigidity)", 0.0, 2.0, 1.0, 0.01)
    mu = st.slider("μ — Memory Tension", 0.0, 5.0, 1.0, 0.01)
    entropy_grad = st.slider("∇S — Entropy Gradient", 0.1, 10.0, 1.0, 0.1)

# Run simulation
t, gammas, kernels = simulate_neurodegeneration(theta, mu, entropy_grad)
final_kernel = kernels[-1]
fate = compute_fate(final_kernel)

with col2:
    st.metric("Final Coherence γ", f"{gammas[-1]:.4f}")
    st.metric("Core Identity κ", f"{final_kernel:.4f}")
    st.metric("System Fate ψ", fate.upper())
    if fate == "coherent":
        st.success("Symbolic identity is holding together.")
    elif fate == "fragmented":
        st.warning("Symbolic identity is partially fractured.")
    else:
        st.error("Symbolic identity has dissolved.")

# Plot
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t, gammas, label="Coherence γ(t)")
ax.plot(t, kernels, label="Identity Kernel κ(t)", linestyle='--')
ax.set_xlabel("Time")
ax.set_ylabel("Value")
ax.set_title("Symbolic Identity Collapse Over Time")
ax.legend()
st.pyplot(fig)

# Footer
st.markdown("---")
st.caption("This simulation uses Recursive Cognitive Dynamics to model symbolic attractor decay under entropic drift and memory tension. Useful for exploring cognitive breakdown patterns in neurodegenerative contexts.")
