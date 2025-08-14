import numpy as np
import matplotlib.pyplot as plt
import gzip
import math
from pathlib import Path

# -------------------
# Configuration
# -------------------
np.random.seed(42)

# New constants for tungsten target and HPS beam
A       = 183.84         # atomic mass units
Lambda  = 71.227         # MeV (nuclear form factor cutoff)
m_mu    = 105.658        # MeV (muon mass)
m_e     = 0.510999       # MeV (electron mass)
n       = 1              # principal quantum number
alpha   = 7.297e-3       # fine-structure constant
B       = 0.85           # nuclear form factor constant
E_beam  = 3740.0         # MeV (beam energy)
Z       = 74             # tungsten atomic number

# TM rest mass ~ 2 m_mu
M_TM   = 211.0        # TM rest mass ~ 2 m_mu

# Physics / modeling knobs
Lambda_ang = Lambda   # MeV, using the new Lambda value
tau_0 = 1.8e-12       # s, TM lifetime in ret frame (not used in LHE here)
c = 3.0e8             # m/s

# Event counts
N_events = 100_000  # total MC events to generate
N_LHE_WRITE = N_events  # how many to write into the LHE

# LHE output
WRITE_GZIP = True
OUTFILE = "tm_3gamma_noaccept.lhe.gz" if WRITE_GZIP else "tm_3gamma_noaccept.lhe"

# QA plots (no acceptance)
MAKE_QA_PLOTS = True

# -------------------
# 3γ dσ/dx (complete calculation)
# Includes nuclear form factors and atomic screening for tungsten target
# -------------------
def dsigma_dx(x):
    """
    Calculate the differential cross section dσ/dx for 3-photon production
    given x = E_true_muonium / E_beam.
    All constants are hard-coded for tungsten target and HPS beam energy.
    """
    x = np.asarray(x)
    
    # Handle array input
    result = np.zeros_like(x)
    valid_mask = (x > 0) & (x < 1)
    
    if not np.any(valid_mask):
        return result
    
    x_valid = x[valid_mask]
    
    # Prefactor
    prefactor = (1 / (4 * n**3)) \
                * (Z**2) * (alpha**7) / (m_mu**2) \
                * ((Z * alpha * Lambda / m_mu)**2) \
                * (4 * B / x_valid)

    # Shape factor
    shape = (1 - x_valid + x_valid**2 / 2)

    # Log term with atomic screening taken into account
    log_term = np.log( ((1 - x_valid) * (m_mu)**2) / (x_valid**2 * (m_e)**2) )

    # Full expression
    result[valid_mask] = prefactor * ( shape * log_term - (1 - x_valid) )
    
    return np.clip(result, 0.0, None)

# Keep the old function name for compatibility with existing code
def three_gamma_dsigma_dx_shape(x):
    return dsigma_dx(x)

# -------------------
# Sample x from the correct distribution (inverse-CDF via table)
# -------------------
x_min = M_TM / E_beam
x_max = 1.0 - 1e-6

grid_size = 200_000
x_grid = np.linspace(x_min, x_max, grid_size)
pdf_grid = three_gamma_dsigma_dx_shape(x_grid)
cdf_grid = np.cumsum(pdf_grid)
cdf_grid /= cdf_grid[-1]

U = np.random.rand(N_events)
x = np.interp(U, cdf_grid, x_grid)

# -------------------
# Kinematics: TM 4-momentum in lab
# -------------------
E_TM = x * E_beam
p_TM = np.sqrt(np.maximum(E_TM**2 - M_TM**2, 0.0))

# Production angles (simple model; not an acceptance)
theta0 = Lambda_ang / (x * E_beam)
theta_prod = np.abs(np.random.normal(loc=0.0, scale=theta0, size=N_events))
phi_prod = np.random.uniform(0.0, 2.0*np.pi, size=N_events)

px_TM = p_TM * np.sin(theta_prod) * np.cos(phi_prod)
py_TM = p_TM * np.sin(theta_prod) * np.sin(phi_prod)
pz_TM = p_TM * np.cos(theta_prod)
TM_mom = np.vstack((px_TM, py_TM, pz_TM)).T

# -------------------
# TM -> e+ e- in rest frame (isotropic)
# include me ≠ 0 in p*
# -------------------
E_e_rest = M_TM / 2.0
p_star = np.sqrt(max(E_e_rest**2 - m_e**2, 0.0))

cos_th = np.random.uniform(-1.0, 1.0, N_events)
sin_th = np.sqrt(1.0 - cos_th**2)
phi = np.random.uniform(0.0, 2.0*np.pi, N_events)

p_decay = np.vstack([
    p_star * sin_th * np.cos(phi),
    p_star * sin_th * np.sin(phi),
    p_star * cos_th
]).T
p_decay_opposite = -p_decay

# -------------------
# Boost decay products to lab frame
# -------------------
p_mag = np.maximum(np.linalg.norm(TM_mom, axis=1), 1e-12)
n_TM = TM_mom / p_mag[:, None]
gamma = E_TM / M_TM
beta  = np.where(E_TM > 0, p_TM / E_TM, 0.0)
E_e_star = E_e_rest  # energy of e± in TM rest frame

def boost(p_vec, E_rest, beta, n, gamma):
    # decompose along n
    proj = np.sum(p_vec * n, axis=1, keepdims=True)
    p_parallel = proj * n
    p_perp = p_vec - p_parallel
    E_boosted = gamma * (E_rest + beta * proj[:, 0])
    p_parallel_boosted = gamma[:, None] * (p_parallel + (beta * E_rest)[:, None] * n)
    p_boosted = p_perp + p_parallel_boosted
    return E_boosted, p_boosted

E_eplus,  p_eplus  = boost(p_decay,          E_e_star, beta, n_TM, gamma)
E_eminus, p_eminus = boost(p_decay_opposite, E_e_star, beta, n_TM, gamma)

# -------------------
# Calculate total cross section
# -------------------
def calculate_total_cross_section():
    """Calculate the total cross section by integrating dσ/dx"""
    x_fine = np.linspace(x_min, x_max, 10000)
    dsigma_dx_values = dsigma_dx(x_fine)
    total_sigma = np.trapz(dsigma_dx_values, x_fine)
    return total_sigma

# Calculate and print cross section information
total_sigma = calculate_total_cross_section()
print(f"Total 3γ → TM cross section: {total_sigma:.2e} MeV⁻²")
print(f"Total 3γ → TM cross section: {total_sigma * 1e-28:.2e} barns")
print(f"Target: Tungsten (Z={Z}, A={A})")
print(f"Beam energy: {E_beam} MeV")

# -------------------
# (Optional) QA plots — no cuts applied anywhere
# -------------------
if MAKE_QA_PLOTS:
    # x PDF vs theory
    plt.figure(figsize=(8,5))
    bins = np.linspace(x_min, x_max, 120)
    hist, edges = np.histogram(x, bins=bins, density=True)
    centers = 0.5*(edges[:-1] + edges[1:])
    theory = dsigma_dx(centers)
    theory /= np.trapezoid(theory, centers)
    plt.step(centers, hist, where='mid', label='Sampled x (density)')
    xf = np.linspace(x_min, x_max, 2000)
    tf = dsigma_dx(xf); tf /= np.trapezoid(tf, xf)
    plt.plot(xf, tf, label='Theory dσ/dx (norm.)')
    plt.xlabel('x = E_TM / E_beam'); plt.ylabel('PDF (arb.)')
    plt.title('3γ production: x distribution (W target, no cuts)')
    plt.legend(); plt.tight_layout(); plt.savefig('qa_x_pdf.png'); plt.close()

    # x CDF check
    plt.figure(figsize=(8,5))
    plt.plot(x_grid, cdf_grid, label='Theory CDF')
    xs = np.sort(x); ys = np.linspace(0.0, 1.0, xs.size, endpoint=False)
    plt.plot(xs, ys, alpha=0.7, label='Empirical CDF')
    plt.xlabel('x'); plt.ylabel('CDF'); plt.legend()
    plt.title('x CDF (samples vs theory)'); plt.tight_layout()
    plt.savefig('qa_x_cdf.png'); plt.close()

    # e± lab polar angle (vertical-angle proxy) — purely for inspection
    def vert_angle(p):
        px, py, pz = p[:,0], p[:,1], p[:,2]
        return np.arctan2(np.abs(py), np.sqrt(px**2 + pz**2))
    th_ep = vert_angle(p_eplus) * 1e3
    th_em = vert_angle(p_eminus) * 1e3

    plt.figure(figsize=(8,5))
    plt.hist(th_ep, bins=150)
    plt.xlabel('e⁺ vertical angle (mrad)'); plt.ylabel('Events / bin')
    plt.title('e⁺ vertical angle (no acceptance)'); plt.tight_layout()
    plt.savefig('qa_theta_ep_mrad.png'); plt.close()

    plt.figure(figsize=(8,5))
    plt.hist(th_em, bins=150)
    plt.xlabel('e⁻ vertical angle (mrad)'); plt.ylabel('Events / bin')
    plt.title('e⁻ vertical angle (no acceptance)'); plt.tight_layout()
    plt.savefig('qa_theta_em_mrad.png'); plt.close()

# -------------------
# LHE writing (final-state e+ e- only, GeV units)
# -------------------
MeV_to_GeV = 1e-3
pxp, pyp, pzp = p_eplus[:N_LHE_WRITE].T * MeV_to_GeV
pxm, pym, pzm = p_eminus[:N_LHE_WRITE].T * MeV_to_GeV
Ep  = E_eplus[:N_LHE_WRITE]  * MeV_to_GeV
Em  = E_eminus[:N_LHE_WRITE] * MeV_to_GeV
me_GeV = m_e * MeV_to_GeV

open_fn = gzip.open if WRITE_GZIP else open
mode = 'wt' if WRITE_GZIP else 'w'
encoding = 'utf-8'

with open_fn(OUTFILE, mode, encoding=encoding) as f:
    f.write("<LesHouchesEvents version=\"1.0\">\n")
    f.write("<header>\n")
    f.write("  TM -> e+ e- (3γ production on W target), no acceptance/geometry\n")
    f.write("  Units: GeV. Status=1 for final-state particles.\n")
    f.write(f"  Target: Tungsten (Z={Z}, A={A})\n")
    f.write(f"  Beam energy: {E_beam} MeV\n")
    f.write(f"  Total cross section: {total_sigma:.2e} MeV⁻²\n")
    f.write("</header>\n")
    # Keep a minimal <init>. Many readers ignore this, but we mirror your previous pattern.
    f.write("<init>\n")
    f.write("  2212 2212  0.0 0.0  0.0 0.0  1001 1002\n")
    f.write("</init>\n")

    for i in range(N_LHE_WRITE):
        f.write("<event>\n")
        # NUP=2 particles, IDPRUP=1, XWGTUP=1.0, SCALUP=AQEDUP=AQCDUP=0.0
        f.write("  2 1 1.0 0.0 0.0 0.0\n")
        # format: IDUP ISTUP MOTH1 MOTH2 ICOL1 ICOL2 px py pz E m
        # e+ (PDG -11), e- (PDG 11); ISTUP=1 (final state).
        f.write(f"  -11 1 0 0 0 0 {pxp[i]:.6e} {pyp[i]:.6e} {pzp[i]:.6e} {Ep[i]:.6e} {me_GeV:.6e}\n")
        f.write(f"   11 1 0 0 0 0 {pxm[i]:.6e} {pym[i]:.6e} {pzm[i]:.6e} {Em[i]:.6e} {me_GeV:.6e}\n")
        f.write("</event>\n")

    f.write("</LesHouchesEvents>\n")

print(f"Wrote {N_LHE_WRITE:,} events to {OUTFILE}")