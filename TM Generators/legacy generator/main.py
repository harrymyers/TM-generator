import numpy as np
import matplotlib.pyplot as plt

# Set parameters
np.random.seed(42)
E_beam = 3740.0  # MeV
M_TM = 211.0     # MeV #restmass of TM
x_min = M_TM / E_beam
N_events = 1000000
Lambda = 20.0  # MeV
tau_0 = 1.8e-12  # seconds (lifetime in rest frame)
c = 3e8  # m/s speed of light

# Generate energy fraction x ~ 1/x
U = np.random.rand(N_events)
x = x_min * np.exp(U * np.log(1.0 / x_min))
E_TM = x * E_beam
p_TM = np.sqrt(E_TM**2 - M_TM**2)

# Angular spread
theta0 = Lambda / (x * E_beam)
theta_prod = np.abs(np.random.normal(loc=0.0, scale=theta0))
phi_prod = np.random.uniform(0, 2*np.pi, N_events)

# TM momentum
p_TM_x = p_TM * np.sin(theta_prod) * np.cos(phi_prod)
p_TM_y = p_TM * np.sin(theta_prod) * np.sin(phi_prod)
p_TM_z = p_TM * np.cos(theta_prod)
TM_mom = np.vstack((p_TM_x, p_TM_y, p_TM_z)).T

# Decay in rest frame
p_star = M_TM / 2.0
cos_theta_decay = np.random.uniform(-1, 1, N_events)
theta_decay = np.arccos(cos_theta_decay)
phi_decay = np.random.uniform(0, 2*np.pi, N_events)
p_decay = np.vstack([
    p_star * np.sin(theta_decay) * np.cos(phi_decay),
    p_star * np.sin(theta_decay) * np.sin(phi_decay),
    p_star * cos_theta_decay
]).T
p_decay_opposite = -p_decay

# Boost setup
n_TM = TM_mom / p_TM[:, None]
gamma = E_TM / M_TM
beta = p_TM / E_TM
E_e = p_star
E_array = np.full(N_events, E_e)

def vectorized_boost(p_vec, E, beta, n, gamma):
    p_parallel = np.sum(p_vec * n, axis=1, keepdims=True) * n
    p_perp = p_vec - p_parallel
    E_boosted = gamma * (E + beta * np.sum(p_vec * n, axis=1))
    p_parallel_boosted = gamma[:, None] * (p_parallel + (beta * E)[:, None] * n)
    p_boosted = p_perp + p_parallel_boosted
    return E_boosted, p_boosted

E_eplus, p_eplus = vectorized_boost(p_decay, E_array, beta, n_TM, gamma)
E_eminus, p_eminus = vectorized_boost(p_decay_opposite, E_array, beta, n_TM, gamma)

def vertical_angle(p):
    p_x, p_y, p_z = p[:,0], p[:,1], p[:,2]
    return np.arctan2(np.abs(p_y), np.sqrt(p_x**2 + p_z**2))

theta_y_eplus_mrad = vertical_angle(p_eplus) * 1000
theta_y_eminus_mrad = vertical_angle(p_eminus) * 1000

# Acceptance
acceptance = ((theta_y_eplus_mrad >= 7.5) & (theta_y_eplus_mrad <= 30.0) &
              (theta_y_eminus_mrad >= 7.5) & (theta_y_eminus_mrad <= 30.0))
accepted_theta_y_eplus = theta_y_eplus_mrad[acceptance]
accepted_theta_y_eminus = theta_y_eminus_mrad[acceptance]

# Opening angle
unit_vec_eplus = p_eplus / np.linalg.norm(p_eplus, axis=1)[:, None]
unit_vec_eminus = p_eminus / np.linalg.norm(p_eminus, axis=1)[:, None]
cos_opening_angle = np.sum(unit_vec_eplus * unit_vec_eminus, axis=1)
opening_angle_mrad = np.arccos(cos_opening_angle) * 1000

# Decay displacement
decay_length_mean = beta * gamma * c * tau_0
decay_distances = np.random.exponential(decay_length_mean)
decay_vertices = decay_distances[:, None] * n_TM

# Detector hits
z_detector = 0.5
def project_to_z_plane(p_vec, start_pos, z_target):
    direction = p_vec / np.linalg.norm(p_vec, axis=1)[:, None]
    dz = z_target - start_pos[:, 2]
    t = dz / direction[:, 2]
    return start_pos + direction * t[:, None]

hit_positions_eplus = project_to_z_plane(p_eplus, decay_vertices, z_detector)
hit_positions_eminus = project_to_z_plane(p_eminus, decay_vertices, z_detector)


#confirming units correct
#print("Hit X range (raw units):", 
#      hit_positions_eminus[:,0].min(), 
#      hit_positions_eminus[:,0].max())
#print("Hit Y range (raw units):", 
#      hit_positions_eminus[:,1].min(), 
#      hit_positions_eminus[:,1].max())


# Decay length correlation plots
decay_lengths_mm = np.linalg.norm(decay_vertices, axis=1) * 1000





# Angular correlation
plt.figure(figsize=(7,6))
plt.hist2d(theta_y_eplus_mrad, theta_y_eminus_mrad, bins=100, range=[[0,200],[0,200]], cmap='viridis')
plt.colorbar(label='Counts')
plt.title("Angular Correlation between e⁺ and e⁻")
plt.xlabel("e⁺ vertical angle (mrad)")
plt.ylabel("e⁻ vertical angle (mrad)")
plt.grid(True)
plt.tight_layout()
plt.savefig("fig1_angular_correlation.png")
plt.close()

# Zoomed-in accepted events
plt.figure(figsize=(7,6))
plt.hist2d(accepted_theta_y_eplus, accepted_theta_y_eminus, bins=50, range=[[7.5,30],[7.5,30]], cmap='plasma')
plt.colorbar(label='Counts')
plt.title("Accepted Events: Angular Correlation (7.5–30 mrad)")
plt.xlabel("e⁺ vertical angle (mrad)")
plt.ylabel("e⁻ vertical angle (mrad)")
plt.grid(True)
plt.tight_layout()
plt.savefig("fig2_zoomed_acceptance.png")
plt.close()

# Momentum space
plt.figure(figsize=(8,6))
plt.hist2d(p_eplus[:,1], p_eplus[:,2], bins=100,range=[[-120,120],[0,800]], cmap='Blues')
plt.colorbar(label='Counts')
plt.xlabel("p_y (MeV/c)")
plt.ylabel("p_z (MeV/c)")
plt.title("e⁺ Momentum: Transverse vs Longitudinal")
plt.tight_layout()
plt.savefig("fig3_eplus_momentum_zoom.png")
plt.close()


plt.figure(figsize=(8,6))
plt.hist2d(p_eplus[:,1], p_eplus[:,2], bins=100, cmap='Blues')
plt.colorbar(label='Counts')
plt.xlabel("p_y (MeV/c)")
plt.ylabel("p_z (MeV/c)")
plt.title("e⁺ Momentum: Transverse vs Longitudinal")
plt.tight_layout()
plt.savefig("fig3_eplus_momentum.png")
plt.close()






plt.figure(figsize=(8,6)) 
plt.hist2d(p_eminus[:,1], p_eminus[:,2], bins=100,range=[[-120,120],[0,800]], cmap='Reds')
plt.colorbar(label='Counts')
plt.xlabel("p_y (MeV/c)")
plt.ylabel("p_z (MeV/c)")
plt.title("e⁻ Momentum: Transverse vs Longitudinal")
plt.tight_layout()
plt.savefig("fig4_eminus_momentum.png")
plt.close()

plt.figure(figsize=(8,6))
plt.hist2d(p_eminus[:,1], p_eminus[:,2], bins=100, cmap='Reds')
plt.colorbar(label='Counts')
plt.xlabel("p_y (MeV/c)")
plt.ylabel("p_z (MeV/c)")
plt.title("e⁻ Momentum: Transverse vs Longitudinal")
plt.tight_layout()
plt.savefig("fig4_eminus_zoom_momentum.png")
plt.close()

# Opening angle
plt.figure(figsize=(8,5))
plt.hist(opening_angle_mrad, bins=100, color='seagreen')
plt.title("Opening Angle Distribution (Lab Frame)")
plt.xlabel("Opening angle (mrad)")
plt.tight_layout()
plt.savefig("fig5_opening_angle_mrad.png")
plt.close()

plt.figure(figsize=(8,5))
plt.hist(opening_angle_mrad, bins=100, range=(100,500), color='darkorange')
plt.title("Opening Angle (100 - 500 mrad)")
plt.xlabel("Opening angle (mrad)")
plt.tight_layout()
plt.savefig("fig6_opening_angle_zoomed.png")
plt.close()

# Decay length
plt.figure(figsize=(8,5))
plt.hist(decay_lengths_mm, bins=100, color='slateblue', log=True)
plt.title("Decay Length Distribution")
plt.xlabel("Decay length (mm)")
plt.tight_layout()
plt.savefig("fig7_decay_length_distribution.png")
plt.close()

# Detector hits
plt.figure(figsize=(8,6))
plt.hist2d(hit_positions_eplus[:,0]*1000, hit_positions_eplus[:,1]*1000, bins=200, range=[[-100,100],[-100,100]], cmap='Blues')
plt.colorbar(label="Counts")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.title("e⁺ Detector Hit Map at z=0.5 m")
plt.tight_layout()
plt.savefig("fig8_detector_hits_eplus.png")
plt.close()

plt.figure(figsize=(8,6))
plt.hist2d(hit_positions_eminus[:,0]*1000, hit_positions_eminus[:,1]*1000, bins=200,range=[[-100,100],[-100,100]], cmap='Reds')
plt.colorbar(label="Counts")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.title("e⁻ Detector Hit Map at z=0.5 m")
plt.tight_layout()
plt.savefig("fig9_detector_hits_eminus.png")
plt.close()

# Correlations
plt.figure(figsize=(8,5))
plt.hist2d(E_TM, decay_lengths_mm, bins=200, range=[[210,230],[0, 0.1]],cmap='viridis')
plt.colorbar(label="Counts")
plt.xlabel("TM Energy (MeV)")
plt.ylabel("Decay length (mm)")
plt.title("Decay Length vs TM Energy")
plt.tight_layout()
plt.savefig("fig10_decay_vs_energy.png")
plt.close()

plt.figure(figsize=(8,5))
plt.hist2d(theta_y_eplus_mrad, decay_lengths_mm, bins=200,range=[[0,200],[0,4]] ,cmap='plasma')
plt.colorbar(label="Counts")
plt.xlabel("e⁺ vertical angle (mrad)")
plt.ylabel("Decay length (mm)")
plt.title("Decay Length vs e⁺ Vertical Angle")
plt.tight_layout()
plt.savefig("fig11_decay_vs_angle.png")
plt.close()



#LHE portion

# Generate LHE for 100 events
lhe_lines = []
lhe_lines.append("<LesHouchesEvents version=\"1.0\">")
lhe_lines.append("<header>")
lhe_lines.append("    Generated with true muonium decay + displacement")
lhe_lines.append("</header>")
lhe_lines.append("<init>")
lhe_lines.append("  2212 2212  0.0 0.0  0.0 0.0  1001 1002")
lhe_lines.append("</init>")

sample_size = 100
for i in range(sample_size):
    vx, vy, vz = decay_vertices[i]
    lhe_lines.append("<event>")
    lhe_lines.append(" 2 1 1.0 0.0 0.0 0.0  # Decay vertex: x={:.3e} y={:.3e} z={:.3e}".format(vx, vy, vz))
    px, py, pz = p_eplus[i]
    E = E_eplus[i]
    lhe_lines.append(f"-11 -1 0 0 0 0 {px:.6e} {py:.6e} {pz:.6e} {E:.6e} 0.0")
    px, py, pz = p_eminus[i]
    E = E_eminus[i]
    lhe_lines.append(f"11 -1 0 0 0 0 {px:.6e} {py:.6e} {pz:.6e} {E:.6e} 0.0")
    lhe_lines.append("</event>")
lhe_lines.append("</LesHouchesEvents>")

with open("true_muonium_events_with_displacement.lhe", "w") as f:
    f.write("\n".join(lhe_lines))