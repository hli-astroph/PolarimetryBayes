import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
    
def polarization_likelihood_2d(mu, S, B, p_0, psi_0, p_r, psi_r):
    """
    Compute the 2D likelihood for polarization parameters.
    """
    sigma2 = 2. * (S + B) / (mu * S) ** 2
    item = p_r ** 2 + p_0 ** 2 - 2. * p_r * p_0 * np.cos(2. * np.deg2rad(psi_r - psi_0)) 
    f = p_r / np.pi / sigma2 * np.exp(-item / 2. / sigma2)
    return f
    
def polarization_likelihood_2d_correlated(mu, S, B, p_0, psi_0, p_r, psi_r):
    # number of source and background photons
    N = S + B
    psi_0 = np.deg2rad(psi_0)
    psi_r = np.deg2rad(psi_r)
    
    mp0 = mu * p_0
    # stokes Q & U
    Q0 = mp0 / 2.0 * np.cos(2. * psi_0)
    U0 = mp0 / 2.0 * np.sin(2. * psi_0)
    # sigma of Q & U
    sigma_Q = np.sqrt(1. / S * (N / 2. / S - Q0 ** 2))
    sigma_U = np.sqrt(1. / S * (N / 2. / S - U0 ** 2))

    mp02 = mp0 ** 2
    mp04 = mp0 ** 4
    s4p = np.sin(4. * psi_0)
    rho = -S * mp02 * s4p / np.sqrt(16. * N * N - 8. * N * S * mp02 + S * S * mp04 * s4p ** 2)

    Q_r = mu * p_r / 2.0 * np.cos(2. * psi_r)
    U_r = mu * p_r / 2.0 * np.sin(2. * psi_r)

    # likelihood of measured Q & U given true Q & U
    item1 = 2. * np.pi * sigma_Q * sigma_U * np.sqrt(1. - rho ** 2)
    item2 = (Q_r - Q0) ** 2 / sigma_Q ** 2
    item3 = (U_r - U0) ** 2 / sigma_U ** 2
    item4 = 2. * rho * (Q_r - Q0) * (U_r - U0) / sigma_Q / sigma_U
    BQU = 1. / item1 * np.exp(-1. / 2. / (1. - rho ** 2) * (item2 + item3 - item4))

    detJ = p_r * mu ** 2 / 2.0

    # likelihood in polar coordinates L(p_r, psi_r | p0, psi0)
    return BQU * detJ

def polarization_prior_2d(p0, psi0):
    """
    Uniform prior: 1 if 0 <= p0 <= 1 and -90 <= psi0 <= 90, else 0.
    """
    prob = np.zeros_like(p0)
    q = (p0 >=0.0) & (p0 <= 1.0) & (psi0 >= -90.0) & (psi0 <= 90.)
    prob[q] = 1.0
    return prob

def polarization_posterior_2d(mu, S, B, pf, pa, d_pf0=0.001, d_pa0=0.01, kw=''):
    """
    Compute the posterior distribution for p0 and psi0.
    """
    pf_grid = np.arange(0.0, 1.0, d_pf0) + d_pf0 / 2.0
    pa_grid = np.arange(-90.0, 90.0, d_pa0) + d_pa0 / 2.0
    pf0, pa0 = np.meshgrid(pf_grid, pa_grid)
    
    if kw[:3] == 'cor':
        likelihood = polarization_likelihood_2d_correlated(mu, S, B, pf0, pa0, pf, pa)
    else:
        likelihood = polarization_likelihood_2d(mu, S, B, pf0, pa0, pf, pa)
    prior = polarization_prior_2d(pf0, pa0)
    posterior = prior * likelihood
    norm = np.sum(posterior) * d_pf0 * d_pa0
    return posterior / norm, pf_grid, pa_grid
    

def polarization_posterior_1d(posterior_2d, pf_grid, pa_grid):
    """
    Compute univariate distributions for pf0 and pa0.
    """
    pdf_pf0 = np.sum(posterior_2d, axis=0) * (pa_grid[1] - pa_grid[0])
    pdf_pa0 = np.sum(posterior_2d, axis=1) * (pf_grid[1] - pf_grid[0])
    return pdf_pf0, pdf_pa0

def posterior_ci_min(x, pdf, CL):
    """
    Compute the MAP and minimum-width credible interval.
    """
    x_map = x[np.argmax(pdf)]
    if x_map.size > 1: x_map = np.mean(x_map)
    
    y_cut = np.linspace(pdf.max(), pdf.min(), 200)
    area = np.array([np.sum(pdf[pdf >= yc]) * (x[1]-x[0]) for yc in y_cut])
    y_cut0 = np.interp(CL, area, y_cut)
    bounds = x[pdf >= y_cut0]
    return x_map, [bounds[0], bounds[-1]]

def posterior_ci_min_pa(x, pdf, CL):
    x_map = x[np.argmax(pdf)]
    if x_map.size > 1: x_map = np.mean(x_map)
    if x_map > 0:
        ext_x = np.concatenate((x, x + 180.))
    else:
        ext_x = np.concatenate((x - 180., x))
    ext_pdf = np.concatenate((pdf, pdf))
    q = np.abs(ext_x - x_map) <= 90.
    x = ext_x[q]
    pdf = ext_pdf[q]
    return posterior_ci_min(x, pdf, CL)

def plot_posterior_2d(posterior, pf_lim=(0, 1), cmap='magma', savefig=None):
    """
    Plot the 2D posterior distribution.
    """
    plt.figure(figsize=(5, 4))
    plt.imshow(posterior, extent=[0, 1, -90, 90], cmap=cmap, origin='lower', aspect='auto')
    plt.colorbar(label='Posterior Probability')
    plt.xlim(*pf_lim)
    plt.ylim(-90, 90)
    plt.xlabel('$p_0$')
    plt.ylabel('$\Psi_0$ (°)')
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
    plt.show()
    
def plot_posterior_1d(pdf_pf, pdf_pa, pf0, pa0, savefig=None):
    """
    Plot the marginal posterior distributions for p0 and psi0.
    """
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    
    axs[0].plot(pf0, pdf_pf, color='purple')
    axs[0].set_xlim(0, 1)
    axs[0].set_xlabel('$p_0$')
    axs[0].set_ylabel('PDF')

    axs[1].plot(pa0, pdf_pa, color='purple')
    axs[1].set_xlim(-90, 90)
    axs[1].set_xlabel('$\Psi_0$ (°)')
    axs[1].set_ylabel('PDF')

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
    plt.show()


def main():
    # Example parameters
    pf_measured, pa_measured = 0.05, 60.0  # Measured polarization degree and angle
    S, B, mu = 100000, 0, 0.3  #Source, background counts, modulation factor

    posterior_2d, pf0, pa0 = polarization_posterior_2d(mu, S, B, pf_measured, pa_measured)
    pdf_pf, pdf_pa = polarization_posterior_1d(posterior_2d, pf0, pa0)

    plot_posterior_2d(posterior_2d, pf_lim=(0, 0.2))
    plot_posterior_1d(pdf_pf, pdf_pa, pf0, pa0)
    
if __name__ == "__main__":
    main()
