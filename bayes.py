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

def polarization_prior_2d(p0, psi0):
    """
    Uniform prior: 1 if 0 <= p0 <= 1 and -90 <= psi0 <= 90, else 0.
    """
    prob = np.zeros_like(p0)
    q = (p0 >=0.0) & (p0 <= 1.0) & (psi0 >= -90.0) & (psi0 <= 90.)
    prob[q] = 1.0
    return prob

def polarization_posterior_2d(mu, S, B, pf, pa, d_pf0=0.001, d_pa0=0.01):
    """
    Compute the posterior distribution for p0 and psi0.
    """
    pf_grid = np.arange(0.0, 1.0, d_pf0) + d_pf0 / 2.0
    pa_grid = np.arange(-90.0, 90.0, d_pa0) + d_pa0 / 2.0
    pf0, pa0 = np.meshgrid(pf_grid, pa_grid)
    
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
    
def posterior_ci_min_pf(x, pdf, CL):
    return posterior_ci_min(x, pdf, CL)

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

def plot_posterior_2d(po, x, y, CL=[0.68, 0.90, 0.99], cmap='magma', savefig=None):
    """
    Plot the 2D posterior contours at CLs.
    """
    n = 100
    cut_value = np.linspace(po.max(), 0., n)
    int_prob = np.zeros(n)
    d_area = (x[1]-x[0]) * (y[1]-y[0])
    for k in range(n):
	    int_prob[k] = po[po >= cut_value[k]].sum()
	    levels = np.interp(CL, int_prob * d_area, cut_value)
    plt.figure(figsize=(5, 4))
    plt.contour(x, y, po, levels[::-1], colors='r')
    plt.xlim(0, 0.2)
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
    S, B, mu = 200000, 0, 0.3  #Source, background counts, modulation factor

    posterior_2d, pf0, pa0 = polarization_posterior_2d(mu, S, B, pf_measured, pa_measured)
    pdf_pf, pdf_pa = polarization_posterior_1d(posterior_2d, pf0, pa0)

    plot_posterior_2d(posterior_2d, pf0, pa0)
    plot_posterior_1d(pdf_pf, pdf_pa, pf0, pa0)
    
if __name__ == "__main__":
    main()