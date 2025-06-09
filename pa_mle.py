import numpy as np

# Likelihood for a single data point
def likelihood_i(param, pdf_i, kw, t_i):
    if kw == 'linear':
        psi_pred = param[1] + param[0] * t_i
        while (psi_pred > 90.): psi_pred = psi_pred - 180.
        while (psi_pred < -90): psi_pred = psi_pred + 180.
        likelihood = max(pdf_i(psi_pred), 1e-10)
    elif kw == 'const':
        psi_pred = param[0]
        likelihood = max(pdf_i(psi_pred), 1e-10)
    elif kw == 'bimodal':
        psi_pred0, psi_pred1, w = param[0], param[1], param[2]
        #likelihood = max(max(pdf_i(psi_pred0), pdf_i(psi_pred1)), 1e-10)
        likelihood = pdf_i(psi_pred0) * w + pdf_i(psi_pred1) * (1-w)
        likelihood = max(likelihood, 1e-10)
    else:
    	likelihood = 0.
    return likelihood
    
# Total log-likelihood
def log_likelihood(param, pdf, kw, time=None):
    if time is None:
        time = np.zeros(len(pdf))
    total_log_likelihood = 0.
    for i in range(len(pdf)):
        total_log_likelihood += np.log(likelihood_i(param, pdf[i], kw, time[i]))
    return total_log_likelihood  # Negative for minimization

def prior_trans_const(u):
	x = np.zeros_like(u)
	x[0] = -90 + 180 * u[0]
	return x
	
def prior_trans_bimodal(u):
	x = np.zeros_like(u)
	x[0] = -90 + 180 * u[0]
	x[1] = -90 + 180 * u[1]
	x[2] = u[2]
	return x
	
def prior_trans_linear(u):
	x = np.zeros_like(u)
	x[1] = -90 + 180 * u[0]
	x[0] = -5 + 30 * u[1]
	return x
    
# AIC and BIC calculation
def compute_aic_bic(logL, n_params, n_data):
    aic = 2 * n_params - 2 * logL
    bic = n_params * np.log(n_data) - 2 * logL
    return aic, bic

