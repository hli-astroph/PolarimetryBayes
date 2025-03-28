import numpy as np

# Likelihood for a single data point
def likelihood_i(t_i, pdf_i, param, kw):
    if kw == 'linear':
        psi_pred = param[1] + param[0] * t_i
        while (psi_pred > 90.): psi_pred = psi_pred - 180.
        while (psi_pred < -90): psi_pred = psi_pred + 180.
        likelihood = max(pdf_i(psi_pred), 1e-10)
    elif kw == 'const':
        psi_pred = param
        likelihood = max(pdf_i(psi_pred), 1e-10)
    elif kw == 'double':
        psi_pred0, psi_pred1 = param[0], param[1]
        likelihood = max(max(pdf_i(psi_pred0), pdf_i(psi_pred1)), 1e-10)
    else:
    	likelihood = 0.
    return likelihood
    
# Total log-likelihood
def log_likelihood(param, time, pdf, kw):
	total_log_likelihood = 0.
	for i in range(len(time)):
		total_log_likelihood += np.log(likelihood_i(time[i], pdf[i], param, kw))
	return -total_log_likelihood  # Negative for minimization
    
# AIC and BIC calculation
def compute_aic_bic(logL, n_params, n_data):
    aic = 2 * n_params - 2 * logL
    bic = n_params * np.log(n_data) - 2 * logL
    return aic, bic

