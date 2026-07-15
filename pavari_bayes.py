import numpy as np
import dynesty
import sys
from scipy.special import logsumexp

def likelihood_array(p0, psi0, q, u, N, mu):
	q0 = 0.5 * p0 * np.cos(2*psi0) * mu
	u0 = 0.5 * p0 * np.sin(2*psi0) * mu
	sig_q = np.sqrt((0.5 - (0.5 * mu * p0 * np.cos(2.*psi0))**2)/N)
	sig_u = np.sqrt((0.5 - (0.5 * mu * p0 * np.sin(2.*psi0))**2)/N)
	
	rho_term1 = mu**2 * p0**2 * np.sin(4.*psi0)
	rho_term2 = 16. - 8.*(mu*p0)**2 + ((mu*p0)**2 * np.sin(4.*psi0))**2
	rho = -rho_term1 / np.sqrt(rho_term2)
	rho = np.clip(rho, -0.999999999999, 0.999999999999)
	
	term1 = (q0-q)**2/sig_q**2 + (u0-u)**2/sig_u**2
	term2 = -2. * rho * (q0-q) * (u0-u)/sig_q/sig_u
	log_norm = -np.log(2*np.pi)-np.log(sig_q)-np.log(sig_u)-0.5*np.log(1-rho**2)
	logL = log_norm - (term1 + term2)/(2*(1-rho**2))
	return logL
	
def log_likelihood(param, q, u, N, mu, time=None, kw=''):
	if kw.startswith('const'):
		p0 = param[0]
		psi0 = np.deg2rad(param[1])
		logL = np.sum(likelihood_array(p0, psi0, q, u, N, mu))
	elif kw.startswith('lin'):
		p0 = param[0]
		psi0 = np.deg2rad(param[1] + param[2] * time)
		psi0 = ((psi0 + np.pi/2) % np.pi) - np.pi/2
		logL = np.sum(likelihood_array(p0, psi0, q, u, N, mu))
	elif kw.startswith('bimodal'):
		p0 = param[0]
		psi0, psi1 = np.deg2rad(param[1]), np.deg2rad(param[2])
		logL0 = likelihood_array(p0, psi0, q, u, N, mu)
		logL1 = likelihood_array(p0, psi1, q, u, N, mu)
		frac0 = param[3]
		log_terms = np.array([np.log(frac0) + logL0, np.log(1.-frac0) + logL1])
		logL = np.sum(logsumexp(log_terms, axis=0))
	else:
		print("Unknown PA vs. Time model !!")
		sys.exit()
	return logL

def prior_trans_const(unit_cube):
	x = np.zeros_like(unit_cube)
	x[0] = unit_cube[0] * 0.3
	x[1] = -90. + 180. * unit_cube[1]
	return x
	
def prior_trans_bimodal(unit_cube):
	x = np.zeros_like(unit_cube)
	x[0] = unit_cube[0] * 0.3
	frac = 1e-8 + (1 - 2e-8) * unit_cube[3]
	psi0 = -90. + 180. * unit_cube[1]
	psi1 = -90. + 180. * unit_cube[2]
	if psi0 > psi1:
		psi0, psi1 = psi1, psi0
		frac = 1.0 - frac
	x[1], x[2], x[3] = psi0, psi1, frac
	return x
	
def prior_trans_linear(unit_cube):
	x = np.zeros_like(unit_cube)
	x[0] = unit_cube[0] * 0.3
	x[1] = -90. + 180. * unit_cube[1]
	x[2] = -20. + 40. * unit_cube[2]
	return x

def run_dynesty(q, u, N, mu, time=None, kw=''):
	if kw.startswith('const'):
		likelihood = lambda param: log_likelihood(param, q, u, N, mu, kw=kw)
		prior_trans = prior_trans_const
		ndim = 2
	elif kw.startswith('bimodal'):
		likelihood = lambda param: log_likelihood(param, q, u, N, mu, kw=kw)
		prior_trans = prior_trans_bimodal
		ndim = 4
	elif kw.startswith('lin'):
		likelihood = lambda param: log_likelihood(param, q, u, N, mu, time, kw=kw)
		prior_trans = prior_trans_linear
		ndim = 3	
	else:
		print("### Unkown model!!!")
		sys.exit()
	sampler = dynesty.NestedSampler(
		likelihood,
		prior_trans,
		ndim=ndim,
		nlive=5000,
		bound="multi",
		sample="rwalk")
	sampler.run_nested(dlogz=0.01, print_progress=True)
	return sampler.results

