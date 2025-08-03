import numpy as np
from sklearn.cluster import KMeans

def initialize_parameters(X, n_components, seed=None):
    kmeans = KMeans(n_clusters=n_components, n_init=10, random_state=0).fit(X)
    mu = kmeans.cluster_centers_
    sigma = [np.cov(X[kmeans.labels_ == k].T) + 1e-6 * np.eye(X.shape[1]) for k in range(n_components)]
    pi = np.array([np.mean(kmeans.labels_ == k) for k in range(n_components)])
    return mu, sigma, pi

def multivariate_gaussian(X, mu, sigma):
    n = X.shape[1]
    det = np.linalg.det(sigma)
    if det <= 0:
        det = 1e-6
    inv = np.linalg.inv(sigma + 1e-6 * np.eye(n))
    norm_const = 1.0 / np.sqrt((2 * np.pi) ** n * det)
    diff = X - mu
    exponent = -0.5 * np.sum(diff @ inv * diff, axis=1)
    exponent = np.clip(exponent, -1e2, 1e2)
    return norm_const * np.exp(exponent)

def e_step(X, mu, sigma, pi):
    n_components = len(pi)
    gamma = np.zeros((X.shape[0], n_components))
    for k in range(n_components):
        gamma[:, k] = pi[k] * multivariate_gaussian(X, mu[k], sigma[k])
    gamma_sum = np.sum(gamma, axis=1, keepdims=True) + 1e-8
    gamma /= gamma_sum
    return gamma

def m_step(X, gamma):
    n_samples, n_features = X.shape  # x.shape = (n,d)
    n_components = gamma.shape[1]  # gamma.shape = (n,k)
    N_k = np.sum(gamma, axis=0)  # sum of gamma's column = (k,)
    mu = (gamma.T @ X) / N_k[:, np.newaxis]  # np.newaxis = (k,) -> (k,1)
    sigma = []
    for k in range(n_components):
        diff = X - mu[k]
        weighted_sum = (gamma[:, k][:, np.newaxis] * diff).T @ diff
        sigma.append(weighted_sum/N_k[k])
    pi = N_k / n_samples
    return mu, sigma, pi

def compute_log_likelihood(X, mu, sigma, pi):
    total = np.zeros(X.shape[0])
    for k in range(len(pi)):
        total += pi[k] * multivariate_gaussian(X, mu[k], sigma[k])
    return np.sum(np.log(total + 1e-8))

def fit_gmm_em(X, n_components=2, max_iter=100, tol=1e-4, seed=None, verbose=True):
    mu, sigma, pi = initialize_parameters(X, n_components, seed)
    log_likelihoods = []

    for i in range(max_iter):
        gamma = e_step(X, mu, sigma, pi)
        mu, sigma, pi = m_step(X, gamma)
        log_likelihood = compute_log_likelihood(X, mu, sigma, pi)
        log_likelihoods.append(log_likelihood)

        if verbose:
            print(f"Iteration {i+1}, Log-Likelihood: {log_likelihood:.4f}")

        if i > 0 and np.abs(log_likelihood - log_likelihood[-2]) < tol:
            break
    
    return mu, sigma, pi, gamma, log_likelihoods