import numpy as np
import scipy.stats as stats
import math
import pandas as pd

def calculate_normal_density_distribution(mu, sigma, n_points=100):
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, n_points)
    return pd.DataFrame({'x': x, 'y': stats.norm.pdf(x, loc=mu, scale=sigma)})
    
def sample_from_normal_distribution(mu, sigma, n_samples=100):
    return pd.DataFrame({'sample': np.random.normal(loc=mu, scale=sigma, size=n_samples)})