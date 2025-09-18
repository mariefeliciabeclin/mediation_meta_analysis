import numpy as np
import pandas as pd

def expit(x):
    return 1 / (1 + np.exp(-x))

SEED = 3
# Function to simulate one RCT population j
def simulate_rct(n, mu, s, lambda0, lambda1, lambda2,
                 beta0, beta1, beta2, beta3, sigma2, seed=SEED):
    rng = np.random.default_rng(seed)
    
    # Covariate
    C = rng.normal(loc=mu, scale=s, size=n)
    
    # Treatment assignment (half receive A=1)
    A = rng.binomial(1, 0.5, size=n)
    
    # Mediator
    pM = expit(lambda0 + lambda1 * C + lambda2 * A)
    M = rng.binomial(1, pM)
    
    # Outcome
    eps = rng.normal(loc=0, scale=np.sqrt(sigma2), size=n)
    Y = beta0 + beta1 * C + beta2 * M + beta3 * A + eps
    
    return pd.DataFrame({"C": C, "A": A, "M": M, "Y": Y})




