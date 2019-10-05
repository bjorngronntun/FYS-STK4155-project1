import numpy as np

def FrankeFunction(x, y, noise = 0.0, seed = 8):
    """
    Compute value of franke function

    Parameters
    ----------
    x, y:   {float}
            Input data
    noise:  {float}
            Standard deviation of randomly generated
            noise (mean=0) that will be added to the
            output
    seed:   {int}
            Random seed used to generate noise

    Returns
    ---------
    Value of Franke function for given input x, y
    """
    term1 = 0.75*np.exp(-(0.25*(9*x - 2)**2) - 0.25*((9*y - 2)**2))
    term2 = 0.75*np.exp(-((9*x + 1)**2)/49.0 - 0.1*(9*y + 1))
    term3 = 0.5*np.exp(-0.25*((9*x - 7)**2) - 0.25*((9*y - 3)**2))
    term4 = -0.2*np.exp(-((9*x - 4)**2) - (9*y - y)**2)
    np.random.seed(seed)
    noise_term = np.random.normal(size=term1.shape, loc=0.0, scale=noise)

    return term1 + term2 + term3 + term4 + noise_term
