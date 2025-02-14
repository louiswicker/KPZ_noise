import numpy as np

def calculate_correlation_length(x, threshold=0.37):
    """
    Calculates the correlation length scale of a 1D array.

    Args:
        x (np.ndarray): The input 1D array.
        threshold (float, optional): The threshold for the normalized ACF to drop below. Defaults to 0.37 (1/e).

    Returns:
        int: The correlation length scale.
    """
    acf = np.correlate(x, x, mode='full')
    acf = acf[acf.size // 2:]
    norm_acf = acf / acf[0]
    for i, value in enumerate(norm_acf):
        if value < threshold:
            return i
    return len(norm_acf)

def Pearson_correlation(X,Y):
    if len(X)==len(Y):
        Sum_xy = sum((X-X.mean())*(Y-Y.mean()))
        Sum_x_squared = sum((X-X.mean())**2)
        Sum_y_squared = sum((Y-Y.mean())**2)  
        corr = Sum_xy / np.sqrt(Sum_x_squared * Sum_y_squared)
    return corr