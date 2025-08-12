import numpy as np


def are_variables_jointly_normally_distributed(vector_a: list[float], vector_b: list[float]) -> bool:
	return False


def pearson_correlation(vector_a: list[float], vector_b: list[float]) -> float:
    a = np.array(vector_a)
    b = np.array(vector_b)
    
    if len(a) != len(b): raise ValueError("Vectors must have the same length")

    mean_a = np.mean(a)
    mean_b = np.mean(b)

    numerator = np.sum((a - mean_a) * (b - mean_b))
    denominator = np.sqrt(np.sum((a - mean_a) ** 2) * np.sum((b - mean_b) ** 2))

    return numerator / denominator
