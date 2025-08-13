from math import sqrt

import numpy as np
import pingouin as pg
import scipy.stats as stats


def are_variables_jointly_normally_distributed(vector_a: list[float], vector_b: list[float]) -> bool:
	return pg.multivariate_normality([(a, b) for a, b in zip(vector_a, vector_b)])[0]


def pearson_correlation(vector_a: list[float], vector_b: list[float]) -> float:
	"""
	Calculates the Pearson Correlation Coefficient between two n-dimensional
	vectors. The given vectors must be the same size.
	"""

	a = np.array(vector_a)
	b = np.array(vector_b)
	
	if len(a) != len(b): raise ValueError("Vectors must have the same length")

	mean_a = np.mean(a)
	mean_b = np.mean(b)

	numerator = np.sum((a - mean_a) * (b - mean_b))
	denominator = np.sqrt(np.sum((a - mean_a) ** 2) * np.sum((b - mean_b) ** 2))

	return numerator / denominator


def heaviside_step_function(x: float) -> float:
	return 0 if x < 0 else 1


def t_statistic(r: float, n: float) -> float:
	return r * sqrt((n - 2) / (1 - r**2))


def two_tailed_p_value(t_statistic: float, n: float) -> float:
	return 2 * (1 - float(stats.t.cdf(abs(t_statistic), df = n - 2)))
