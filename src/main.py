from data import CSV
import os
import numpy as np
from itertools import product

from stats import heaviside_step_function, pearson_correlation, t_statistic, two_tailed_p_value


def main():
	demographics = CSV("data/1000BRAINS/sourceFiles/demographics.csv", where = lambda entry: entry["ID"].endswith("1"))
	subject_count = len(demographics)
	processing_speed: list[float] = demographics["Processing_Speed_raw"]
	reasoning: list[float] = demographics["Reasoning_raw"]

	structural_connectomes = []
	connectomes_folder = "data/1000BRAINS/connectomes/structuralConnectomes"
	for filename in filter(lambda file: file.endswith("1.csv"), os.listdir(connectomes_folder)):
		connectome = np.loadtxt(f"{connectomes_folder}/{filename}", delimiter=",")
		structural_connectomes.append(connectome)

	edges = np.empty((100, 100, subject_count))
	for edge_x, edge_y in product(range(100), range(100)):
		edges[edge_x][edge_y] = list(map(lambda matrix: matrix[edge_x][edge_y], structural_connectomes))

	processing_matrices = matrices(edges, processing_speed, subject_count)
	reasoning_matrices = matrices(edges, reasoning, subject_count)


def matrices(edges, performance, subject_count):
	correlation = correlation_matrix(edges, performance)
	t_values = t_matrix(correlation, subject_count)
	p_values = p_matrix(t_values, subject_count)
	mask = mask_matrix(p_values)
	return correlation, t_values, p_values, mask


def correlation_matrix(edges: np.ndarray[tuple[int, int, int]], performance: list[float]):
	correlation = np.empty((edges.shape[0], edges.shape[1]))
	for edge_x, edge_y in product(range(correlation.shape[0]), range(correlation.shape[1])):
		correlation[edge_x][edge_y] = pearson_correlation(edges[edge_x][edge_y], performance)
	return correlation


def t_matrix(correlation: np.ndarray[tuple[int, int]], subject_count):
	ts = np.empty(correlation.shape)
	for edge_x, edge_y in product(range(ts.shape[0]), range(ts.shape[1])):
		ts[edge_x][edge_y] = t_statistic(correlation[edge_x][edge_y], subject_count)
	return ts


def p_matrix(t_matrix: np.ndarray[tuple[int, int]], subject_count):
	ps = np.empty(t_matrix.shape)
	for edge_x, edge_y in product(range(ps.shape[0]), range(ps.shape[1])):
		ps[edge_x][edge_y] = two_tailed_p_value(t_matrix[edge_x][edge_y], subject_count)
	return ps


def mask_matrix(p_matrix: np.ndarray[tuple[int, int]], alpha = 0.05):
	mask = np.empty(p_matrix.shape)
	for edge_x, edge_y in product(range(mask.shape[0]), range(mask.shape[1])):
		mask[edge_x][edge_y] = heaviside_step_function(alpha - p_matrix[edge_x][edge_y])
	return mask


if __name__ == "__main__":
	main()
