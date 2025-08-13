import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['font.family'] = 'CMU Serif'
plt.rcParams['mathtext.fontset'] = 'cm'


def visualize_matrix(matrix: np.ndarray[tuple[int, int]], filename: str, title: str, vmin=0, vmax=1) -> None:
	colors = ['#ff0000', '#ff7f00', '#ffff00', '#00ff00', '#00ffff', '#0000ff', '#8b00ff']
	cmap = LinearSegmentedColormap.from_list('custom_rainbow', colors)

	image = plt.imshow(matrix, cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)

	colorbar = plt.colorbar(image)
	colorbar.set_label(label='Pearson Correlation Coefficient $r_p$', labelpad=20)

	plt.title(title)
	plt.savefig(f'images/{filename}.png', dpi=300, bbox_inches='tight')
	plt.close()


def visualize_binary_matrix(matrix: np.ndarray[tuple[int, int]], filename: str, title: str) -> None:
	plt.imshow(matrix, cmap='gray', interpolation='nearest', vmin=0, vmax=1)

	plt.colorbar(label='Value')

	plt.title(title)
	plt.savefig(f'images/{filename}.png', dpi=300, bbox_inches='tight')
	plt.close()
