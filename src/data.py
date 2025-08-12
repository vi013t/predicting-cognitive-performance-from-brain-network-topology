from typing import Callable, Any
import numpy as np


def is_float(string: str) -> bool:
	if '_' in string: 
		return False
	try:
		float(string)
		return True
	except:
		return False


class CSVEntry:
	"""
	An entry (row) in a CSV file.
	"""

	categories: list[str]
	"""
	The categories of the CSV file, as listed in the first row of the file.
	"""

	values: list[Any]
	"""
	The values stored in this entry.
	"""

	def __init__(self, categories: list[str], values: list[Any]):
		self.categories = categories
		self.values = values
		if all(is_float(value) for value in self.values):
			self.values = list(map(float, self.values))

	def __getitem__(self, name: str) -> Any:
		return self.values[self.categories.index(name)]


class CSV:
	"""
	A utility object for fetching data from a CSV file.
	"""

	data: list[list[Any]]
	"""
	The raw data matrix of the CSV.
	"""

	def __init__(self, filename: str, where: Callable[[CSVEntry], bool] = lambda _: True):
		"""
		Loads CSV data from a file.

		# Parameters

		- `filename: str` - The path to the CSV file
		- `where: Callable[[CSVEntry], bool]` - A filter predicate in which entries that do not return `True`
		will be removed

		# Returns

		A `CSV` object holding data from the CSV file.
		"""
		data = np.loadtxt(filename, delimiter = ",", dtype = str)
		entries = [CSVEntry(list(data[0]), list(entry)) for entry in data[1:]]
		self.entries = [entry for entry in entries if where(entry)]
		self.data = [data[0]] + [entry for entry in self.entries if where(entry)]

	def __getitem__(self, name: str) -> list[Any]:
		return [entry[name] for entry in self.entries]
