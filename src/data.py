from typing import Callable, Any
import numpy as np


def try_float(string: str) -> str | float:
	"""
	Returns a basic check of whether the given string is a float. Rejects
	strings with underscores.

	# Parameters

	- `string: str` - The string to check.

	# Returns
	
	Whether the given string is a float.
	"""
	if '_' in string: return string
	try: return float(string)
	except: return string


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
		self.values = list(map(try_float, values))

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

	entries: list[CSVEntry]

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

	def __len__(self) -> int:
		return len(self.entries)
