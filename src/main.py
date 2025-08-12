from data import CSV
import os


def main():
	demographics = CSV("data/1000BRAINS/sourceFiles/demographics.csv", where = lambda entry: entry["ID"].endswith("1"))
	processing_speed: list[float] = demographics["Processing_Speed_raw"]
	reasoning: list[float] = demographics["Reasoning_raw"]

	structural_connectomes = []
	connectomes_folder = "data/1000BRAINS/connectomes/structuralConnectomes"
	for filename in filter(lambda file: file.endswith("1.csv"), os.listdir(connectomes_folder)):
		connectome = CSV(f"{connectomes_folder}/{filename}")
		structural_connectomes.append(connectome)


if __name__ == "__main__":
	main()
