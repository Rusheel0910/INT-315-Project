import csv
from pathlib import Path

from pyspark.sql import DataFrame


def write_spark_csv(dataframe: DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(dataframe.columns)
        for row in dataframe.toLocalIterator():
            writer.writerow([value if value is not None else "" for value in row])