# INT315 Scholarship Eligibility Dataset: Cleaning and EDA

This project inspects and explores the supplied student dataset with PySpark. It does not train a scholarship-eligibility model because the file contains no verified eligibility decision.

## Dataset

The source is the existing root-level `data.csv`. Inspection found a UTF-8 CSV with a byte-order mark and semicolon delimiters, not an Excel workbook. It contains 4,424 data rows and 37 fields; every parsed row has the same 37 fields. The columns only appear combined when the CSV is opened with a comma delimiter. The scripts read it with the correct separator and do not modify or copy over the original.

The `Target` field contains student outcomes (`Dropout`, `Graduate`, and `Enrolled`). `Scholarship holder` records holder status; it is not an eligibility decision. Neither field is represented as genuine scholarship eligibility, and no derived label is created. The dataset has economic indicators (`Unemployment rate`, `Inflation rate`, and `GDP`) but no student family-income field.

## What the scripts do

`src/data_cleaning.py` reads the file into a Spark DataFrame, normalizes header whitespace and string values, infers source types, reports missing values and exact duplicate records, imputes missing feature values with medians or modes, leaves `Target` missing values untouched, and writes a cleaned CSV plus a JSON cleaning summary. Exact duplicates are reported but retained because there is no student identifier to establish whether they are redundant.

`src/eda.py` reads the cleaned output and uses Spark DataFrame and Spark SQL operations for descriptive statistics, target counts, categorical frequencies, academic summaries, and economic-indicator comparisons. It generates Matplotlib figures from Spark-aggregated data and writes a findings summary. It uses actual columns only; family-income and eligibility plots are omitted because those fields are unavailable.

Expected generated outputs:

- `outputs/cleaned_data/students.csv/`: Spark CSV output directory containing the cleaned data.
- `outputs/analysis/cleaning_summary.json`: row, missing-value, duplicate, and imputation summary.
- `outputs/analysis/`: descriptive statistics, target and category counts, grouped summaries, and EDA findings.
- `diagrams/`: target distribution, numerical histograms, correlation heatmap, academic performance by student outcome, and recorded scholarship-holder status by outcome, when the corresponding columns are present.

## Environment and execution

Use a Python version supported by the installed Spark release. For the inspected Spark 3.5.9 installation, Python 3.10 is available and was used for the read/schema execution check. Java is required. Install project dependencies in the environment used by `spark-submit`:

```powershell
python -m pip install -r requirements.txt
spark-submit --master 'local[*]' src/data_cleaning.py
spark-submit --master 'local[*]' src/eda.py
```

On Windows, local Spark output may require Hadoop's `winutils.exe` and a configured `HADOOP_HOME`. This machine's Spark installation does not include it, so the cleaning run reached schema and sample display but failed when writing output. Matplotlib is also absent in the checked Python environments. The output files and diagrams have therefore not yet been generated or runtime-verified here. The original `data.csv` remains unchanged.