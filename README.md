# INT315 Scholarship Eligibility Prediction Project

## Scope and Label Limitation

The supplied file does **not** contain a verified scholarship-eligibility decision. Its `Target` column contains student outcomes (`Dropout`, `Graduate`, `Enrolled`); `Scholarship holder` records holder status, not eligibility. To avoid fabricating institutional decisions, the runnable Decision Tree demonstrates classification of the actual `Target` student outcome. Its scores must not be reported as scholarship-eligibility performance. A genuine eligibility model needs an approved eligibility label and appropriate feature timing.

## Problem Statement and Objectives

The requested assignment concerns scholarship eligibility, but the available workbook labels student academic outcomes rather than eligibility. The defensible implemented problem is therefore to classify the recorded student outcome from the other 36 available columns. This project demonstrates Spark-based data preparation and classification while clearly documenting that the model is not a scholarship decision system.

Objectives are to inspect and validate the supplied workbook; clean and summarize its records using PySpark; explore outcome, academic and economic patterns; prepare actual numeric and coded-category features without training/test leakage; train one Spark MLlib Decision Tree; evaluate multiclass outcomes; and prepare transparent Power BI inputs for the held-out test results.

## Tools and Technologies

Python 3.14.2, Apache Spark/PySpark 4.1.4, Java 21, openpyxl for reading the Excel workbook, and Matplotlib for static EDA figures. Power BI Desktop is present on the machine, but building and testing the PBIX requires manual interaction with its Windows UI. No pandas, scikit-learn, database, or cloud service is used for processing/modeling.

## Dataset

The primary input is the root-level `scholarship_dataset_fixed.xlsx`, a genuine Excel workbook. It has one visible worksheet, `Student Data`, with 4,424 student records and 37 columns (`A1:AK4425`). The workbook header and sampled rows match the existing `data.csv` source; the cleaner now reads the workbook first and uses the original semicolon-delimited CSV only as a fallback if the workbook is absent. Both source files remain unchanged. Excel cell values are read with `openpyxl`, then passed to Spark for all data cleaning, analysis, and model operations. Numeric columns containing both whole and decimal cell values are normalized to a consistent Spark numeric type during ingestion. The actual columns include academic/course history, demographic and coded categories, `Scholarship holder`, economic context (`Unemployment rate`, `Inflation rate`, `GDP`), and `Target`. There is no family-income column.

## Project Structure

```text
.
├── scholarship_dataset_fixed.xlsx
├── data.csv (fallback)
├── src/
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── export_utils.py
│   ├── preprocessing.py
│   ├── train_model.py
│   └── evaluation.py
├── outputs/
│   ├── cleaned_data/students.csv
│   ├── analysis/
│   ├── predictions/test_predictions.csv
│   └── metrics/
├── diagrams/
├── powerbi/Dashboard_Guide.md
├── screenshots/
├── README.md
├── requirements.txt
└── .gitignore
```

`students.csv` and analysis exports are ordinary CSV/JSON/Markdown files. Transformations and aggregations are performed with Spark. `src/export_utils.py` streams Spark rows to CSV with Python's standard CSV writer because the Windows Spark installation lacks `winutils.exe`; it does not perform data analysis.

## Environment Setup

Requires Python 3.14, Java 17 or newer, and the dependencies in `requirements.txt` (`pyspark`, `matplotlib`, and `openpyxl`). The implementation was verified using Python 3.14.2, PySpark 4.1.4, Matplotlib 3.11.2, openpyxl, and Java 21. Spark 4.1.4 is installed in `.venv`; do not launch the separate old `C:\Spark` 3.5.9 distribution with it. From PowerShell in this project folder:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:JAVA_HOME = 'C:\Program Files\Eclipse Adoptium\jdk-21.0.12.101-hotspot'
$env:PATH = "$env:JAVA_HOME\bin;$env:PATH"
$env:SPARK_HOME = ''
$env:PYSPARK_PYTHON = (cmd /c 'for %I in (".venv\Scripts\python.exe") do @echo %~sI').Trim()
$env:PYSPARK_DRIVER_PYTHON = $env:PYSPARK_PYTHON
```

The short worker-Python path avoids command parsing failures when the workspace path contains spaces. If Java is installed elsewhere, set `JAVA_HOME` to its JDK 17+ directory. Spark may log a missing `winutils.exe` warning on Windows; CSV exports use the compatibility writer described above.

## Execution

Run in order from the project root using the configured environment:

```powershell
& $env:PYSPARK_PYTHON .\src\data_cleaning.py
& $env:PYSPARK_PYTHON .\src\eda.py
& $env:PYSPARK_PYTHON .\src\preprocessing.py
& $env:PYSPARK_PYTHON .\src\train_model.py
& $env:PYSPARK_PYTHON .\src\evaluation.py
```

Cleaning and EDA can be run independently before the model. `preprocessing.py` validates the reproducible 80:20 split and feature composition; the training pipeline fits all imputers and encoders on training data only.

## Data Cleaning

`data_cleaning.py` reads the primary Excel workbook with `openpyxl`, validates the expected worksheet and consistent row width, then creates the Spark DataFrame used for all data processing. It normalizes mixed integer/decimal numeric cells to consistent numeric types, trims header/value whitespace, reports per-column missingness and exact duplicates, normalizes numeric NaN values to nulls, and never imputes the target. If the workbook is absent, it falls back to Spark's CSV reader with the semicolon delimiter and fail-fast malformed-row handling. For this workbook, execution found 0 missing values, 0 exact duplicate rows, removed 0 records, and retained all 4,424 records. Duplicates are retained rather than removed because there is no student identifier to assess repeated records. Feature nulls, if present, are preserved by cleaning so no test-set information is used to calculate fill values; the model's median imputer is fitted on training records only, while categorical missing/unseen values use the fitted indexer's invalid bucket.

The cleaning summary is `outputs/analysis/cleaning_summary.json`; it records `scholarship_dataset_fixed.xlsx` and worksheet `Student Data` as the source. The cleaned Power BI-friendly comma-delimited CSV is `outputs/cleaned_data/students.csv`.

## Exploratory Analysis

`eda.py` uses Spark DataFrames and Spark SQL for descriptive statistics, target counts, category frequencies, grade summaries by outcome, and economic-indicator comparisons. It generates Matplotlib plots from Spark aggregates/distributed histogram results. Verified `Target` counts are Graduate 2,209 (49.9%), Dropout 1,421 (32.1%), and Enrolled 794 (17.9%), a largest/smallest class ratio of 2.78. These outcome associations are descriptive, not causal. Economic indicators are not household income.

Analysis exports in `outputs/analysis/` include descriptive statistics, target distribution, categorical frequencies, academic performance by outcome, economic indicators by outcome, cleaning summary, and `eda_findings.md`. Five generated diagrams are in `diagrams/`: target distribution, numerical histograms, numeric correlation heatmap, academic performance by outcome, and recorded scholarship-holder status by outcome. Family-income and eligibility plots are omitted because the corresponding fields are absent.

## Preprocessing and Model

The real `Target` is the label. The 36 other actual source columns are the feature inputs; no synthetic eligibility field is created. Coded category fields are cast to strings and indexed/one-hot encoded. Other numeric fields are median-imputed with Spark ML `Imputer`; category fields use Spark `StringIndexer` and `OneHotEncoder` with invalid values retained in an extra bucket. A seeded 80:20 split is made before fitting these transformations. There is no feature scaling because decision trees do not require it. A single PySpark MLlib `DecisionTreeClassifier` with `maxDepth=5` is trained in a Pipeline.

The dataset includes second-semester results, so this is a retrospective outcome-classification demonstration, not an admission-time forecast. No trained model file is persisted. `train_model.py` writes the held-out results to `outputs/predictions/test_predictions.csv`, including source features, actual/predicted outcomes, class indices, and prediction confidence.

## Evaluation and Results

`evaluation.py` uses Spark's multiclass evaluator for accuracy, weighted precision, weighted recall, and weighted F1. It also exports one-vs-rest precision, recall, F1, and support per outcome class, and the full confusion matrix. In the verified run, the test set had 829 students:

| Metric | Result |
|---|---:|
| Accuracy | 0.7479 |
| Weighted precision | 0.7602 |
| Weighted recall | 0.7479 |
| Weighted F1 | 0.7502 |

Per-class recall: Dropout 0.7184 (277 test examples), Enrolled 0.4586 (133), Graduate 0.8592 (419). Accuracy alone obscures the lower recall for Enrolled. The confusion matrix defines rows as actual outcome and columns as predicted outcome; diagonal cells are correct classifications. Off-diagonal cells represent errors (actual Dropout predicted Graduate, for example). These are student-outcome metrics, not scholarship eligibility metrics.

Exports in `outputs/metrics/`: `evaluation_metrics.csv`, `per_class_metrics.csv`, `confusion_matrix.csv`, `actual_vs_predicted.csv`, and `outcome_distribution.csv`. The test prediction file contains 829 data rows; the confusion matrix includes zero-count class pairs and sums to the same test total.

## Report-Ready Conclusion

The workbook contains 4,424 student records across 37 columns, with no missing values or exact duplicate rows detected during the verified cleaning run. Spark EDA found outcome counts of 2,209 Graduate, 1,421 Dropout, and 794 Enrolled. The Decision Tree classified the recorded outcome correctly for 74.79% of the 829 held-out records, with weighted F1 of 0.7502. Class-level recall varied, especially for Enrolled (0.4586), so aggregate accuracy alone is incomplete. These results demonstrate a Spark multiclass workflow on this dataset; they do not establish scholarship eligibility or predict scholarship awards. The model also uses later semester results, making it retrospective rather than a pre-enrollment eligibility assessment.

For a report, cite the plots in `diagrams/`, the grouped academic/economic exports in `outputs/analysis/`, the test metrics and confusion matrix in `outputs/metrics/`, and actual test predictions in `outputs/predictions/test_predictions.csv`. Do not invent execution screenshots: capture the Power BI page after creating it and place those genuine screenshots in `screenshots/`.

## Power BI

Power BI Desktop is installed, but this VS Code session cannot operate its native UI. `powerbi/Scholarship_Dashboard.pbix` has **not** been created or tested. Follow [powerbi/Dashboard_Guide.md](powerbi/Dashboard_Guide.md) to import each actual CSV, create the five KPI DAX measures, configure all five visuals and slicers, validate totals, and save the PBIX. The one-page report is titled **Student Outcome Classification Evaluation** and must not present eligibility counts.

`diagrams/` contains real generated EDA figures suitable for an academic report. Power BI screenshots are not present; capture screenshots after building the PBIX and add them to `screenshots/`. Do not substitute EDA figures for screenshots of the Power BI page.

## Known Limitations

- There is no verified eligibility label or family-income feature.
- `Target` classifies student outcome and cannot support claims about scholarship eligibility.
- No student identifier is available for deduplication.
- The split is random and reproducible, but this dataset's class distribution is imbalanced.
- The tree uses semester results, making it retrospective rather than a pre-study scholarship forecast.
- Manual Power BI report creation, PBIX validation, and screenshot capture remain outstanding.
