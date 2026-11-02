from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.types import NumericType

from data_cleaning import CATEGORICAL_COLUMNS, TARGET_COLUMN, quoted_column
from export_utils import write_spark_csv


ROOT = Path(__file__).resolve().parents[1]
CLEANED_PATH = ROOT / "outputs" / "cleaned_data" / "students.csv"
ANALYSIS_PATH = ROOT / "outputs" / "analysis"
DIAGRAMS_PATH = ROOT / "diagrams"

CONTINUOUS_CANDIDATES = [
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (grade)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]
ACADEMIC_GRADE_CANDIDATES = [
    "Admission grade",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (grade)",
]
FINANCIAL_CANDIDATES = ["Unemployment rate", "Inflation rate", "GDP"]


def write_csv(dataframe: DataFrame, path: Path) -> None:
    write_spark_csv(dataframe, path)


def save_target_distribution(dataframe: DataFrame, target_counts: list) -> str:
    if TARGET_COLUMN not in dataframe.columns:
        return ""
    labels = [str(row[0]) for row in target_counts]
    counts = [int(row[1]) for row in target_counts]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(labels, counts, color="#287271")
    axis.set(title=f"Distribution of {TARGET_COLUMN}", xlabel=TARGET_COLUMN, ylabel="Students")
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    path = DIAGRAMS_PATH / "target_distribution.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path.name


def save_histograms(dataframe: DataFrame, columns: list[str]) -> str:
    if not columns:
        return ""
    figure, axes = plt.subplots(2, 4, figsize=(14, 8))
    for axis, name in zip(axes.flat, columns):
        values = (
            dataframe.select(quoted_column(name).cast("double").alias("value"))
            .where(F.col("value").isNotNull() & ~F.isnan("value"))
            .rdd.map(lambda row: row[0])
        )
        edges, counts = values.histogram(10)
        if len(edges) > 1:
            centers = [(edges[index] + edges[index + 1]) / 2 for index in range(len(counts))]
            widths = [edges[index + 1] - edges[index] for index in range(len(counts))]
            axis.bar(centers, counts, width=widths, color="#e09f3e", edgecolor="white")
        axis.set(title=name, xlabel="Value", ylabel="Records")
    for axis in axes.flat[len(columns) :]:
        axis.set_visible(False)
    figure.suptitle("Selected Numerical Feature Distributions")
    figure.tight_layout()
    path = DIAGRAMS_PATH / "numerical_histograms.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path.name


def correlation_matrix(dataframe: DataFrame, columns: list[str]) -> tuple[list[str], np.ndarray | None]:
    if len(columns) < 2:
        return columns, None
    assembled = VectorAssembler(
        inputCols=columns, outputCol="_correlation_features", handleInvalid="skip"
    ).transform(dataframe)
    matrix = Correlation.corr(
        assembled, "_correlation_features", "pearson"
    ).first()[0]
    return columns, matrix.toArray()


def save_correlation_plot(columns: list[str], matrix: np.ndarray | None) -> str:
    if matrix is None:
        return ""
    figure, axis = plt.subplots(figsize=(11, 9))
    image = axis.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(columns)), columns, rotation=75, ha="right")
    axis.set_yticks(range(len(columns)), columns)
    axis.set_title("Pearson Correlation of Numerical Features")
    figure.colorbar(image, ax=axis, label="Correlation")
    figure.tight_layout()
    path = DIAGRAMS_PATH / "numerical_correlation.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path.name


def save_academic_boxplot(dataframe: DataFrame, grade_column: str) -> str:
    quantiles = []
    targets = [row[0] for row in dataframe.select(TARGET_COLUMN).distinct().collect()]
    for target in sorted(targets, key=str):
        values = dataframe.where(quoted_column(TARGET_COLUMN) == target).select(
            quoted_column(grade_column).cast("double").alias("grade")
        ).where(F.col("grade").isNotNull())
        if values.limit(1).count() == 0:
            continue
        q1, median, q3 = values.approxQuantile("grade", [0.25, 0.5, 0.75], 0.001)
        bounds = values.agg(F.min("grade").alias("min"), F.max("grade").alias("max")).first()
        quantiles.append(
            {
                "label": str(target),
                "q1": q1,
                "med": median,
                "q3": q3,
                "whislo": bounds["min"],
                "whishi": bounds["max"],
                "fliers": [],
            }
        )
    if not quantiles:
        return ""
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bxp(quantiles, showfliers=False)
    axis.set(
        title=f"{grade_column} by {TARGET_COLUMN}",
        xlabel=TARGET_COLUMN,
        ylabel=grade_column,
    )
    figure.tight_layout()
    path = DIAGRAMS_PATH / "academic_performance_by_outcome.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path.name


def save_scholarship_status_plot(dataframe: DataFrame) -> str:
    status_column = "Scholarship holder"
    if status_column not in dataframe.columns or TARGET_COLUMN not in dataframe.columns:
        return ""
    grouped = (
        dataframe.groupBy(quoted_column(TARGET_COLUMN), quoted_column(status_column))
        .count()
        .collect()
    )
    outcomes = sorted({str(row[0]) for row in grouped})
    statuses = sorted({str(row[1]) for row in grouped})
    counts = {(str(row[0]), str(row[1])): int(row[2]) for row in grouped}
    positions = np.arange(len(outcomes))
    width = 0.8 / max(len(statuses), 1)
    figure, axis = plt.subplots(figsize=(9, 5))
    for index, status in enumerate(statuses):
        values = [counts.get((outcome, status), 0) for outcome in outcomes]
        axis.bar(positions + index * width, values, width, label=f"Recorded code {status}")
    axis.set_xticks(positions + width * (len(statuses) - 1) / 2, outcomes)
    axis.set(
        title="Recorded Scholarship Holder Status by Student Outcome",
        xlabel=TARGET_COLUMN,
        ylabel="Students",
    )
    axis.legend(title="Scholarship holder")
    figure.tight_layout()
    path = DIAGRAMS_PATH / "scholarship_holder_by_outcome.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path.name


def main() -> None:
    if not CLEANED_PATH.exists():
        raise FileNotFoundError(
            f"Run data_cleaning.py first; cleaned dataset not found at {CLEANED_PATH}"
        )

    spark = (
        SparkSession.builder.appName("INT315 Scholarship Dataset EDA")
        .master("local[*]")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    try:
        dataframe = (
            spark.read.option("header", "true")
            .option("sep", ",")
            .option("inferSchema", "true")
            .csv(str(CLEANED_PATH))
        )
        row_count = dataframe.count()
        column_count = len(dataframe.columns)
        print("\nDATASET DIMENSIONS")
        print(f"Rows: {row_count}")
        print(f"Columns: {column_count}")
        print("\nDATASET SCHEMA")
        dataframe.printSchema()
        sample_columns = [
            "Age at enrollment",
            "Admission grade",
            "Gender",
            "Scholarship holder",
            "Curricular units 1st sem (grade)",
            "Curricular units 2nd sem (grade)",
            TARGET_COLUMN,
        ]
        sample_columns = [name for name in sample_columns if name in dataframe.columns]
        print("\nSAMPLE RECORDS")
        dataframe.select(*[quoted_column(name) for name in sample_columns]).show(
            8, truncate=True, vertical=True
        )

        ANALYSIS_PATH.mkdir(parents=True, exist_ok=True)
        DIAGRAMS_PATH.mkdir(parents=True, exist_ok=True)
        dataframe.createOrReplaceTempView("students")
        numeric_columns = [
            field.name for field in dataframe.schema.fields if isinstance(field.dataType, NumericType)
        ]
        continuous_columns = [
            name for name in CONTINUOUS_CANDIDATES if name in numeric_columns
        ]
        grade_columns = [
            name for name in ACADEMIC_GRADE_CANDIDATES if name in numeric_columns
        ]
        categorical_columns = [
            name for name in dataframe.columns if name in CATEGORICAL_COLUMNS
        ]

        descriptive = dataframe.select(*numeric_columns).describe()
        write_csv(descriptive, ANALYSIS_PATH / "descriptive_statistics.csv")

        target_counts = []
        if TARGET_COLUMN in dataframe.columns:
            target_sql_name = TARGET_COLUMN.replace("`", "``")
            target_counts_df = spark.sql(
                f"SELECT `{target_sql_name}` AS label, COUNT(*) AS records "
                f"FROM students GROUP BY `{target_sql_name}` ORDER BY records DESC"
            )
            target_counts = target_counts_df.collect()
            write_csv(target_counts_df, ANALYSIS_PATH / "target_distribution.csv")

        frequencies = None
        for name in categorical_columns:
            part = dataframe.groupBy(quoted_column(name)).count().select(
                F.lit(name).alias("feature"),
                quoted_column(name).cast("string").alias("value"),
                F.col("count").alias("records"),
            )
            frequencies = part if frequencies is None else frequencies.unionByName(part)
        if frequencies is not None:
            write_csv(frequencies.orderBy("feature", F.desc("records")), ANALYSIS_PATH / "categorical_frequencies.csv")

        academic_df = None
        if TARGET_COLUMN in dataframe.columns and grade_columns:
            academic_df = dataframe.groupBy(quoted_column(TARGET_COLUMN)).agg(
                *[
                    F.avg(quoted_column(name)).alias(f"mean_{index + 1}")
                    for index, name in enumerate(grade_columns)
                ],
                *[
                    F.expr(
                        f"percentile_approx(`{name.replace('`', '``')}`, 0.5, 10000)"
                    ).alias(f"median_{index + 1}")
                    for index, name in enumerate(grade_columns)
                ],
                F.count(F.lit(1)).alias("records"),
            )
            write_csv(academic_df, ANALYSIS_PATH / "academic_performance_by_outcome.csv")

        financial_columns = [name for name in FINANCIAL_CANDIDATES if name in numeric_columns]
        financial_df = None
        if TARGET_COLUMN in dataframe.columns and financial_columns:
            financial_df = dataframe.groupBy(quoted_column(TARGET_COLUMN)).agg(
                *[
                    F.avg(quoted_column(name)).alias(name.replace(" ", "_"))
                    for name in financial_columns
                ],
                F.count(F.lit(1)).alias("records"),
            )
            write_csv(financial_df, ANALYSIS_PATH / "economic_indicators_by_outcome.csv")

        plots = [save_target_distribution(dataframe, target_counts)]
        plots.append(save_histograms(dataframe, continuous_columns))
        correlation_columns, matrix = correlation_matrix(dataframe, continuous_columns)
        plots.append(save_correlation_plot(correlation_columns, matrix))
        if TARGET_COLUMN in dataframe.columns and grade_columns:
            plots.append(save_academic_boxplot(dataframe, grade_columns[-1]))
        plots.append(save_scholarship_status_plot(dataframe))
        plot_files = [plot for plot in plots if plot]

        findings = [
            "# Exploratory Data Analysis Findings",
            "",
            f"The cleaned dataset contains {row_count:,} records and {column_count} columns.",
            "",
            "## Target and Scholarship Label",
            "",
            "The available `Target` column records student outcomes, not scholarship eligibility.",
            "`Scholarship holder` is a recorded holder-status field and is not treated as an eligibility decision or used to create a target.",
        ]
        if target_counts:
            total = sum(int(row[1]) for row in target_counts)
            distribution = "; ".join(
                f"{row[0]}: {int(row[1]):,} ({int(row[1]) / total:.1%})"
                for row in target_counts
            )
            class_sizes = [int(row[1]) for row in target_counts]
            ratio = max(class_sizes) / min(class_sizes) if min(class_sizes) else float("inf")
            findings.extend(
                [
                    "",
                    f"Observed `Target` distribution: {distribution}.",
                    f"Largest-to-smallest outcome class count ratio: {ratio:.2f}.",
                ]
            )
        if academic_df is not None:
            findings.extend(
                [
                    "",
                    "## Academic Performance",
                    "",
                    "Spark-computed mean and median grade summaries by student outcome are in `academic_performance_by_outcome.csv`; the box plot uses the second-semester grade where available.",
                    "These are descriptive associations and do not establish causation.",
                ]
            )
        if financial_df is not None:
            findings.extend(
                [
                    "",
                    "## Economic Features",
                    "",
                    f"The dataset contains these economic indicators: {', '.join(financial_columns)}. They are contextual indicators, not student family-income measurements.",
                    "Group averages by student outcome are in `economic_indicators_by_outcome.csv`.",
                ]
            )
        family_income_columns = [name for name in dataframe.columns if "income" in name.lower()]
        if not family_income_columns:
            findings.extend(
                [
                    "",
                    "No family-income column is present, so family-income analysis and its requested plot are not applicable.",
                ]
            )
        findings.extend(
            [
                "",
                "## Data Quality and Figures",
                "",
                "Detailed missing-value and duplicate counts are recorded in `cleaning_summary.json`. Exact duplicates were retained because the file contains no student identifier for judging whether repeated rows represent the same student.",
                f"Generated figures: {', '.join(plot_files) if plot_files else 'none; required fields were unavailable'}.",
                "",
                "A genuine scholarship-eligibility model requires an approved eligibility label. No derived label was created in this analysis.",
            ]
        )
        findings_path = ANALYSIS_PATH / "eda_findings.md"
        findings_path.write_text("\n".join(findings) + "\n", encoding="utf-8")
        print("\n".join(findings))
        print(f"EDA findings saved to {findings_path}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()