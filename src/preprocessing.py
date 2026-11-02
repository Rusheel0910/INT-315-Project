from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.types import NumericType

from data_cleaning import CATEGORICAL_COLUMNS, TARGET_COLUMN, quoted_column


RANDOM_SEED = 315


def feature_columns(dataframe: DataFrame) -> list[str]:
    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(f"Required source outcome column {TARGET_COLUMN!r} is missing.")
    return [name for name in dataframe.columns if name != TARGET_COLUMN]


def prepare_splits(dataframe: DataFrame) -> tuple[DataFrame, DataFrame, list[str], list[str], list[str]]:
    features = feature_columns(dataframe)
    numeric_features = [
        field.name
        for field in dataframe.schema.fields
        if field.name in features and isinstance(field.dataType, NumericType)
    ]
    categorical_features = [
        name for name in numeric_features if name in CATEGORICAL_COLUMNS
    ] + [
        field.name
        for field in dataframe.schema.fields
        if field.name in features and not isinstance(field.dataType, NumericType)
    ]
    continuous_features = [
        name for name in numeric_features if name not in categorical_features
    ]

    selected = dataframe.select(
        *[
            quoted_column(name).cast("string").alias(name)
            if name in categorical_features
            else quoted_column(name).cast("double").alias(name)
            for name in features
        ],
        quoted_column(TARGET_COLUMN).cast("string").alias(TARGET_COLUMN),
    ).where(quoted_column(TARGET_COLUMN).isNotNull())

    training, testing = selected.randomSplit([0.8, 0.2], seed=RANDOM_SEED)
    if training.limit(1).count() == 0 or testing.limit(1).count() == 0:
        raise ValueError("Train/test split produced an empty partition.")
    return training, testing, continuous_features, categorical_features, features


def build_preprocessing_pipeline(
    continuous_features: list[str], categorical_features: list[str]
) -> tuple[list, str]:
    stages = []
    imputed_continuous = continuous_features
    if continuous_features:
        imputed_continuous = [f"{name}__imputed" for name in continuous_features]
        stages.append(
            Imputer(
                inputCols=continuous_features,
                outputCols=imputed_continuous,
                strategy="median",
            )
        )

    indexed_categories = [f"{name}__indexed" for name in categorical_features]
    for name, output in zip(categorical_features, indexed_categories):
        stages.append(
            StringIndexer(inputCol=name, outputCol=output, handleInvalid="keep")
        )

    encoded_categories = [f"{name}__encoded" for name in categorical_features]
    if categorical_features:
        stages.append(
            OneHotEncoder(
                inputCols=indexed_categories,
                outputCols=encoded_categories,
                handleInvalid="keep",
            )
        )

    stages.append(StringIndexer(inputCol=TARGET_COLUMN, outputCol="label", handleInvalid="error"))
    stages.append(
        VectorAssembler(
            inputCols=imputed_continuous + encoded_categories,
            outputCol="features",
            handleInvalid="keep",
        )
    )
    return stages, "features"


def build_pipeline(continuous_features: list[str], categorical_features: list[str]) -> Pipeline:
    stages, _ = build_preprocessing_pipeline(continuous_features, categorical_features)
    return Pipeline(stages=stages)


def main() -> None:
    cleaned_path = Path(__file__).resolve().parents[1] / "outputs" / "cleaned_data" / "students.csv"
    if not cleaned_path.is_file():
        raise FileNotFoundError("Run src/data_cleaning.py before preprocessing.")
    spark = (
        SparkSession.builder.appName("INT315 Preprocessing Check")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        dataframe = (
            spark.read.option("header", "true")
            .option("sep", ",")
            .option("inferSchema", "true")
            .csv(str(cleaned_path))
        )
        training, testing, continuous, categorical, features = prepare_splits(dataframe)
        print(f"Training rows: {training.count()}")
        print(f"Testing rows: {testing.count()}")
        print(f"Continuous features ({len(continuous)}): {continuous}")
        print(f"Categorical features ({len(categorical)}): {categorical}")
        print(f"Target: {TARGET_COLUMN}; target excluded from the feature vector.")
        print(f"Pipeline stages: {len(build_pipeline(continuous, categorical).getStages())}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
