import mlflow
import mlflow.xgboost
import mlflow.pyfunc

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.functions import col, hour, to_timestamp, lit, udf
from pyspark.sql.types import StringType

from xgboost import XGBClassifier

from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, SparkTrials
from hyperopt.pyll.base import scope

import joblib
import logging
import json
import traceback
from typing import List, Dict, Any, Tuple
import yaml
import time
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.abspath('../'))
from reference import *
print(ENV_VARS, MODELS_NAME)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def initialize_spark():
    try:
        spark = SparkSession.builder \
            .appName("Hyperparameter Tuning") \
            .config("spark.executor.memory", "4g") \
            .config("spark.driver.memory", "4g") \
            .getOrCreate()
        return spark
    except Exception as e:
        logger.error(f"Failed to initialize Spark session: {str(e)}")
        raise


def get_current_user_email():
    try:
        return dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
    except Exception as e:
        logger.error(f"Failed to get user email: {str(e)}")
        return "unknown_user@fluentco.com"


def get_spark() -> SparkSession:
    try:
        from databricks.connect import DatabricksSession
        return DatabricksSession.builder.getOrCreate()
    except ImportError:
        return SparkSession.builder.getOrCreate()


def load_data(spark):
    try:
        query = """
        SELECT * 
        FROM centraldata_{}.datascience_model_deployment.adflow_click_training_data_version_2
        WHERE timestamp > date_add(DAY, -22, now())
        """.format(ENV_VARS['ENV'])
        df_spark = spark.sql(query)
        
        return df_spark
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise


def test_train_split(df_spark, train_test_ratio=0.8):
    try:
        train_test_splits = df_spark.randomSplit([train_test_ratio, 1 - train_test_ratio], seed=1234)
        train_data = train_test_splits[0]
        test_data = train_test_splits[1]
        return train_data.toPandas(), test_data.toPandas()
    except Exception as e:
        logger.error(f"Error in train-test split: {str(e)}")
        raise


def get_encodings(df, features):
    le_dict = {}
    for col in features:
        if col in df.columns:
            le_dict[col] = df[col].astype("category").cat.categories
    return le_dict


def encode_features(df, le_dict):
    for col in le_dict.keys():
        if col in df.columns:
            df[col] = df[col].astype(CategoricalDtype(categories=le_dict[col]))
        else:
            logger.warning(f"Column {col} not found in DataFrame")
    return df


def apply_feature_types(df: pd.DataFrame, features_types: Dict[str, type]) -> pd.DataFrame:
    for feature, dtype in features_types.items():
        if dtype == str:
            df[feature] = df[feature].astype('category')  # Ensure it's retained as a categorical type
        else:
            df[feature] = df[feature].astype(dtype)
    return df


def convert_integers_with_missing_to_float(df: pd.DataFrame, features_types: Dict[str, type]) -> pd.DataFrame:
    for feature, dtype in features_types.items():
        if pd.api.types.is_integer_dtype(df[feature]):
            # Check if the column is an integer and has null values
            if df[feature].isnull().any():
                df[feature] = df[feature].astype('float64')
    return df


def clean_os(os_name: str) -> str:
    if os_name in ["Android", "iOS", "Windows", "Mac"]:
        return os_name
    return "Other"


def clean_gender(gender: str) -> str:
    gender = gender.lower()
    if gender in ["f", "female"]:
        return "F"
    if gender in ["m", "male"]:
        return "M"
    return None


clean_os_udf = udf(clean_os, StringType())
clean_gender_udf = udf(clean_gender, StringType())


def clean_data(df_spark):
    df_spark = df_spark.withColumn("os_name", clean_os_udf(df_spark["os_name"]))
    df_spark = df_spark.withColumn("gender", clean_gender_udf(df_spark["gender"]))
    return df_spark


def main():
    try:
        # Initialize Spark
        spark = get_spark()

        # Get the current user's email to set the experiment name
        user_email = get_current_user_email()

        experiment_name = f"/Users/{user_email}/{MODELS_NAME['MODEL1']}_{ENV_VARS['ENV']}_hyperparams"

        # Set MLflow experiment
        mlflow.set_experiment(experiment_name)

        # Retrieve experiment id by name
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        print('experiment_id', experiment_id)

        # Start MLflow run with the model name as the run name
        with mlflow.start_run(run_name=f"{MODELS_NAME['MODEL1']}_{ENV_VARS['ENV']}_run") as run:
            # run_id = run.info.run_id
        
            features_types = {
                    "gender": str, 
                    "campaign_id": str, 
                    "traffic_partner_id": str, 
                    "traffic_source_id": float,  # Updated from Int64 to float64
                    "os_name": str, 
                    "age": float,  # Updated from Int64 to float64
                    "hour": pd.Int64Dtype(), 
                    "household_income": float,
                    "position": pd.Int64Dtype(),  # Retains Int64 as it doesn't have nulls
                }
            
            features = list(features_types.keys())
            categorical_features = [
                "gender", "campaign_id", "traffic_partner_id", "traffic_source_id", 
                "os_name", "hour", "position",
            ]

            target = "click"

            # Load and preprocess data
            logger.info("Loading data...")
            df_spark_ = load_data(spark)
            df_spark = clean_data(df_spark_)

            # Add null values for robustness testing
            null_samples = {
                'campaign_id': df_spark.sample(False, 0.05).withColumn('campaign_id', F.lit(None)),
                'traffic_source_id': df_spark.sample(False, 0.05).withColumn('traffic_source_id', F.lit(None)),
                'traffic_partner_id': df_spark.sample(False, 0.05).withColumn('traffic_partner_id', F.lit(None))
            }
            df_spark = df_spark.unionAll(null_samples['campaign_id']) \
                            .unionAll(null_samples['traffic_source_id']) \
                            .unionAll(null_samples['traffic_partner_id'])

            logger.info("Splitting data into train and test sets...")
            train_pd, test_pd = test_train_split(df_spark, train_test_ratio=0.8)

            train_pd = apply_feature_types(train_pd, features_types)
            test_pd = apply_feature_types(test_pd, features_types)

            logger.info(f"Train DataFrame dtypes after applying features_types:\n{train_pd.dtypes}")
            logger.info(f"Test DataFrame dtypes after applying features_types:\n{test_pd.dtypes}")

            train_pd = convert_integers_with_missing_to_float(train_pd, features_types)
            test_pd = convert_integers_with_missing_to_float(test_pd, features_types)

            logger.info(f"Train DataFrame dtypes after converting integers with missing values to float:\n{train_pd.dtypes}")
            logger.info(f"Train DataFrame dtypes after converting integers with missing values to float:\n{test_pd.dtypes}")
            
            encodings = get_encodings(train_pd, categorical_features)
            train_pd = encode_features(train_pd, encodings)
            test_pd = encode_features(test_pd, encodings)

            X_train = train_pd[features]
            y_train = train_pd[target]
            X_test = test_pd[features]
            y_test = test_pd[target]

            # Define the objective function for hyperopt
            def objective(params):
                model = XGBClassifier(**params)
                model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
                preds = model.predict_proba(X_test)[:, 1]
                loss = log_loss(y_test, preds)
                return {'loss': loss, 'status': STATUS_OK}

            # Define the search space
            search_space = {
                'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.1)),
                'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 0.7),
                'subsample': hp.uniform('subsample', 0.6, 1.0),
                'early_stopping_rounds': scope.int(hp.quniform('early_stopping_rounds', 1, 10, 1)),
                'n_estimators': scope.int(hp.quniform('n_estimators', 300, 400, 1)),
                'max_depth': scope.int(hp.quniform('max_depth', 5, 8, 1)),
                'enable_categorical': hp.choice('enable_categorical', [True]),
                'eval_metric': hp.choice('eval_metric', ['logloss']),
                'tree_method': hp.choice('tree_method', ['hist']),
                "max_cat_to_onehot": hp.choice('max_cat_to_onehot', [1])
            }

            # Run hyperopt
            trials = SparkTrials(parallelism=4)
            with mlflow.start_run(run_name="Model Tuning with Hyperopt Demo1", nested=True) as parent_run:
                best_params = fmin(
                    fn=objective,
                    space=search_space,
                    algo=tpe.suggest,
                    max_evals=2,
                    trials=trials
                )

            # Convert best_params to the correct format
            # best_params = {k: v[0] if isinstance(v, list) else v for k, v in best_params.items()}
            print(f"Best hyperparameters: {best_params}")

            mlflow.log_param(f"{MODELS_NAME['MODEL1']}_hyperparams", best_params)

    except Exception as e:
        error_message = f"An error occurred in main execution: {str(e)}\nTraceback: {traceback.format_exc()}"
        logger.error(error_message)
        raise

    logger.info("Script execution completed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        critical_error = f"Critical error in main script execution: {str(e)}\nTraceback: {traceback.format_exc()}"
        logger.critical(critical_error)
        raise
