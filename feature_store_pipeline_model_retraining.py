import mlflow
import mlflow.xgboost
import mlflow.pyfunc

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.functions import col, hour, to_timestamp, lit, udf
from pyspark.sql.types import StringType, FloatType

from xgboost import XGBClassifier

from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from databricks.feature_store import FeatureLookup
from databricks.feature_engineering import FeatureEngineeringClient

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, SparkTrials

import joblib
import logging
import json
import traceback
from typing import List, Dict, Any, Tuple, Optional, Union
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

RUN_ID = dbutils.jobs.taskValues.get(taskKey="data_ingestion_task", key="run_id")
print(f"RUN_ID: {RUN_ID}")


def initialize_spark():
    try:
        spark = SparkSession.builder \
            .appName(MODELS_NAME['MODEL2']) \
            .config("spark.executor.memory", "4g") \
            .config("spark.driver.memory", "4g") \
            .getOrCreate()
        return spark
    except Exception as e:
        logger.error(f"Failed to initialize Spark session: {str(e)}")
        raise


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


def apply_feature_types(df: pd.DataFrame, features_types: Dict[str, type]) -> pd.DataFrame:
    for feature, dtype in features_types.items():
        if dtype == str:
            df[feature] = df[feature].astype('category')  # Ensure it's retained as a categorical type
        else:
            df[feature] = df[feature].astype(dtype)
    return df


def convert_integers_with_missing_to_float(df: pd.DataFrame, features_types: Dict[str, type]) -> pd.DataFrame:
    for feature, dtype in features_types.items():
        if feature != 'campaign_id' and pd.api.types.is_integer_dtype(df[feature]):
            # Check if the column is an integer and has null values
            if df[feature].isnull().any():
                df[feature] = df[feature].astype('float64')
    return df


def create_robust_signature(X_train: pd.DataFrame, model: XGBClassifier, n_examples: int = 5) -> tuple:
    try:
        input_example = X_train.sample(n=n_examples, random_state=42).copy()
        input_example = input_example.astype(str)
        for col in input_example.columns:
            input_example.loc[input_example.index[0], col] = None

        signature = mlflow.models.signature.infer_signature(input_example, model.predict(None, X_train))
        return signature, input_example
    except Exception as e:
        logger.error(f"Error in create_robust_signature: {str(e)}")
        raise


# Custom transformer for cleaning data
class DataCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # Nothing to fit
    

    def transform(self, X):
        # Assuming X is a DataFrame
        X_copy = X.copy()
        X_copy["os_name"] = X_copy["os_name"].apply(self.clean_os)
        X_copy["gender"] = X_copy["gender"].apply(self.clean_gender)
        X_copy["age"] = X_copy["age"].apply(self.clean_age)
        X_copy["household_income"] = X_copy["household_income"].apply(self.clean_household_income)
        return X_copy
    

    @staticmethod
    def clean_os(os_name: str) -> str:
        if os_name is None:
            return None  # Return None or a default value for null os_name
        if os_name in ["Android", "iOS", "Windows", "Mac"]:
            return os_name
        return "Other"


    @staticmethod
    def clean_gender(gender: str) -> str:
        if gender is None:
            return None  # Return None or a default value for null gender
        gender = gender.lower()
        if gender in ["f", "female"]:
            return "F"
        if gender in ["m", "male"]:
            return "M"
        return None


    @staticmethod
    def clean_age(age: Optional[Union[str, int, float]]) -> Optional[float]:
        if age in ["null", "NA", "", None, -1]:
            return np.nan
        if isinstance(age, (int, float)):
            return float(age)
        if isinstance(age, str):
            try:
                return float(age)
            except ValueError:
                return np.nan
        return np.nan


    @staticmethod
    def clean_household_income(household_income: Optional[Union[str, int, float]]) -> Optional[float]:
        if household_income in ["null", "NA", "", None, -1]:
            return np.nan
        if isinstance(household_income, (int, float)):
            return float(household_income)
        if isinstance(household_income, str):
            try:
                return float(household_income)
            except ValueError:
                return np.nan
        return np.nan
      

# Custom transformer for encoding
class CustomEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features
        self.le_dict = {}
    

    def fit(self, X, y=None):
        for col in self.features:
            if col in X.columns:
                self.le_dict[col] = X[col].astype("category").cat.categories
        return self


    def transform(self, X):
        X_copy = X.copy()
        for col in self.le_dict.keys():
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].astype(CategoricalDtype(categories=self.le_dict[col]))
            else:
                print(f"Column {col} not found in DataFrame")
        # Return the entire DataFrame, including both transformed and untransformed columns
        return X_copy


def log_model_and_metrics(model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, training_set: pd.DataFrame,
                          hyperparams: Dict, features_types: Dict, pip_requirements: List[str], run_id):
    try:
        start_time = time.time()
        with mlflow.start_run(run_id=run_id) as run:
            test_preds = model.predict_proba(X_test)[:, 1]
            test_log_loss = log_loss(y_test, test_preds)
            train_preds = model.predict_proba(X_train)[:, 1]
            train_log_loss = log_loss(y_train, train_preds)

            mlflow.log_param(f"{MODELS_NAME['MODEL2']}_version", 1)
            mlflow.log_param(f"{MODELS_NAME['MODEL2']}_hyperparams", hyperparams)
            mlflow.log_metric(f"{MODELS_NAME['MODEL2']}_test_log_loss", test_log_loss)
            mlflow.log_metric(f"{MODELS_NAME['MODEL2']}_train_log_loss", train_log_loss)

            mlflow.log_param(f"{MODELS_NAME['MODEL2']}_training_environment", f"https://{spark.conf.get('spark.databricks.workspaceUrl')}")
            mlflow.log_param(f"{MODELS_NAME['MODEL2']}_training_timestamp", datetime.now().isoformat())
            end_time = time.time()
            training_duration = timedelta(seconds=end_time - start_time)
            mlflow.log_param(f"{MODELS_NAME['MODEL2']}_training_duration", str(training_duration))
            mlflow.log_param(f"{MODELS_NAME['MODEL2']}_training_run_id", run.info.run_id)

            features_types_str = {k: str(v) for k, v in features_types.items()}

            mlflow.log_param(f"{MODELS_NAME['MODEL2']}_features_types", json.dumps(features_types_str))
            mlflow.log_param(f"{MODELS_NAME['MODEL2']}_encoding_scheme", "CategoricalDtype")
            mlflow.log_param(f"{MODELS_NAME['MODEL2']}_scaling_strategy", "None")
            
            fe = FeatureEngineeringClient()

            pipeline_model = PipelineModelWrapper(RUN_ID, model)

            signature, _ = create_robust_signature(X_train, pipeline_model)

            pipeline_model_name = f"pipeline-{MODELS_NAME['MODEL2']}"

            model_info = fe.log_model(
                model=pipeline_model,
                artifact_path=pipeline_model_name,
                signature=signature,
                flavor=mlflow.pyfunc,
                training_set=training_set,
                pip_requirements = pip_requirements
            )

            model_uri = f"runs:/{RUN_ID}/{pipeline_model_name}"

            return run_id, model_uri
        
    except Exception as e:
        logger.error(f"Error in logging model and metrics: {str(e)}")
        raise


def train_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, hyperparams: Dict) -> XGBClassifier:
    try:
        model = XGBClassifier(**hyperparams)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        return model
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise


class PipelineModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, run_id, model):
      self.run_id = run_id
      self.model = model

      self.loaded_features_types = {
          "gender": 'string', 
          "campaign_id": 'string', 
          "traffic_partner_id": 'string', 
          "traffic_source_id": 'string', 
          "os_name": 'string', 
          "age": 'float64', 
          "hour": 'string', 
          "household_income": 'float64',
          "position": 'string'
      }
      self.fs_features = ['vertical', 'subvertical']


    def predict(self, context, model_input):
        try:
            if isinstance(model_input, dict) and 'instances' in model_input:
                model_input = model_input['instances']

            if isinstance(model_input, pd.DataFrame):
                df = model_input
            else:
                return 'Invalid model input specified'

            # Generate predictions
            pred_prob = self.model.predict_proba(df)
            df['model_ctr'] = pred_prob[:, 1]
            df['model_version'] = f"adflow_click_{MODELS_NAME['MODEL2']} - " + str(self.run_id)

            specific_columns = ['campaign_id', 'position', 'model_ctr', "model_version"]
            final = df[specific_columns]

            return final
        except Exception as e:
            error_message = f"Error processing input: {e}"
            logger.error(error_message)
            raise e


def train_and_register_model(run_id):
    try:
        # Initialize Spark
        spark = initialize_spark()

        features_types = {
            "gender": StringType(), 
            "campaign_id": StringType(), 
            "traffic_partner_id": StringType(), 
            "traffic_source_id": StringType(), 
            "os_name": StringType(), 
            "age": StringType(), 
            "hour": StringType(), 
            "household_income": StringType(),
            "position": StringType(),
            "vertical": StringType(),
            "subvertical": StringType()
        }

        # feature store look up features
        fs_features = ['vertical', 'subvertical']

        # all features
        model_features = list(features_types.keys())
        features = [feature for feature in features_types.keys() if feature not in fs_features]
        
        # loaded data feautures
        loaded_features = set(features) - set(fs_features)

        # features which we want to have the 'category' dtype.
        categorical_features = [
            "gender", 
            "campaign_id", 
            "traffic_partner_id", 
            "traffic_source_id", 
            "os_name", 
            "hour",
            "position",
            "vertical",
            "subvertical"
        ]

        target = "click"

        fe = FeatureEngineeringClient()

        # Define the feature lookup for the feature table you want to use
        feature_lookups = [
            FeatureLookup(
                table_name=f'centraldata_prod.minion.gold_campaign',
                feature_names=fs_features,  # The names of the features you want to use
                lookup_key='campaign_id'  # The key column to join on
            )
        ]

        hyperparams = {
            'colsample_bytree': 0.6421218939603257,
            'early_stopping_rounds': 9,
            'learning_rate': 0.03711165197308055, 
            'max_depth': 8, 
            'n_estimators': 369,
            'subsample': 0.9173736479523668,
            'enable_categorical': True, 
            'eval_metric': 'logloss', 
            'tree_method': 'hist',
            'max_cat_to_onehot': 1,
        }

        # Load and preprocess data
        logger.info("Loading data...")

        # Load Data and get the loaded data and features
        df_spark = load_data(spark)

        # Iterate over the loaded features and cast the corresponding columns to the desired data types
        for feature in features:
            if feature in features_types:
                data_type = features_types[feature]
                df_spark = df_spark.withColumn(feature, F.col(feature).cast(data_type))

        # Select the input features (without the feature store features) and click
        df_spark = df_spark.select(*loaded_features, "click")

        df_spark = df_spark.withColumn('click', F.col('click').cast(FloatType()))

        # Create a full data set by joining your loaded data with the feature table
        training_set = fe.create_training_set(
            df=df_spark,
            feature_lookups=feature_lookups,
            label=target,  # The name of the label column in your loaded data
        )

        df_spark = training_set.load_df()

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

        logger.info(f"Train DataFrame dtypes after applying features_types:\n{train_pd.dtypes}")
        logger.info(f"Test DataFrame dtypes after applying features_types:\n{test_pd.dtypes}")

        train_pd = convert_integers_with_missing_to_float(train_pd, features_types)
        test_pd = convert_integers_with_missing_to_float(test_pd, features_types)

        logger.info(f"Train DataFrame dtypes after converting integers with missing values to float:\n{train_pd.dtypes}")
        logger.info(f"Train DataFrame dtypes after converting integers with missing values to float:\n{test_pd.dtypes}")

        X_train = train_pd.drop(target, axis=1)
        y_train = train_pd[target]
        X_test = test_pd.drop(target, axis=1)
        y_test = test_pd[target]

        data_cleaner = DataCleaner()
        X_train = data_cleaner.transform(X_train)
        X_test = data_cleaner.transform(X_test)

        # Apply CustomEncoder
        custom_encoder = CustomEncoder(features=categorical_features)
        encodings = custom_encoder.fit(X_train)
        # and apply to the train, validation, and test sets
        X_train = encodings.transform(X_train)
        X_test = encodings.transform(X_test)

        # Train model
        logger.info("Training the model...")
        model = train_model(X_train, y_train, X_test, y_test, hyperparams)

        data_cleaner_transformer = FunctionTransformer(data_cleaner.transform)
        custom_encoder_transformer = FunctionTransformer(lambda X: custom_encoder.transform(X))

        pipeline_model = Pipeline(steps=[
            ('data_cleaner', data_cleaner_transformer),
            ('custom_encoder', custom_encoder_transformer),
            ('model', model)
        ])

        # Define pip requirements
        pip_requirements = [
            "mlflow==2.11.3", "scikit-learn==1.3.0", "scipy==1.10.0",
            "psutil==5.9.0", "pandas==1.5.3", "cloudpickle==2.2.1",
            "numpy==1.23.5", "category-encoders==2.6.3", "xgboost==2.0.3",
            "lz4==4.3.2", "typing-extensions==4.10.0"
        ]

        # Log model and metrics
        logger.info("Logging model and metrics...")
        run_id, model_uri = log_model_and_metrics(
                                                    model=pipeline_model, 
                                                    X_train=X_train, 
                                                    y_train=y_train, 
                                                    X_test=X_test, 
                                                    y_test=y_test,
                                                    training_set=training_set, 
                                                    hyperparams=hyperparams, 
                                                    pip_requirements=pip_requirements, 
                                                    features_types=features_types,
                                                    run_id=run_id  # Use the retrieved run_id
                                                )


        dbutils.jobs.taskValues.set("run_id", run_id)
        dbutils.jobs.taskValues.set("model_name", MODELS_NAME['MODEL2'])
        dbutils.jobs.taskValues.set("model_uri", model_uri)

        logger.info("Model training and logging completed successfully.")
        logger.info(f"run_id received {run_id}")
        logger.info(f"model_name received {MODELS_NAME['MODEL2']}")
        logger.info(f"model_uri received {model_uri}")

    except Exception as e:
        error_message = f"An error occurred in main execution: {str(e)}\nTraceback: {traceback.format_exc()}"
        logger.error(error_message)
        raise

    logger.info("Script execution completed.")


train_and_register_model(run_id=RUN_ID)
