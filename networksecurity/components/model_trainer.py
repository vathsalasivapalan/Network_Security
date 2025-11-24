import os
import sys

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logger.logger import logging

from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object, load_object, load_numpy_array_data
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score


class ModelTrainer:

    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    # ✅ Method properly indented inside the class
    def perform_hyper_parameter_tunig(self, x_train, y_train):
        try:
            logging.info("Starting hyperparameter tuning for XGBoost...")

            xgb_clf = XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False
            )

            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.7, 0.8, 1.0],
                'colsample_bytree': [0.7, 0.8, 1.0],
                'gamma': [0, 0.1, 0.3]
            }

            grid_search = GridSearchCV(
                estimator=xgb_clf,
                param_grid=param_grid,
                scoring='f1',
                cv=3,
                verbose=2,
                n_jobs=-1
            )

            grid_search.fit(x_train, y_train)
            best_params = grid_search.best_params_
            logging.info(f"Best Parameters found for XGBoost: {best_params}")

            best_model = XGBClassifier(
                **best_params,
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False
            )
            best_model.fit(x_train, y_train)
            logging.info("Best XGBoost model trained successfully.")
            return best_model

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def train_model(self, x_train, y_train):
        try:
            xgb_clf = XGBClassifier()
            xgb_clf.fit(x_train, y_train)
            return xgb_clf
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            # ✅ Use hyperparameter tuning
            model = self.perform_hyper_parameter_tunig(x_train, y_train)


            y_train_pred = model.predict(x_train)
            classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)

            if classification_train_metric.f1_score <= self.model_trainer_config.expected_accuracy:
                logging.warning("Trained model did not meet the expected accuracy threshold.")

            y_test_pred = model.predict(x_test)
            classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

            diff = abs(classification_train_metric.f1_score - classification_test_metric.f1_score)
            if diff > self.model_trainer_config.overfitting_underfitting_threshold:
                raise Exception("Model is overfitting or underfitting. Try more experimentation.")

            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)

            network_model = NetworkModel(preprocessor=preprocessor, model=model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=network_model)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric
            )

            logging.info(f"Model trainer artifact created successfully: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
