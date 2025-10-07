from networksecurity.constant.training_pipeline import SCHEMA_FILE_PATH
from networksecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.exception.exception import NetworkSecurityException 
from networksecurity.logger.logger import logging 
from networksecurity.utils.main_utils.utils import read_yaml_file,write_yaml_file
from scipy.stats import ks_2samp
import pandas as pd
import os,sys


class DataValidation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifact=data_ingestion_artifact
            self.initiate_data_validation_config=data_validation_config
            self._schema_config=read_yaml_file(SCHEMA_FILE_PATH)
        
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    
        
    def validate_number_of_columns(self,dataframe:pd.DataFrame)->bool:
        try:
            number_of_columns = len(self._schema_config["columns"])
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Data frame has columns: {len(dataframe.columns)}")
            
            if len(dataframe.columns)==number_of_columns:
                return True
            return False
        
        
        except Exception as e:
            raise NetworkSecurityException(e,sys)
            
        
        
    def is_numerical_column_exist(self,dataframe:pd.DataFrame)->bool:
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    @staticmethod 
    def read_data(self,file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def detect_data_drift(self,base_df,current_df,threshold=0.05)->bool:
        try:
            pass 
        except Exception as e:
            raise NetworkSecurityException(e,sys)    
        
    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            self.read_data()
            self.validate_number_of_columns()
            self.detect_data_drift()
            self.is_numerical_column_exist()
        except Exception as e:
            raise NetworkSecurityException(e,sys)   