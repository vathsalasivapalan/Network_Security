from datetime import datetime
import os 
from networksecurity.constant import training_pipeline

print(training_pipeline.ARTIFACT_DIR)

print(training_pipeline.PIPELINE_NAME)


class TrainingPipelineConfig:
    def __init__(self,timestamp=datetime.now()):
        timestamp =timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name=training_pipeline.PIPELINE_NAME
      
        

class DataIngestionConfig:
    def __init__(self):
        pass 
    
class DataValidationConfig:
    def __init__(self):
        pass

class DataTransformationConfig:
    def __init__(self):
        pass

class ModelTrainerConfig:
    def __init__(self):
        pass
    
class ModelEvaluationConfig:
    def __init__(self):
        pass

class ModelPusherConfig:
    def __init__(self):
        pass