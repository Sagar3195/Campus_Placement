import os 
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from dataclasses import dataclass
from src.utils import save_object 


@dataclass 
class DataTransformationConfig:
    preprocessor_obj_file_path= os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config= DataTransformationConfig()
    
    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:

            categorical_columns = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex','specialisation']

            numerical_columns = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']

            num_pipeline= Pipeline(
                steps= [
                    ('scaler',  StandardScaler())
                ]
            )

            cat_pipeline= Pipeline(
                steps= [
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean= False))
                ]
            )

            logging.info(f"Categorical Columns: {categorical_columns}")
            
            logging.info(f"Numerical Columns: {numerical_columns}")

            preprocessor= ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )


            return preprocessor

        except Exception as e:
            raise CustomException(e, sys) 

    def initiate_data_transformation(self, train_path, test_path):
        
        try:
            train_df= pd.read_csv(train_path)
            test_df= pd.read_csv(test_path)

            logging.info("Read train and test data completed.")

            logging.info("Obtaining preprocessing object.")

            preprocessing_obj= self.get_data_transformer_object() 

            target_column_name= 'status'

            #Encoding Target Column 
            logging.info("Encoding Target Variable")
            train_df[target_column_name] = train_df[target_column_name].replace({'Placed':0, 'Not Placed':1}) 
            
            test_df[target_column_name] = test_df[target_column_name].replace({'Placed':0, 'Not Placed':1})

            #Dropping salary column from datasets because if we replace missing value with 0 value and students didn't get placements, 
            #it would be bad idea, as it would mean student get placements if he earns salary.
            #Also we will drop 'sl_no' column from dataset which none of use for the model.
             
            drop_column= 'sl_no'
            
            input_feature_train_df= train_df.drop(columns= [target_column_name, drop_column], axis = 1)
            target_feature_train_df= train_df[target_column_name]

            input_feature_test_df= test_df.drop(columns= [target_column_name, drop_column], axis = 1)
            target_feature_test_df= test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr= preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr= preprocessing_obj.transform(input_feature_test_df)


            train_arr= np.c_[input_feature_train_arr, np.array(target_feature_train_df)]

            test_arr= np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            # print("Train array:\n", train_arr)
            # print()
            # print("Test array:\n", test_arr)

            logging.info("Save Preprocessing object.")

            save_object(
                file_path= self.data_transformation_config.preprocessor_obj_file_path, 
                obj= preprocessing_obj
            )

            return (
                train_arr, 
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,

            )

        except Exception as e:
            raise CustomException(e, sys)

