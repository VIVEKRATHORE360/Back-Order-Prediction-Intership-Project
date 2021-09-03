import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from imblearn.under_sampling import NearMiss

class Preprocessor:
    """
        This class shall  be used to clean and transform the data before training.
        Written By: Vikas

    """
    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object

    def remove_columns(self,data,columns):
        """
                Method Name: remove_columns
                Description: This method removes the given columns from a pandas dataframe.
                Output: A pandas DataFrame after removing the specified columns.
                On Failure: Raise Exception
                Written By: Vikas
        """

        self.logger_object.log(self.file_object, 'Entered the remove_columns method of the Preprocessor class')
        self.data=data
        self.columns=columns

        try:
            self.useful_data=self.data.drop(columns = [self.columns])
            # drop the labels specified in the columns
            self.logger_object.log(self.file_object,'Column removal Successful.Exited the remove_columns method of the Preprocessor class')
            return self.useful_data

        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in remove_columns method of the Preprocessor class. Exception message:  '+str(e))
            self.logger_object.log(self.file_object,'Column removal Unsuccessful. Exited the remove_columns method of the Preprocessor class')
            raise Exception()

    def separate_label_feature(self, data, label_column_name):
        """
            Method Name: separate_label_feature
            Description: This method separates the features and a Label Coulmns.
            Output: Returns two separate Dataframes, one containing features and the other containing Labels .
            On Failure: Raise Exception
            Written By: Vikas

        """
        self.logger_object.log(self.file_object, 'Entered the separate_label_feature method of the Preprocessor class')
        self.data = data
        self.column = label_column_name

        try:
            self.X=self.data.drop(columns = [self.column]) # drop the columns specified and separate the feature columns
            self.Y=self.data[self.column] # Filter the Label columns
            self.logger_object.log(self.file_object,'Label Separation Successful. Exited the separate_label_feature method of the Preprocessor class')
            return self.X,self.Y

        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in separate_label_feature method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'Label Separation Unsuccessful. Exited the separate_label_feature method of the Preprocessor class')
            raise Exception()

    def is_null_present(self,data):
        """
            Method Name: is_null_present
            Description: This method checks whether there are null values present in the pandas Dataframe or not.
            Output: Returns True if null values are present in the DataFrame, False if they are not present and
                    returns the list of columns for which null values are present.
            On Failure: Raise Exception
            Written By: Vikas

        """

        self.logger_object.log(self.file_object, 'Entered the is_null_present method of the Preprocessor class')
        self.null_present = False
        self.cols_with_missing_values=[]
        self.data = data
        self.cols = self.data.columns
        try:
            self.null_counts=self.data.isnall().sum() # check for the count of null values per column
            for i in range(len(self.null_counts)):
                if self.null_counts[i]>0:
                    self.null_present=True
                    self.cols_with_missing_values.append(self.cols[i])

            if(self.null_present): # write the logs to see which columns have null values
                self.dataframe_with_null = pd.DataFrame()
                self.dataframe_with_null['columns'] = self.cols
                self.dataframe_with_null['missing values count'] = np.asarray(self.data.isnall().sum())
                self.dataframe_with_null.to_csv('Data_Information/null_values.csv')
                # storing the null column information to file

            self.logger_object.log(self.file_object,'Finding missing values is a success.Data written to the null values file. Exited the is_null_present method of the Preprocessor class')
            return self.null_present, self.cols_with_missing_values

        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in is_null_present method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'Finding missing values failed. Exited the is_null_present method of the Preprocessor class')
            raise Exception()

    def impute_missing_values(self, data, cols_with_missing_values):
        """
            Method Name: impute_missing_values
            Description: This method replaces all the missing values in the Dataframe using Frequency Category Imputation Method.
            Output: A Dataframe which has all the missing values imputed.
            On Failure: Raise Exception
            Written By: Vikas
        """

        self.logger_object.log(self.file_object, 'Entered the impute_missing_values method of the Preprocessor class')
        self.data = data
        self.cols_with_missing_values=cols_with_missing_values
        try:
            ### Handle Nan Value with Median Imputation Method
            def Impute_nan(data, feature):
                frequent_feature = data[feature].median()
                data[feature].fillna(frequent_feature, inplace=True)

            for col in self.cols_with_missing_values:
                Impute_nan(self.data,col)

            self.logger_object.log(self.file_object, 'Imputing missing values Successful. Exited the impute_missing_values method of the Preprocessor class')
            return self.data

        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in impute_missing_values method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'Imputing missing values failed. Exited the impute_missing_values method of the Preprocessor class')
            raise Exception()

    def scale_numerical_columns(self,data):
        """
            Method Name: scale_numerical_columns
            Description: This method scales the numerical values using the Robust Scaler.
            Output: A dataframe with scaled
            On Failure: Raise Exception
            Written By: Vikas
        """

        self.logger_object.log(self.file_object,'Entered the scale_numerical_columns method of the Preprocessor class')
        self.data=data

        try:
            self.num_df = self.data.select_dtypes(include=['int64']).copy()
            self.scaler = RobustScaler()
            self.scaled_data = self.scaler.fit_transform(self.num_df)
            self.scaled_num_df = pd.DataFrame(data=self.scaled_data, columns=self.num_df.columns)
            self.logger_object.log(self.file_object, 'scaling for numerical values successful. Exited the scale_numerical_columns method of the Preprocessor class')
            return self.scaled_num_df

        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in scale_numerical_columns method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'scaling for numerical columns Failed. Exited the scale_numerical_columns method of the Preprocessor class')
            raise Exception()

    def encode_categorical_columns(self,data):
        """
            Method Name: encode_categorical_columns
            Description: This method encodes the categorical values to numeric values.
            Output: Data set In Numeric Value Column.
            On Failure: Raise Exception
            Written By: Vikas
        """

        self.logger_object.log(self.file_object, 'Entered the encode_categorical_columns method of the Preprocessor class')
        self.data = data
        try:

            self.cat_df = self.data.select_dtypes(include=['object']).copy()

            for x in self.cat_df.columns:
                if x != 'sku':
                    self.data[x].replace({'Yes': 1, 'No': 0}, inplace=True)
                    self.data[x].astype(int)

            self.logger_object.log(self.file_object, 'encoding for categorical values successful. Exited the encode_categorical_columns method of the Preprocessor class')
            return self.data

        except Exception as e:

            self.logger_object.log(self.file_object,'Exception occured in encode_categorical_columns method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'encoding for categorical columns Failed. Exited the encode_categorical_columns method of the Preprocessor class')
            raise Exception()

    def handle_imbalanced_dataset(self,x,y):
        """
            Method Name: handle_imbalanced_dataset
            Description: This method handles the imbalanced dataset to make it a balanced one.
            Output: new balanced feature and target columns
            On Failure: Raise Exception
            Written By: Vikas
        """

        self.logger_object.log(self.file_object,'Entered the handle_imbalanced_dataset method of the Preprocessor class')

        try:
            self.nmsample = NearMiss()
            self.x_sampled,self.y_sampled  = self.nmsample.fit_sample(x,y)
            self.logger_object.log(self.file_object,'dataset balancing successful. Exited the handle_imbalanced_dataset method of the Preprocessor class')
            return self.x_sampled,self.y_sampled

        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in handle_imbalanced_dataset method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'dataset balancing Failed. Exited the handle_imbalanced_dataset method of the Preprocessor class')
            raise Exception()
