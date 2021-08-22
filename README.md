# Back-Order-Prediction-Intership-Project
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Problem Statement 
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Backorders are unavoidable, but by anticipating which things will be backordered, planning can be streamlined at several levels, preventing unexpected strain on production, logistics, and transportation. ERP systems generate a lot of data (mainly structured) and also contain a lot of historical data; if this data can be properly utilized, a predictive model to forecast backorders and plan accordingly can be constructed. Based on past data from inventories, supply chain, and sales, classify the products as going into backorder (Yes or No).

# Approach
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

The main goal is to predict the fares of the flights based on different factors available in the dataset.

* Data Exploration     : Exploring dataset using pandas,numpy,matplotlib and seaborn. 
* Data visualization   : Ploted graphs to get insights about dependend and independed variables. 
* Feature Engineering  :  Removed missing values and created new features as per insights.
* Model Selection I    :  1. Tested all base models to check the base accuracy.
                          2. Also ploted AUC curve to check whether a model is a good fit or not and confusion matrix etc.
* Model Selection II   :  Performed Hyperparameter tuning using gridsearchCV.
* Pickle File          :  Selected model as per best accuracy and created pickle file using pickle library.
* Webpage & deployment :  Created a webform that takes all the necessary inputs from user and shows output.
                          After that I have deployed project on AWS and Microsoft Azure
                          
# Technologies Used
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

*	Pycharm Is Used For IDE.
*	For Visualization Of The Plots Matplotlib , Seaborn Are Used.
*	AWS , Azure Are Used For Model Deployment.
*	Cassandra Database Is Used To As Data Base.
*	Front End Deployment Is Done Using HTML , CSS.
*	Python Flask Is Used For Backend Deployment.
*	Git Hub Is Used As A Version Control System.
*	Pandas And Numpy Is Used For Data Exploration.

# Conclusions 
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

We Developed A Product Backorder Predictive Model With The Capability Of Identifying Items To Be Backordered Using Machine Learning Models. The Proposed Approach Accept Input Data Was Pre-Processed By Way Of Missing Values Imputation, Non-Numeric To Numeric Feature Conversion And Normalization, And Split Into Training And Test Set. The Training Set Is Passed Into A Data Balancing Module To Ensure Equal Class Distribution And Avoid Biasness In Learning Model Decisions. The Imbalanced Training Data Were Subjected Sampling As We Concurrently Fed The Data Into Sampling Techniques Fed Into ML Models To Predict Product Backorders. The Predictive Models Were Validated On Test Data And Their Performances Were Evaluated. The Evaluation Of The Result Obtained Showed By Precision, Recall And F1-Score.
