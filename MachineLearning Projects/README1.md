### Machine Learning Projects

#### 1. Machine Learning project - Logistic Regression:
            Modified "cardio_train.csv" dataset on Kaggle and added some new features (x1, x2, x3 ...) 
            to improve the predictivity of the label (y),i.e. the presence or not of cardiovascular 
            disease:
            
            - MEAN ARTERIAL PRESSURE
            - HEART RATE
            - SYSTOLIC VOLUME
            - BODY MASS INDEX
            
            Using the following features brings the test accuracy 
            from 0.70 to 0.73 with Cross-Validation=5.
            

#### 2. Machine Learning project - Decision Tree Regression for Time Series:
            Forecasting analysis AMAZON stock prize using "AMZN_data.csv" dataset on Kaggle.
            
            The analysis uses DecisionTreeRegression model for predict the stocks price for the 
            next 25 days. In this first experience with Time Series data, I didn't use cross-validation, 
            which would have allowed me to get an idea of the accuracy level of the higher analysis.
            
            In any case, I decided to carry out a manual cross-validation that allowed me to become 
            more familiar with the theoretical concept of Cross-Validation.
            
            The result with cross_validation=5 is R2_SCORE = 0.8223
            

#### 6. Machine Learning project (Healthcare) - XGB classification:
           This example of project is based on a particular type of machine learning classification (XGB).
           Modified "heart_failure_clinical_records_dataset" dataset on Kaggle and added and modify some new 
           features (x1, x2, x3 ...) to improve the predictivity of the label (y),i.e. the presence or not of 
           cardiovascular disease.
           The initial weights of the dataset characteristics were quite low.
![features non numeriche](https://user-images.githubusercontent.com/78934727/137938543-45fb71ab-99c3-46f8-a22f-0c4704af3c22.png)
![peso features](https://user-images.githubusercontent.com/78934727/137938556-c7ecf5d5-4b76-4085-98a7-81a9323cbf70.png)
           
           Specifically, I added two characteristics "troponin" and "lactate" by taking the statistical 
           distribution of how many people with a heart attack have out-of-range troponin and lactate values.
           In particular, by calculating the correlation matrix it is possible to note how these two new
           characteristics are particularly predictive of the phenomenon we are studying. Even in scientific
           literature, these two values are highly indicative of heart failure.
![correlation matrix](https://user-images.githubusercontent.com/78934727/137938165-ca549fe7-4a3b-4389-912a-6f0f61905f43.png)
           
           The model used is Gradient boosting technique (regression and statistical classification algorithm 
           that produce a predictive model, typically decision trees).
           
##### The result is a validation accuracy with cross_validation of: 86,4%, and in particular
##### the model train_logloss:0.26906 and validation_logloss:0.31355
![log_loss](https://user-images.githubusercontent.com/78934727/137937571-e86a981f-1300-4b19-95b0-656b13927bb2.png)



