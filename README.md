# MachineLearning-projects
Projects carried out by modifying and using datasets on Kaggle or other public dataset platforms such as Google Dataset Search or Microsoft Azure.

#### 1. Logistic Regression project:
            Modified "cardio_train.csv" dataset on Kaggle and added some new features (x1, x2, x3 ...) to 
            improve the predictivity of the label (y),i.e. the presence or not of cardiovascular disease:
            
            - MEAN ARTERIAL PRESSURE
            - HEART RATE
            - SYSTOLIC VOLUME
            - BODY MASS INDEX
            
            Using the following features brings the test accuracy from 0.70 to 0.736.

#### 2. Decision Tree Regression project for Time Series
            Forecasting analysis AMAZON stock prize using "AMZN_data.csv" dataset on Kaggle.
            
            The analysis uses DecisionTreeRegression model for predict the stocks price for the next 25 days.
            In this first experience with Time Series data, I didn't use cross-validation, which would have
            allowed me to get an idea of the accuracy level of the higher analysis.
            
            In any case, I decided to carry out a manual cross-validation that allowed me to become more 
            familiar with the theoretical concept of Cross-Validation.
            
            The result with cross_validation=5 is R2_SCORE = 0.82
