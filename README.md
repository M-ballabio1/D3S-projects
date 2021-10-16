# DataScience_MachineLearning_DeepLearning-projects
Projects carried out by modifying and using datasets on Kaggle or other public dataset platforms such as Google Dataset Search or Microsoft Azure.

#### 1. Logistic Regression project:
            Modified "cardio_train.csv" dataset on Kaggle and added some new features (x1, x2, x3 ...) to 
            improve the predictivity of the label (y),i.e. the presence or not of cardiovascular disease:
            
            - MEAN ARTERIAL PRESSURE
            - HEART RATE
            - SYSTOLIC VOLUME
            - BODY MASS INDEX
            
            Using the following features brings the test accuracy from 0.70 to 0.73 with Cross-Validation=5.
            

#### 2. Decision Tree Regression project for Time Series
            Forecasting analysis AMAZON stock prize using "AMZN_data.csv" dataset on Kaggle.
            
            The analysis uses DecisionTreeRegression model for predict the stocks price for the next 25 days.
            In this first experience with Time Series data, I didn't use cross-validation, which would have
            allowed me to get an idea of the accuracy level of the higher analysis.
            
            In any case, I decided to carry out a manual cross-validation that allowed me to become more 
            familiar with the theoretical concept of Cross-Validation.
            
            The result with cross_validation=5 is R2_SCORE = 0.8223
            
#### 3. Data Science project for Optimization Portfolio
            Optimization portfolio based on 10 equity shares. In this project, the dataset was 
            taken from yahoo finance.
            
            The analysis evaluates the trend in the value of the shares of 10
            companies from 2015 to today. 
![Immagine1](https://user-images.githubusercontent.com/78934727/136968720-79082c29-15cc-4c7b-90a7-237289e102b0.png)
            
            The analysis starts with the inizialization for all 10 equity shares of uniform 
            weights in my portfolio (10%). After that, I evalute Expected annual return and Annual volatility. 
 
            At this point, I perform a portfolio optimization, starting from the 10 chosen actions,
            by defining the Efficient Frontier. We have obtained that the performance of the 
            portfolio with OPTIMIZED distribution, maximizes the return and minimizes the financial risk.
            
##### Uniform Wallet:
Expected Annual Return: 27%
Financial Risk: 21%

##### Optimized Wallet:
Expected Annual Return: 59,4%
Financial Risk: 33.3%
            
![portfolio_uniforme](https://user-images.githubusercontent.com/78934727/137339497-de1b8b80-6533-4513-a7ad-515abbd7642c.png)
![portfolio_ottimizzato](https://user-images.githubusercontent.com/78934727/137340865-dbbdadac-d19c-49f0-98fe-0980a931210f.png)
![comparison_portfolio](https://user-images.githubusercontent.com/78934727/137339586-98b14e78-ae45-4043-adfe-76026a5b61f0.PNG)


#### 4. Data Science project - Markov Model in Healthcare case
            The purpose of the Markov Model is to model the situation of an individual, in which time
            plays a strategic role. Indeed, the timeline of events is important and each
            event can happen more than once
            
            The analysis consists of the following situation: a health company must evaluate the introduction 
            of a medical device for the improvement of osteoarthritis. Some information is known:
            - device cost
            - probability of osteoporosis fractures by age group
            - mortality by age group
            - cost of treatment following the fracture.
            - cost of deaths.

            What we want to evaluate is the ICER. The incremental cost-effectiveness ratio (ICER)
            is a statistic used in cost-effectiveness analysis to summarise the cost-effectiveness of 
            a health care intervention. It is defined by the difference in cost between  two possible
            interventions, divided by the difference in their effect. It represents the average incremental 
            cost associated with 1 additional unit of the measure of effect.
            
            The result is positive for all age groups, in fact, all DeltaCost and DeltaLY couples are 
            greater than the red line that delimits the cost-life years convenience.
            
![LY_COSTI2](https://user-images.githubusercontent.com/78934727/137211357-5b3c4f24-320a-4fd4-b533-064617214702.png)
![DESCRIZ](https://user-images.githubusercontent.com/78934727/137210698-986a96f9-97cb-4ff0-872c-dc09f8fcf99d.PNG)

#### 4. Deep Learning project (Computer Vision) - CNN for classification mushrooms
           This is my first experiment with a neural network and certainly there will be a lot to work on,
           but this is my first job.
           I have created a CNN network for the classification of mushrooms in 5 possible categories. 
           Subsequently, with a graphical interface called Gradio I was able to verify the ability to 
           generalize on images different from those of the training dataset. The result is the following:
           
##### Accuracy: 0.8335 - val_accuracy: 0.7791

![ACCURACY](https://user-images.githubusercontent.com/78934727/137590545-a9177cf4-a872-4ba7-9fc1-9370d90f8daf.PNG)

           The result reveals that the model has a slight overfitting problem, as the accuracy on the 
           training set > accuracy on the validation set. However, this being the first test I can be quite 
           satisfied to have obtained a val_accuracy of 77.91%.
![giallino](https://user-images.githubusercontent.com/78934727/137590800-1d5c7a9f-b977-4a21-9edd-efb171d4fa0b.png)
![porcino](https://user-images.githubusercontent.com/78934727/137590694-fc57ef1f-2b75-431a-9a49-befb52165b61.png)

           



