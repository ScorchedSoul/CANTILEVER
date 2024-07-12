# MOVIE_REVIEW_SENTIMENT_ANALYSIS
SENTIMENT ANALYSIS : MOVIE_REVIEW


## Table of Contents
1. Project Overview
2. Dataset
3. Sample Dataset Used
4. Solution Architecture
5. Model Details
6. Installation and Setup
7. Usage
8. Results
9. Visualization
10. Conclusion
11. Acknowledgements

## Introduction
- This project focuses on sentiment analysis of movie reviews using various machine learning algorithms.
- The goal is to classify reviews as positive or negative based on their content.

## Project Overview
- The sentiment analysis system uses natural language processing techniques and machine learning models to predict the sentiment of IMDB movie reviews.
- The project includes data preprocessing, model training, evaluation, and visualization of results.

## Dataset
- The dataset used in this project is the IMDB dataset, which contains 50,000 movie reviews labeled as positive or negative.
- It is widely used for binary sentiment classification tasks.

- For more information, visit: [IMDB Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)

## Sample Dataset Used
- Due to the large size of the original dataset, a sample of 10,000 reviews was used for this project to ensure efficient processing and training.

## Solution Architecture

1. Data Preprocessing:
   - Remove HTML tags
   - Remove non-alphabetic characters
   - Convert text to lowercase
   - Tokenize text
   - Remove stopwords
   - Lemmatize words
   - Handle negations

2. Model Training:
   - Decision Tree
   - Logistic Regression
   - Random Forest
   - Support Vector Classifier (SVC)
   - Recurrent Neural Network (RNN)

3. Evaluation:
   - Accuracy
   - Precision
   - Recall
   - F1-Score

4. Visualization:
   - Sentiment distribution
   - Word clouds for positive and negative reviews
   - Review length distribution

## Model Details

- Decision Tree: Basic classifier with low performance
- Logistic Regression: High accuracy and balanced performance
- Random Forest: Better than Decision Tree but less effective than SVC
- Support Vector Classifier (SVC): High accuracy, similar to Logistic Regression
- Recurrent Neural Network (RNN): Lowest accuracy among tested models

## Installation and Setup

Install the required libraries:

- numpy: `pip install numpy`
- pandas: `pip install pandas`
- matplotlib: `pip install matplotlib`
- seaborn: `pip install seaborn`
- scikit-learn: `pip install scikit-learn`
- wordcloud: `pip install wordcloud`
- beautifulsoup4: `pip install beautifulsoup4`
- nltk: `pip install nltk`
- joblib: `pip install joblib`
- keras: `pip install keras`

## Usage

1. Preprocess the dataset using the provided cleaning functions.
2. Train the models on the preprocessed data.
3. Evaluate the models using accuracy, precision, recall, and F1-score.
4. Visualize the results using various plots and word clouds.
5. Predict the sentiment of new reviews using the trained SVC model.

## Results

- Decision Tree: Accuracy = 0.694
- Logistic Regression: Accuracy = 0.867
- Random Forest: Accuracy = 0.745
- Support Vector Classifier (SVC): Accuracy = 0.869
- Recurrent Neural Network (RNN): Accuracy = 0.5017

## Visualization

- Distribution of positive and negative sentiments
- Word clouds for positive and negative reviews
- Review length distribution by sentiment
- Outlier detection in review lengths

## Conclusion

- Both Support Vector Classifier (SVC) and Logistic Regression show high accuracy (0.869) with balanced precision, recall, and F1-scores. They perform similarly well and are recommended if computational cost is not a major concern.
- Decision Tree has the lowest performance among the models tested.
- Random Forest shows better performance than the Decision Tree but is outperformed by SVC and Logistic Regression.
- The RNN model has the lowest accuracy (0.5017), indicating it is not performing well for this task.
- Choosing between Logistic Regression and SVC would be appropriate based on their similar high performance.
- SVC is chosen for this project due to its consistent high accuracy and balanced metrics.

### Choosing between Logistic Regression and SVC would be appropriate based on their similar high performance. For future work, hyperparameter tuning and the use of more advanced models like Transformer-based architectures could further improve the system's performance. Additionally, addressing class imbalance through techniques such as SMOTE or class-weight adjustment might enhance model robustness.

## Acknowledgements

- The IMDB dataset is provided by Stanford AI Lab.
- Thanks to the developers of the libraries used in this project.
- Special thanks to the open-source community for their valuable contributions.
