# Surabhi_Portfolio


# [Project1: Employee Retention Prediction Project](https://github.com/Surabhi-1996/Human_Resources)

## With collected extensive data of employees, we are going to develop a model that could predict which employees are most likey to quit.

- Hiring and retaining employees are extreemly complex tasks that requires capital, time and skills.
- The HR team have collected extensive data on their employees and approched to develop a model that could predict which employees are most likely to quit.
- Explore and visualize dataset
- Data cleaning to remove the unwanted features
- Training and evaluating models using below algorithms:
  - Logistic Regression 
  - RandomForestClassifier
  - Deep Learning model using Keras


# [Project2: Movie Review using Text Classification Method](https://github.com/Surabhi-1996/movie_review1)

## In this project we'll try to develop a classification model - that is, we'll try to predict the Positive/Negative labels based on text content alone for movie review dataset.

- The dataset contains the text of 2000 movie reviews. 1000 are positive, 1000 are negative, and the text.
- Data cleaning to remove missing values and empty strings.
- Build pipelines to vectorize the data, then train and fit a model
- Feed the training data through the pipeline of below models
  - Naive Bayes
  - Linear SVC
-Run predictions and analyze the results 


# [Project3: Classifying disease using Deep Learning](https://github.com/Surabhi-1996/disease_xray_classification)

## We will automate the process of detecting and classifying chest diseases. 

- Deep learning proven to be superior in detecting and classifying disease using imagery data.The project is about to automate the process of detecting and classifying chest disease and reduce the cost and time of detection. Dataset contains 133 images that belong to 4 classes.
  - Healthy
  - Covid-19
  - Bacterial Pneumonia
  - Viral Pneumonia
- Use image generator to generate tensor images data and normalize them, split the data into training and validation.
- Visualize the dataset using matplotlib
- Import the model ResNet of pretrained weights
- Build and train deep learning model using Keras
- Use ModelCheckpoint to save the best model
- Evaluate trained deep learning model


# [Project4: Topic Modeling using Latent Dirichlet Allocation and Non-negative Matrix Factorization](https://github.com/Surabhi-1996/topic_modeling)

## We are using npr dataset, which contains comprehensive identity database for every usual resident in the country.

- Import the dataset using pandas and conduct data preprocessing.
- Use TF-IDF Vectorization to create a vectorized document term matrix. 
- Using Scikit-Learn create an instance of LDA and NMF with 7 expected components.
- Print our the top 15 most common words for each of the 7 topics.
- Assign the rows with maximum probability of relation with topic
- Using LDA and NMF, we will classify the Article data into below 7 topics:
  - health
  - election
  - legis
  - politics
  - election
  - music
  - edu


# [Project5: Russian License Plate Blurring](https://github.com/Surabhi-1996/object_detection)

## We are using Cascade Classifier(Haar Cascade) to detect the object in an image.

- Our goal will be to use Haar Cascades to blur license plates detected in an image!
- Read in the car_plate.jpg file using OpenCV.
- Visualize the image using matplotlib.
- Load the haarcascade_russian_plate_number.xml file.
- Create a function that takes in an image and draws a rectangle around what it detects to be a license plate.
- Create a blurred version of the ROI (the license plate) we will want to paste this blurred image back on to the original image at the same original location.


# [Project6: Text Generation using LSTM](https://github.com/Surabhi-1996/text_generation)

## Moby-Dick is a novel by American writer Harman Melville. We have used Moby-Dick Chapter-4 dataset for training purpose.

We are using Keras,spacy,LSTM and Embedding layers in this project.

- Reading in file as a string text
- Tokenize and clean the text using spacy
- Organize into sequences of tokens (25 trained words and one target word)
- Encode sequence of words using Keras
- Create a model using Embedding, LSTM and Dense layers
- Function to generate new text


# [Project7: Movie Review using Sentiment Analysis](https://github.com/Surabhi-1996/movie_review2)

## With movie reviews dataset, we are going to predict the labels(positive or negative) using sentiment analysis method.

- For this project, we'll perform the NLTK VADER sentiment analysis on movie reviews dataset.
- Data cleaning to remove missing values and empty strings.
- Import SentimentIntensityAnalyzer and create an sid object
- Find the polarity and use sid to append a compound score to the dataset
- Perform a comparison analysis between the original label and compound score
