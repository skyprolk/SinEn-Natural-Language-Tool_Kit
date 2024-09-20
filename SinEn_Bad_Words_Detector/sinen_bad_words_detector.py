# Import necessary libraries
import os
import sys
import joblib
import pickle
import nltk
import absl.logging
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from art import *

# Download the punkt tokenizer for sentence tokenization
nltk.download('punkt')

# Set logging level to reduce amount of log messages displayed
absl.logging.set_verbosity(absl.logging.ERROR)

# Set environment variable to disable TensorFlow log messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Define project name for tracking experiments in MLFlow
project_name = "sinen_bad_words_detector"

# Directory path for the bad word list
bad_words_list_dir = "../data/bad_word_list_singlish.csv"

# Set n-gram range for CountVectorizer
ngram_range = (2, 4)

# Get the name of the directory where this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Get the parent directory name where the current directory is present.
parent = os.path.dirname(current)

# Add the parent directory to the sys.path.
sys.path.append(parent)

# now we can import the module in the parent
# directory.
from option_menu import option_menu

# Get the path of the SinEn_Stemmer directory.
stemmer_dir = os.path.join(parent, 'SinEn_Stemmer')

# Add the SinEn_Stemmer directory to the system path.
sys.path.append(stemmer_dir)

# Get the path of the SinEn_Tokenizer directory.
tokenizer_dir = os.path.join(parent, 'SinEn_Tokenizer')

# Add the SinEn_Tokenizer directory to the system path.
sys.path.append(tokenizer_dir)

# Import the SinEnStemmer class from the SinEn_Stemmer module
from sinen_stemmer import SinEnStemmer

# Import the SinEnTokenizer class from the SinEn_Tokenizer module
from sinen_tokenizer import SinEnTokenizer

# Define function for loading datasets from file
def load_datasets(directory:str):
    print("### Loading data from file...")
    try:
        # Load data from file
        data = pd.read_csv(directory)

        # Print information about the datasets
        print("### Datasets have been loaded successfully!")
        labels_name_list = []
        labels = []
        for label in data['label']:
            label = "".join(str(label).split())
            if label not in labels_name_list:
                labels_name_list.append(label)
            labels.append(label)
        datasets_size = ""
        for label in labels_name_list:
            datasets_size += str(f"[{label} = {labels.count(label)}] ")
        print(f">> Labels Count: {len(labels_name_list)} \n>> Labels Names: {labels_name_list} \n>> Datasets Size: {datasets_size}")

        # Return only the 'sentence' column of the dataset
        data = data['sentence']
        return data, labels
    except FileNotFoundError:
        print(f"Error: File '{directory}' not found.")
    except Exception as e:
        print(f"Error: {e}")
        
# Define function for cleaning texts
def clean_texts(texts:list):
    
    # Initialize an empty list to store cleaned texts
    cleaned_texts = []
    
    # Instantiate a stemmer and tokenizer object
    stemmer = SinEnStemmer()
    tokenizer = SinEnTokenizer()

    # Iterate through each text in the input list
    for text in texts:
        # Tokenize the text using the tokenizer object
        tokenized_text = tokenizer.tokenize(text)

        # Initialize an empty list to store filtered words
        filtered_sentence = []

        # Iterate through each word in the tokenized text
        for word in tokenized_text:
            # Stem the word using the stemmer object and append the first stem to the filtered sentence list
            filtered_sentence.append(stemmer.stem(word)[0])

        # Join the filtered sentence list into a string and append it to the cleaned_texts list
        cleaned_texts.append(" ".join(filtered_sentence))

    # Return the cleaned_texts list
    return cleaned_texts

# Define a function to preprocess the data
def preprocess_data(data: list, labels: list):
    # Display a message indicating that the data is being processed
    print("### Processing data...")
    
    # Create a CountVectorizer object to convert text into a matrix of token counts
    vectorizer = CountVectorizer(analyzer='char', ngram_range=ngram_range)
    
    # Transform the data into a matrix of token counts
    data = vectorizer.fit_transform(data)
    
    # Create a LabelEncoder object to encode the labels
    label_encoder = LabelEncoder()
    
    # Encode the labels
    labels = label_encoder.fit_transform(labels)
    
    # Return the preprocessed data, vectorizer, labels, and label encoder
    return data, vectorizer, labels, label_encoder

# Define a function to train the model
def train_model(X_train: list, y_train: list):
    # Display a message indicating that the model is being trained
    print("### Training model...")
    
    # Create an SVM object with a linear kernel, regularization parameter C=1, and gamma=scale
    svc = svm.SVC(kernel='linear', C=1, gamma='scale')
    
    # Train the model using the training data
    svc.fit(X_train, y_train)
    
    # Display a message indicating that the model training is complete
    print("### Model training complete!")
    
    # Return the trained model
    return svc

# Prints the details of incorrect predictions made by a model on test data
def get_incorrect_predictions(predictions: list, X_test: list, y_test: list, vectorizer: CountVectorizer, label_encoder: LabelEncoder):
    # Print header for this section
    print("## Getting results of the incorrect predictions...")
    
    # Initialize list to store incorrect predictions
    incorrect_predictions = []
    
    # Iterate over predictions and compare to true labels
    for i in range(len(predictions)):
        # If the prediction is incorrect, store details in list
        if y_test[i] != predictions[i]:
            # Format the incorrect prediction details as a string
            prediction_details = f"  {i+1}. {vectorizer.inverse_transform(X_test[i])[0]}: Expected Result '{label_encoder.inverse_transform([y_test[i]])[0]}' : Predicted Result '{label_encoder.inverse_transform([predictions[i]])[0]}'"
            incorrect_predictions.append(prediction_details)
    
    # Print number of incorrect predictions
    print(f">> Number of incorrect predictions: {len(incorrect_predictions)}")
    
    # Print details of each incorrect prediction
    for incorrect_prediction in incorrect_predictions:
        print(incorrect_prediction)

# Define a function to evaluate the performance of the model
def evaluate_model(model: svm.SVC, X_test: list, y_test: list, vectorizer: CountVectorizer, label_encoder: LabelEncoder):
    # Display a message indicating that the model performance is being evaluated
    print("### Evaluating model performance...")
    
    # Predict the labels for the test data
    y_pred = model.predict(X_test)
    
    # If the model is a TensorFlow model, get the probabilities for the predicted labels
    if isinstance(model, tf.keras.Sequential):
        y_pred = get_probabilities(y_pred)
    
    # Display the classification report and model accuracy
    print(f">> Classification Report: ")
    print(classification_report(y_test, y_pred))
    print(f">> Model accuracy: {accuracy_score(y_test, y_pred)}")
    
    # Call the get_incorrect_predictions function to print out details of any incorrect predictions made by the model
    get_incorrect_predictions(y_pred, X_test, y_test, vectorizer, label_encoder)

# Function for finding hyperparameters for model optimization
def find_hyperparameters(X_train:list, y_train:list, X_test:list, y_test:list, vectorizer:CountVectorizer, label_encoder:LabelEncoder):
    print("### Finding suitable parameters for model optimization...")
    
    # Setting the parameter grid for SVM
    param_grid = {'C': [0.1, 1, 10],
                  'kernel': ['linear']}
    
    # Creating an SVM object
    svc = svm.SVC()
    
    # Using grid search to find the best hyperparameters
    grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy', verbose=2)
    grid_search.fit(X_train, y_train)
    best_svc = grid_search.best_estimator_
    
    # Printing the best hyperparameters
    print('>> Best Hyperparameters: ', grid_search.best_params_)
    print('>> Best Accuracy Score: ', grid_search.best_score_)
    
    # Option list for choice in the application menu
    option_list = {1: "Try Again", 2: "Train Model", 0: "Back"} 
    
    # Displaying the option menu
    choice = option_menu(option_list, "\nPlease choose an option from the menu.:") 
    
    # Conditional statements for the menu options
    if choice == 1:
        find_hyperparameters(X_train, y_train, X_test, y_test, vectorizer, label_encoder)
    elif choice == 2:
        option_train_model(best_svc, X_train, y_train, X_test, y_test, vectorizer, label_encoder)
    else:
        main()

# Function for predicting the model output
def predict_model(model, input_texts:list, vectorizer:CountVectorizer, label_encoder:LabelEncoder):
    print("### Predicting...")
    
    # Vectorizing the input text using the provided vectorizer
    vectorized_text = vectorizer.transform(clean_texts(input_texts))
    
    # Predicting using the trained model
    prediction = model.predict(vectorized_text)
    
    # If the model is a sequential keras model, probabilities are obtained using get_probabilities function
    if isinstance(model,tf.keras.Sequential):
        prediction = get_probabilities(prediction)
        
    # Inverse transforming the predicted values using the provided label encoder
    print("### Getting result...")
    prediction = label_encoder.inverse_transform(prediction)
    
    # Returning the predicted result
    return prediction

# Function for transferring the model from SVM to TensorFlow
def transfer_model(svc_model:svm.SVC, X_test:list):
    print("### Transferring model into TensorFlow model...")
    
    # Creating a TensorFlow model with a single dense layer and sigmoid activation
    tf_model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(X_test.shape[1],)),
        tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    
    # Setting the weights and biases of the Keras model using the trained SVM model
    tf_model.layers[0].weights[0].assign(svc_model.coef_.transpose().toarray())
    tf_model.layers[0].bias.assign(svc_model.intercept_)
    
    # Compiling the Keras model
    tf_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Printing the completion message
    print("### Model has been transferred successfully!")
    
    # Returning the transferred model
    return tf_model

# Function to get probabilities of prediction
def get_probabilities(prediction):
    return np.where(prediction > 0.5, 1,0).flatten()

# Function to verify transfered model
def verify_transfered_model(transfered_model:tf.keras.Sequential,orginal_model:svm.SVC,X_test:list):
    print("### Verifying transfered model...")
    transfered_model_prediction = get_probabilities(transfered_model.predict(X_test))
    orginal_model_prediction = orginal_model.predict(X_test)
    
    # Check if the prediction of transfered model and original model are same
    model_verification = np.all(transfered_model_prediction == orginal_model_prediction)
    if model_verification:
        print("### Model verification has been successfully!")
    else:
        print("### Model verification has been unsuccessfully!")
        
    # Raise an assertion error if model verification is unsuccessful
    assert model_verification
    return model_verification

# Function to save the trained model, vectorizer and label encoder
def save_model(tf_model:tf.keras.Sequential,vectorizer:CountVectorizer,label_encoder:LabelEncoder):
    print("### Saving model to file...")
    try:
        # Save the model, vectorizer and label encoder
        tf_model.save(project_name+'_model.h5')
        joblib.dump(vectorizer, project_name+"_vectorizer.joblib")
        with open(project_name+"_label_encoder.pickle", 'wb') as f:
            pickle.dump(label_encoder, f)
        print("### Model saved successfully!")
        
        # Ask user for next step
        option_list = {1: "Try Again", 2: "Test Model", 0: "Back"} 
        choice = option_menu(option_list,"\nPlease choose an option from the menu.:")
        if choice == 1:
            save_model(tf_model,vectorizer,label_encoder)
        elif choice == 2:
            test_model(tf_model,vectorizer,label_encoder)
        else:
            main()
    except Exception as e:
        print("### An error occurred while saving the model:", e)

# Function to load saved model, vectorizer and label encoder
def load_model():
    try:
        print("### Loading saved model from local storage...")
        
        # Load the saved model, vectorizer and label encoder
        tflite_model = tf.keras.models.load_model(project_name+'_model.h5')
        vectorizer = joblib.load(project_name+"_vectorizer.joblib")
        with open(project_name+"_label_encoder.pickle", 'rb') as f:
            label_encoder = pickle.load(f)
        print("### Model has been loaded successfully!")
    except Exception as e:
        print("### Error loading the model:", e)
        return None, None, None
    return tflite_model, vectorizer, label_encoder

# This function trains the model and gives the user an option to save it or try again
def option_train_model(model:svm.SVC,X_train:list,y_train:list,X_test:list,y_test:list,vectorizer:CountVectorizer,label_encoder:LabelEncoder):
    
    # if there is no model, train one
    if not model:
        model = train_model(X_train,y_train)
        
    # evaluate the model with test data
    evaluate_model(model,X_test,y_test,vectorizer,label_encoder)
    
    # transfer the model to new_model
    new_model = transfer_model(model,X_test)
    
    # verify if the new model is better
    verify_transfered_model(new_model,model,X_test)
    
    # evaluate the new model
    evaluate_model(new_model,X_test,y_test,vectorizer,label_encoder)
    
    # display options for the user
    option_list = {1: "Try Again", 2: "Save Model", 0: "Back"} 
    
    # ask for user choice
    choice = option_menu(option_list,"\nPlease choose an option from the menu.:")
    
    # handle the user choice
    if choice == 1:
        # train the model again
        option_train_model(model,X_train,y_train,X_test,y_test,vectorizer,label_encoder)
    elif choice == 2:
        # save the new model
        save_model(new_model,vectorizer,label_encoder)
    else:
        # go back to main menu
        main()

# This function tests the model and gives the user an option to try again, load model or go back
def test_model(model:tf.keras.Sequential,vectorizer:CountVectorizer,label_encoder:LabelEncoder):
    
    # get user input
    user_input = input("Prediction for input text: ")
    
    # if any of the model, vectorizer, and label_encoder is missing, load them
    if not(model or vectorizer or label_encoder):
        model, vectorizer, label_encoder = load_model()
        
    # predict the result for the user input
    prediction = predict_model(model,[user_input],vectorizer,label_encoder)
    print(f">> Predicted output: {prediction}")
    
    # display options for the user
    option_list = {1: "Try Again", 2: "Load Model", 0: "Back"} 
    
    # ask for user choice
    choice = option_menu(option_list,"\nPlease choose an option from the menu:")
    
    # handle the user choice
    if choice == 1:
        # test the model again
        test_model(model,vectorizer,label_encoder)
    elif choice == 2:
        # load the model
        model, vectorizer, label_encoder = load_model()
        
        # test the model again
        test_model(model,vectorizer,label_encoder)
    else:
        # go back to main menu
        main()

# This is the main function that handles user inputs and calls other functions accordingly
def main():
    
    # Print a logo or signature of the creator.
    print("#BY_SKY_PRODUCTION") 

    # Create ASCII art using the text2art library.
    Art = text2art("SinEn", font="univers") 

    # Print the ASCII art created.
    print(Art) 

    # Print a welcome message.
    print(decor("barcode1") + "Welcome, SinEn Bad Words Detector!" + decor("barcode1", reverse=True))
    
    # Creating a dictionary with the available options and their corresponding numbers
    option_list = {1: "Find Hyperparameters", 2: "Train Model", 3: "Test Model", 0: "Exit"} 
    
    # Displaying the available options to the user and getting their choice
    choice = option_menu(option_list,"\nWelcome to our application! Please choose an option from the menu.:") 
    
    # If the user chooses either to find hyperparameters or train a model
    if (choice == 1) or (choice == 2):
        # Loading the dataset and labels from the CSV file
        data, labels = load_datasets(bad_words_list_dir)
        
        # Checking if the dataset or labels are empty
        if not(data.empty or (len(labels) == 0)):
            # Preprocessing the dataset by cleaning the text, vectorizing it, and encoding the labels
            data, vectorizer, labels, label_encoder = preprocess_data(clean_texts(data),labels)
            
            # Splitting the preprocessed dataset into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(data,labels, test_size=0.3, random_state=42)
            
            # If the user chooses to find hyperparameters, call the corresponding function
            if choice == 1:
                find_hyperparameters(X_train,y_train,X_test,y_test,vectorizer,label_encoder)
            # If the user chooses to train a model, call the corresponding function
            elif choice == 2:
                option_train_model(None,X_train,y_train,X_test,y_test,vectorizer,label_encoder)
                
    # If the user chooses to test a model, call the corresponding function
    elif choice == 3:
        test_model(None,None,None)
    # If the user chooses to exit the application, print a farewell message and exit the program
    elif choice == 0:
        print("---------- Thank you for using our program! ----------")
        sys.exit()
    # If the user enters an invalid input, display an error message and call the main function again
    else:
        print("---------- Error: Invalid input. Please try again. ----------") 
        main()
    
if __name__ == "__main__":
    main()