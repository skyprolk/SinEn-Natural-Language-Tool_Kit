# Import necessary libraries and modules
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import os
import sys
import pandas as pd
from art import *
import joblib
import pickle

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

# Create a class called "SinEn" to encapsulate the Sinhala-English code mixing detection functionality
class SinEn:
    def __init__(self,dir):
        # Store the directory containing the data file as an instance variable
        self.dir = dir
        # Set the name of the model
        self.model_name = "sinen_bad_words_detector"
        # Set the directory to save resources
        self.directory = "../resources/"
        
    # Define a function to clean the text data
    def clean_text(self,texts:list):
        # Create an empty list to store cleaned texts
        cleaned_texts = []
        # Create a SinEnStemmer object
        stemmer = SinEnStemmer()
        # Create a SinEnTokenizer object
        tokenizer = SinEnTokenizer()
        # Iterate over each text in the list of texts
        for text in texts:
            # Tokenize the text using the SinEnTokenizer object
            text = tokenizer.tokenize(text)
            # Create an empty list to store cleaned words
            cleaned_words = []
            # Iterate over each word in the tokenized text
            for word in text:
                # Stem the word using the SinEnStemmer object
                word = stemmer.stem(word)[0]
                # Append the cleaned word to the cleaned_words list
                cleaned_words.append(word)
            # Join the cleaned words with a space and append to the cleaned_texts list
            cleaned_texts.append(" ".join(cleaned_words))
        # Return the cleaned texts
        return cleaned_texts

    # Define a function to load the dataset
    def load_dataset(self):
        # Load the data into a pandas DataFrame
        data = pd.read_csv(self.dir)
    
        # Extract the labels from the DataFrame
        label_list = []  # initialize an empty list to store unique labels
        labels = []  # initialize an empty list to store all labels
        for label in data['label']:
            label = "".join(str(label).split())  # remove whitespace from the label
            if label not in label_list:
                label_list.append(label)  # add the unique label to the list
            labels.append(label)  # add the label to the labels list
    
        # Print some information about the labels
        print(f"Labels Count : {len(label_list)} Labels : {label_list}")
        datasets_size = ""
        for label in label_list:
            datasets_size += f"[{label} = {labels.count(label)}] "
        # Print the size of the dataset
        print("Dataset Size: "+datasets_size)
    
        # Clean the text in the 'sentence' column of the DataFrame
        self.sentences = self.clean_text(data['sentence'])
        self.labels = labels


    # Preprocess the data
    def preprocess_data(self):
        # Create a vectorizer for converting text into vectors of character n-grams
        self.vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 4))
        # Convert the sentences into vectors using the vectorizer
        self.X = self.vectorizer.fit_transform(self.sentences)
        self.y = self.labels
        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
    
    # Train the SVM model
    def train_model(self):
        # Create an SVM model with optimal hyperparameters
        svm = SVC(C=1, kernel='linear',verbose=2)
        # Train the SVM model on the entire training set
        svm.fit(self.X_train, self.y_train)
        self.model = svm

    # Use grid search to find the best hyperparameters for the SVM model
    def grid_search(self):
        # Define the parameter grid to search
        param_grid = {'C': [0.1, 1, 10],
                      'kernel': ['linear', 'rbf', 'poly']}
        # Create an SVM model
        svm = SVC()
        # Create a grid search object
        grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy',verbose=2)
        # Fit the grid search to the training data
        grid_search.fit(self.X_train, self.y_train)
        # Print the best hyperparameters and accuracy score
        print('Best Hyperparameters: ', grid_search.best_params_)
        print('Best Accuracy Score: ', grid_search.best_score_)

    # Evaluate the performance of the trained model on the test set
    def evaluate_model(self):
        # Use the trained model to predict class labels for the test set
        y_pred = self.model.predict(self.X_test)
        
        # Compute the accuracy of the predictions using the ground-truth labels
        accuracy = accuracy_score(self.y_test, y_pred)
        
        # Print the accuracy to the console
        print(f"Accuracy: {accuracy}")
        
        # Compute cross-validation scores using the trained model and test data
        scores = cross_val_score(self.model, self.X_test, self.y_test, cv=5)
    
        # Print the cross-validation accuracy scores and the mean score
        print(f"Cross-validation scores: {scores}")
        print(f"Mean accuracy: {scores.mean()}")

    # This method saves the trained model and training data
    def save_model(self):
        try:
            # Save the model as a joblib file
            joblib.dump(self.model, self.model_name+".joblib")

            # Save the training data as a pickle file
            with open((self.model_name+'_train_data.pickle'), 'wb') as file:
                pickle.dump(self.vectorizer, file, protocol=pickle.HIGHEST_PROTOCOL)

            # Print a message indicating that the model saving process completed successfully
            print("Model saving completed!")
        except Exception as e:
            # If an exception occurs while saving the model or training data, print an error message
            print("Error in saving model and train data: ", e)

        # Create a dictionary of options for the user to choose from
        option_list = {1: "Save Again", 2: "Test Model", 0: "Back"} 

        # Prompt the user to choose an option from the dictionary using another method called `option_menu`
        choice = option_menu(option_list,"\nPlease choose an option:")

        # Take action based on the user's choice
        if choice == 1:
            # If the user selects option 1, recursively call this method to save the model and training data again
            self.save_model()
        elif choice == 2:
            # If the user selects option 2, call another method called `test_model` to test the saved model
            self.test_model()
        else:
            # If the user selects any other option, call the main method to return to the main menu
            main()

    # This method loads the saved model and training data
    def load_model(self):
        # Create a directory string with the directory name and model name
        directory =  (self.directory+ self.model_name)

        # Create the joblib directory string
        joblib_dir = directory+".joblib"

        # Check if the joblib file exists
        if os.path.isfile(joblib_dir):
            # If yes, load the model
            model = joblib.load(joblib_dir)
            self.model = model
        else:
            # If not, print a message and go back to the main function
            print("Model not found in the specified directory.")
            main()

        # Create the pickle directory string
        pickle_dir = directory+"_train_data.pickle"

        # Check if the pickle file exists
        if os.path.isfile(pickle_dir):
            # If yes, load the training data
            with open(pickle_dir, "rb") as file:
                train_data = pickle.load(file)
                self.vectorizer = train_data
        else:
            # If not, print a message and go back to the main function
            print("Train data not found in the specified directory.")
            main()

    # This method tests the saved model
    def test_model(self):
        # Load the saved model
        self.load_model()

        # Get the user input
        user_input = input("Enter your text: ")

        # Clean the text
        user_input = self.clean_text([user_input])

        # Vectorize the text
        text_vec = self.vectorizer.transform(user_input)

        # Predict the class of the text
        prediction = self.model.predict(text_vec)

        # Check if the text is appropriate or not
        if prediction[0].lower() == 'good':
            print("The text is appropriate.")
        else:
            print("The text is inappropriate.")

        # Create a dictionary of options for the user to choose from
        option_list = {1: "Again", 0: "Back"} 

        # Prompt the user to choose an option from the dictionary
        choice = option_menu(option_list,"\nPlease choose an option:") 

        # Check the user's choice
        if choice == 1:
            # If the user chooses 1, run the test_model function again
            self.test_model()
        else:
            # If the user chooses anything else, go back to the main function
            main()

    # Define the `train_model_option` method
    def train_model_option(self, grid_search=False):
        # Load and preprocess the data
        self.load_dataset()
        self.preprocess_data()

        # Perform grid search if `grid_search` is True, else train and evaluate the model
        if grid_search:
            self.grid_search()
        else:
            self.train_model()
            self.evaluate_model()

        # Define the dictionary of options for the user to choose from
        if grid_search:
            option_list = {1: "Search Again", 2: "Train Model", 0: "Back"} 
        else:
            option_list = {1: "Train Again", 2: "Save Model", 0: "Back"} 

        # Prompt the user to choose an option from the dictionary
        choice = option_menu(option_list, "\nPlease choose an option:")

        # Take action based on the user's choice
        if choice == 1:
            # If the user selects option 1, recursively call this method to perform grid search again or train the model again
            self.train_model_option(grid_search)
        elif choice == 2 and grid_search:
            # If the user selects option 2 and grid_search is True, recursively call this method to train the model again
            self.train_model_option()
        elif choice == 2:
            # If the user selects option 2, call the `save_model` method to save the trained model and cross-validation data
            self.save_model()
        else:
            # If the user selects any other option, call the main method to return to the main menu
            main()

# Define the `main` method
def main():
    # Print a logo or signature of the creator.
    print("#BY_SKY_PRODUCTION") 
    
    # Create an art using the text2art library
    Art=text2art("SinEn",font="univers") 
    
    # Print the art created
    print(Art) 

    # Print a welcome message
    print(decor("barcode1") + "Welcome, SinEn Bad Words Detector!" + decor("barcode1",reverse=True))
    
    # Dictionary of options for the user to choose from
    option_list = {1: "Grid Search", 2: "Train Model", 3: "Test Model", 0: "Exit"} 

    # Prompts the user to choose an option from the dictionary
    choice = option_menu(option_list,"\nPlease choose an option:") 

    if((choice == 1) or (choice == 2) or (choice == 3)): 
        # Create a SinEn object with the path to the bad word list CSV file
        sinen = SinEn('../data/bad_word_list_singlish.csv') 

    # If user chooses option 1, perform a grid search to find best hyperparameters
    if(choice == 1):
        # Call the `train_model_option` method with `grid_search=True` to perform a grid search
        sinen.train_model_option(True)

    # If user chooses option 2, train the model with default hyperparameters
    elif(choice == 2):
        # Call the `train_model_option` method with `grid_search=False` to train the model
        sinen.train_model_option()

    # If user chooses option 3, test the saved model
    elif(choice == 3):
        # Call the `test_model` method to test the saved model
        sinen.test_model()

    # If user chooses option 0, exit the program with a goodbye message
    elif(choice == 0):
        print("Thank you for using our program!\n")
        sys.exit()
    else:
        # If the user input is invalid, prompt the user again
        print("Invalid input, please enter a valid option from the menu.\n") 
        main()
    
if __name__ == "__main__":
    main()


