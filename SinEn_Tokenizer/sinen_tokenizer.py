# Import necessary libraries and modules.
import os
import sys
import string
from art import *
from nltk import word_tokenize
from string import digits

# Get the name of the directory where this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Get the parent directory name where the current directory is present.
parent = os.path.dirname(current)

# Add the parent directory to the sys.path.
sys.path.append(parent)

# Now we can import the module in the parent directory.
from option_menu import option_menu

# Define a class for the SinEn tokenizer.
class SinEnTokenizer:
    def __init__(self):
        super().__init__()
        self.stop_words = load_stop_words()
        
    def tokenize(self, sentence):
        # Remove all punctuation from the input sentence
        sentence = str(sentence).translate(str.maketrans("","",string.punctuation))

        # Remove all digits from the input sentence
        sentence = sentence.translate(str.maketrans("","",digits))

        # Convert the input sentence to lowercase
        sentence = str(sentence).lower()

        # Tokenize the input sentence into individual words
        words = word_tokenize(sentence)

        # Load the stop words
        stop_words = set(self.stop_words)

        # Filter out any stop words from the tokenized words
        filtered_words = [word for word in words if word not in stop_words]

        # Return the filtered words
        return filtered_words

# Define a function to load the stop words.
def load_stop_words():
    stop_words = []
    with open("../data/stop_words_singlish.txt", 'r') as fp:
        for stop_word in fp.read().split('\n'):
            stop_words.append(stop_word)
    return stop_words

# Define a function to tokenize the text.
def text_tokenize():
    # Prompt the user to input some text.
    text = input("Input your text: ")
    
    # Create a SinEn tokenizer object.
    tokenizer = SinEnTokenizer()
    
    # Tokenize the input text.
    tokenized_text = tokenizer.tokenize(text)
    
    # Print the tokenized text.
    print(f"Tokenized Text: {tokenized_text}")
    
    # Dictionary of options for the user to choose from.
    option_list = {1: "Try another text", 0: "Back"} 

    # Prompt the user to choose an option from the dictionary.
    choice = option_menu(option_list, "\nSelect an option: ") 
    
    if choice == 1:
        # If the user chooses to try another text, call the text_tokenize() function again.
        text_tokenize()
    elif choice == 0:
        # If the user chooses to go back, call the main() function.
        main()

# Define the main function.
def main():
    # Print a logo or signature of the creator.
    print("#BY_SKY_PRODUCTION") 

    # Create ASCII art using the text2art library.
    Art = text2art("SinEn", font="univers") 

    # Print the ASCII art created.
    print(Art) 

    # Print a welcome message.
    print(decor("barcode1") + "Welcome, SinEn Tokenizer!" + decor("barcode1", reverse=True))
    
    # Define a dictionary of options for the user to choose from.
    option_list = {1: "Text Tokenizing", 0: "Exit"} 

    # Prompt the user to choose an option from the dictionary.
    choice = option_menu(option_list, "\nSelect an option: ")
    
    if choice == 1:
        # If the user chooses text tokenizing, call the text_tokenize() function.
        text_tokenize()
    elif choice == 0:
        # If the user chooses to exit, print a goodbye message and exit the program.
        print("Thank you for using our program!\n")
        exit()

if __name__ == "__main__":
    main()
