import pygtrie as trie
from art import *
import os
import sys

# Get the name of the directory where this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Get the parent directory name where the current directory is present.
parent = os.path.dirname(current)

# Add the parent directory to the sys.path.
sys.path.append(parent)

# Now we can import the module in the parent directory.
from option_menu import option_menu

# Define the vowel list.
vowel_list = ('a','e','i','o','u')

# Define a SinEnStemmer class that will contain functions for word stemming.
class SinEnStemmer:
    def __init__(self):
        super().__init__()

        # Load the stem dictionary and suffixes.
        self.stem_dictionary = load_stem_dictionary()
        self.suffixes = load_suffixes()

    # Define a function that will perform the actual word stemming.
    def stem(self, word):
        # Check if the word is in the stem dictionary.
        if word in self.stem_dictionary:
            return self.stem_dictionary[word]
        else:
            # If not, find the longest matching suffix and return the base and suffix as a tuple.
            suffix = self.suffixes.longest_prefix(word[::-1]).key
            if suffix is not None:
                # If a suffix is found, get the root word and its length.
                root_word = word[0:-len(suffix)]
                root_len = 0
                # Count the number of consonants in the root word.
                for letter in root_word:
                    if not letter in vowel_list:
                        root_len += 1
                # Apply the singlish stemming rules to the root word and suffix.
                if (str(root_word).startswith(vowel_list) and root_len >= 1) or (not str(root_word).startswith(vowel_list) and root_len > 1 and len(root_word) > 2):
                    # If the root word starts with a vowel, it must have at least one consonant.
                    # If it starts with a consonant, it must have at least two consonants and be longer than two characters.
                    return root_word, word[len(word) - len(suffix):]
                else:
                    # If the root word does not meet the stemming rules, return the original word and an empty suffix.
                    return word, ''
            else:
                # If no matching suffix is found, return the original word and an empty suffix.
                return word, ''

# Load a trie containing the list of Singlish suffixes.
def load_suffixes():
    suffixes = trie.Trie()
    with open("../data/suffixes_list_singlish.txt", 'r') as fp:
        for suffix in fp.read().split('\n'):
            suffixes[suffix[::-1]] = suffix
    return suffixes

# Load a dictionary containing the stem words and their suffixes.
def load_stem_dictionary():
    stem_dict = dict()
    with open("../data/stem_dictionary_singlish.txt", 'r') as fp:
        for line in fp.read().split('\n'):
            try:
                base, suffix = line.strip().split('\t')
                stem_dict[f'{base}{suffix}'] = (base, suffix)
            except ValueError as _:
                # Skip lines that can't be parsed into a base and suffix.
                pass
    return stem_dict

# Define a function for prompting the user to input a word and stemming it.
def word_stemming():
    word = input("Input your word: ")
    stemmer = SinEnStemmer()
    stemmed_word = stemmer.stem(word)
    print(f"Stemmed Word: {stemmed_word}")
    
    # Dictionary of options for the user to choose from.
    option_list = {1: "Try another word", 0: "Back"} 

    # Prompt the user to choose an option from the dictionary.
    choice = option_menu(option_list, "\nSelect an option: ") 
    
    # If the user chooses to try another word, call the word_stemming function again.
    if choice == 1:
        word_stemming()
    # If the user chooses to go back to the main menu, call the main function.
    elif choice == 0:
        main()

# Define a function for the main menu.
def main():
    
    # Print a logo or signature of the creator.
    print("#BY_SKY_PRODUCTION") 

    # Create ASCII art using the text2art library.
    Art = text2art("SinEn", font="univers") 

    # Print the ASCII art created.
    print(Art) 

    # Print a welcome message.
    print(decor("barcode1") + "Welcome, SinEn Stemmer!" + decor("barcode1", reverse=True))

    # Define a dictionary of options for the user to choose from.
    option_list = {1: "Word Stemming", 0: "Exit"} 

    # Prompt the user to choose an option from the dictionary.
    choice = option_menu(option_list, "\nSelect an option: ")

    # Check the user's choice and call the corresponding function.
    if choice == 1:
        word_stemming()
    elif choice == 0:
        # Print a goodbye message and exit the program.
        print("Thank you for using our program!\n")
        exit()
        
if __name__ == "__main__":
    main()