import glob
import chardet
import os
import sys
import re
from art import *

# Getting the directory where this file is located.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory of the current directory.
parent = os.path.dirname(current)

# Adding the parent directory to the sys.path.
sys.path.append(parent)

# Importing the module from the parent directory.
from option_menu import option_menu

# Defining the name of the output file.
output_file_name = "bag_of_words.txt"

# Defining the range of Sinhala script characters.
sinhala_script_range = "[\u0D80-\u0DFF]+"

# Class to handle the bag of words making process.
class SinEn:
    
    # Constructor to initialize the directory.
    def __init__(self,directory):
        self.directory = directory
        
    # Method to get all text files in the directory.
    def get_files(self):
        # Use the glob library to get all text files in the directory.
        text_files = glob.glob(self.directory+'*.txt')
        return text_files
    
    # Method to get the contents of the text files.
    def get_texts(self,text_files):
        texts = []
        # Loop through each text file in the list.
        for text_file in text_files:
                # Check if the file exists.
                if os.path.exists(text_file):
                    # Open the file in binary mode to detect the encoding.
                    with open(text_file, "rb") as file:
                        result = chardet.detect(file.read())
                        encoding = result["encoding"]
                        # Open the file in text mode using the detected encoding.
                        with open(text_file, 'r', encoding=encoding) as file:
                            content = file.read()
                            print("Get texts from = = => "+os.path.basename(text_file))
                            # Append the contents of the file to the texts list.
                            texts.append(content)
                else:
                    # Print an error message if the file does not exist.
                    print("The file does not exist")
        return texts, encoding
    
    # Method to save the contents of the texts to a file.
    def save_texts_file(self,file_name,texts,encoding):
        # Open the file in write mode using the given encoding.
        with open(file_name, "w" , encoding=encoding) as f:
            # Write each text in the texts list to a new line in the file.
            for text in texts:
                f.write(text + "\n")
                
    # Method to filter the texts to only contain Sinhala script characters.
    def filter_texts(self,texts):
        # Join the texts into a single string.
        texts = "".join(texts)
        # Use regex to find all instances of Sinhala script characters in the string.
        texts = re.findall(sinhala_script_range, texts)
        return texts
    
    def remove_duplicates(self,texts):
        # Remove duplicates from the list of texts
        texts = list(set(texts))
        # Return the list without duplicates
        return texts

    def sort_texts(self,texts):
        # Sort the texts in alphabetical order
        texts = sorted(texts)
        # Return the sorted list of texts
        return texts

def main_options(sinen):
    # Dictionary of options for the user to choose from
    option_list = {1: ("Make "+output_file_name),2: ("Filter "+output_file_name), 0: "Exit"}
    # Prompts the user to choose an option from the dictionary
    choice = option_menu(option_list,"\nPlease choose an option:")
    if(choice == 1):
        # Get the texts and the encoding of the input files
        texts , encoding = sinen.get_texts(sinen.get_files())
        # Save the texts to a file
        sinen.save_texts_file(output_file_name,texts, encoding)
        # Print a success message
        print("\nSuccessfully saved ["+output_file_name+ "]")
        # Call the main options function again to let the user make another choice
        main_options(sinen)
    elif(choice == 2):
        # Get the texts and encoding from the previously created output file
        texts , encoding = sinen.get_texts([output_file_name])
        # Filter the texts
        texts = sinen.filter_texts(texts)
        # Remove duplicates from the filtered texts
        texts = sinen.remove_duplicates(texts)
        # Sort the texts in alphabetical order
        texts = sinen.sort_texts(texts)
        # Get the name of the output file without the extension
        file_name = os.path.splitext(output_file_name)
        # Create a new name for the filtered file
        file_name = file_name[0]+"_filtered"+file_name[1]
        # Save the filtered and sorted texts to a new file
        sinen.save_texts_file(file_name,texts,encoding)
        # Print a success message
        print("\nSuccessfully filtered ["+output_file_name+ "] and saved as [" + file_name + "]\n")
    elif(choice == 0):
        # Print a goodbye message and exit the program
        print("Thank you for using our program!\n")
        # Exit the program
        sys.exit() 
        
def main():
    # Create an instance of the SinEn class
    sinen = SinEn("../data/extracted_data/")
    # Print the creator's logo or signature
    print("#BY_SKY_PRODUCTION") 

    # Create a text art with the text "SinEn" and font "univers"
    Art = text2art("SinEn", font="univers") 

    # Print the created text art
    print(Art) 

    # Print a welcome message
    print(decor("barcode1") + "Welcome, SinEn Bag of Words Maker!" + decor("barcode1", reverse=True))

    # Call the main_options function with the SinEn instance
    main_options(sinen)

if __name__ == "__main__":
    main()
    
