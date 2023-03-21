# Import the required modules
from converter import Converter
from art import *
import gui
import chardet

# Define the SinEn class
class SinEn:
    def __init__(self):
        # Create a Converter object
        self.converter = Converter()

    # Method to convert a word to Singlish
    def convert_to_singlish(self, word: str):
        # Convert the word to Singlish
        return self.converter.convert(word, Converter.ConvertTo.English)

    # Method to convert a word to Phoneme
    def convert_to_phoneme(self, word: str):
        # Convert the word to Phoneme
        return self.converter.convert(word, Converter.ConvertTo.Phoneme)

    # Method to open a file and read its content
    def open_file(self, file_path):
        try:
            # Use the "chardet" module to detect the file's encoding
            with open(file_path, "rb") as file:
                result = chardet.detect(file.read())

            # Open the file with the detected encoding and read its content
            with open(file_path, "r", encoding=result["encoding"]) as file:
                text = file.read()
                return text,""
        except Exception as e:
            title = "Error opening file"
            body = f"An error occurred while opening the file. Error: {str(e)}"
            return title, body


    # Method to save the converted text to a file
    def save_file(self, file_path, text):
        try:
            # Open the file for writing
            with open(file_path, "w", encoding="utf-8") as f:
                # Write the text to the file
                f.write(text)
                title = "File Saved!"
                body = ("The file has been successfully.")
                error = False
                return title,body, error
        except Exception as e:
            # Handle the exception and display an error message
            title = "Error saving file"
            body = f"An error occurred while saving the file. Error: {str(e)}"
            error = True
            return title, body, error


# Define the main function
def main():
    # Print the "#BY_SKY_PRODUCTION" message
    print("#BY_SKY_PRODUCTION")

    # Print the "SinEn" text art
    art = text2art("SinEn", font="univers")
    print(art)

    # Print a welcome message
    print(decor("barcode1") + "Welcome to the SinEn Converter!" + decor("barcode1", reverse=True))

    # Open the GUI
    print("[--GUI OPENED--]")
    gui_ = gui.GUI()
    gui_.main()
    print("[--GUI CLOSED--]")

# Check if the script is being run as the main program
if __name__ == "__main__":
    # Call the main function
    main()