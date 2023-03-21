# Source code from https://github.com/aspriya/Sinhala-Transliterator/blob/master/Sripali.py
# Modified by KNOiT
# coding : UTF-8
from tkinter import *
from typing import List
import json
from enum import Enum

class Converter:
    # Open the mapping.json file containing the mappings of Sinhala letters to their corresponding phonemes and English transliterations
    with open('../data/mapping.json', 'r', encoding='UTF-8') as f:
        mappings = json.load(f)

    # Create lists of vowels, sripali letters, and special characters
    vowel_list = mappings["vowel_list"]  # list of vowels
    sripali_list = mappings["sripali_list"]  # list of sripali letters
    big_list = vowel_list + sripali_list  # list of both vowels and sripali letters

    special_character_list = mappings["special_character_list"]  # list of special characters
    unwanted_symbols = mappings["unwanted_symbols"]  # list of unwanted symbols

    not_mapping_identifier = "$X0"  # unique identifier for letters not in the mappings dictionary

    # Enum class to specify whether the input text should be converted to phonemes or English transliteration
    class ConvertTo(Enum):
        Phoneme = "phoneme"  # convert text to phonemes
        English = "english"  # convert text to English transliteration

    # Main function to convert the input text
    def convert(self, word, convertTo):
        # Remove English and unwanted symbols
        #text = self.remove_unwanted_symbols(word)
        text = word

        # Create a list of mapped letters from the input word
        mapped_text_list = []
        mapping_letters = [list(x.keys())[0] for x in self.mappings["letters"]]
        for letters in word:
            # Check if the letter is in the mappings dictionary and append the mapped letter to the list
            for letters_list in self.mappings["letters"]:
                if letters in letters_list:
                    mapped_text_list.append(letters_list[letters][convertTo.value])
            # If the letter is not in the mappings dictionary, append it to the list with the unique identifier
            if not letters in mapping_letters:
                mapped_text_list.append(letters+self.not_mapping_identifier)

        # Insert the appropriate phoneme for the letter "a"
        result = self.insert_a(text, mapped_text_list)

        # Return the joined result string
        return "".join(result)

    # Function to remove unwanted symbols from the input text
    def remove_unwanted_symbols(self,text):
        # remove english and unwanted symbols
        text_temp_list = list(text)
        text_list = [x for x in text_temp_list if x not in Converter.unwanted_symbols]
        #print("original is: ", text_temp_list)
        #print("unwanted symbols removed list is ", text_list)
        result = "".join(text_list)
        #print("unwanted symbols removed text is ", result)
        return result

    # Function to insert the /a/ phoneme in the appropriate places in the mapped text
    def insert_a(self, text: str, mapped_text_list: List[str]) -> List[str]:
        # Convert the input text to a list
        text_list = list(text)
        # Append a dummy non effecting symbol at end of text_list to overcome error of length mis match of occuring in rule 2)
        text_list.append("_")
        a_inserted_list = []
    
        # Iterate over the mapped text list and insert the /a/ phoneme where appropriate
        for i, letter in enumerate(mapped_text_list):
            # If the letter is not in the mappings dictionary, add it to the list without the /a/ phoneme
            if letter.endswith(self.not_mapping_identifier) and (len(letter) == (len(self.not_mapping_identifier)+1)):
                a_inserted_list.append(letter.removesuffix(self.not_mapping_identifier))
                continue
            else:
                a_inserted_list.append(letter)
    
                # rule 1 : don't insert /a/ after a vowel phoneme representative
                if text_list[i] in self.big_list:
                    continue
                
                # rule 2 : don't insert /a/ after a phoneme symbol, if the next relevant symbol in text_list is in sripali_list.
                elif text_list[i+1] in self.sripali_list:
                    # Special case for handling characters like පංචිකාවත්ත where ං is just after a consonant
                    if text_list[i+1] == u"ං":
                        a_inserted_list.append('a')
                    continue
                
                # Handle spaces, commas, and new line characters
                elif letter in self.special_character_list:
                    continue
                
                # rule 3 : if the letter is not filtered by the above rules, then append /a/
                else:
                    a_inserted_list.append('a')
    
        # Return the list with the /a/ phoneme inserted
        return a_inserted_list
