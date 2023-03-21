import os
import sys
import requests
from bs4 import BeautifulSoup
from art import *
from urllib.parse import urljoin
from urllib.parse import urlsplit

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
 
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
 
# adding the parent directory to
# the sys.path.
sys.path.append(parent)
 
# now we can import the module in the parent
# directory.
from option_menu import option_menu

class SinEn:
    
    # A list of base urls for different Sinhala news websites
    base_urls = [
        "https://www.bbc.com/sinhala/",
        "https://www.hirunews.lk/",
        "https://sinhala.adaderana.lk/",
        "https://www.lankadeepa.lk/",
        "https://www.silumina.lk/",
        "https://sinhala.newsfirst.lk/",
        "https://androidwedakarayo.com/",
        "https://technews.lk/",
        "https://roar.media/sinhala/",
        "https://divaina.lk/",
        "https://mawbima.lk/",
        "https://www.ada.lk/",
        "https://www.deshaya.lk/",
        "https://www.gossiplankanews.com/",
        "https://ceylondaily.com/",
        "https://www.gossipclanka.com/",
        "https://sinhala.news.lk/",
        "https://lankatruth.com/si/",
        "https://www.vikalpa.org/",
        "https://si.wikipedia.org/wiki/",
        "https://si.wikibooks.org/wiki/",
        "https://www.navaliya.com/navaliya/",
        "https://nethnews.lk/",
        "https://gagana.lk/",
        "https://sinhala.lankanewsweb.net/",
        "https://sinhala.mawratanews.lk/",
        "https://gossip.hirufm.lk/",
        "https://www.itnnews.lk/",
        "https://puwath.lk/",
        "https://liveat8.lk/",
        "https://www.vidusara.lk/",
        "https://sinhala.asianmirror.lk/",
    ]

    def __init__(self,output_file_type):
        # Output file type to save the extracted text
        self.output_file_type = output_file_type
        self.file_name = ""

    def save_file(self,texts,file_name):
        # Setting the instance variable "file_name" to the specified file name with the output file type
        self.file_name = file_name+self.output_file_type
        with open(self.file_name, "w", encoding="utf-16") as f:
            # Write each text in the list to the file
            for text in texts:
                f.write(text + "\n")
                
def exit():
    # Print a goodbye message and exit the program
    print("Thank you for using our program!\n")
    # Exit the program
    sys.exit() 

def main():
    
    # Initialize an instance of the SinEn class
    sinen = SinEn(".txt")

    # Print a logo or signature of the creator
    print("#BY_SKY_PRODUCTION") 

    # Create an art using the text2art library
    Art = text2art("SinEn", font="univers") 

    # Print the art created
    print(Art) 

    # Print a welcome message
    print(decor("barcode1") + "Welcome, SinEn Bag of Words Collector!" + decor("barcode1", reverse=True))
    
    # Dictionary of options for the user to choose from
    option_list = {1: "Select All",2: "Select Manual",3: "Select Custom" , 0: "Exit"} 
    # Prompts the user to choose an option from the dictionary
    choice = option_menu(option_list,"\nSelect an option for the sites to extract text from: ")
    
    # Check if user selected option 2 (URL selection)
    if(choice == 2):
        # Initialize a new list to store the selected URLs
        new_base_urls = []
        # Loop through the list of all URLs
        for url in sinen.base_urls:
            # Ask the user if they want to extract text from the current URL
            answer = input("Do you want to extract text from the URL " + url + " ? (Y/n) : ")

            # If the user entered 'y' or 'Y', add the URL to the new list
            if answer == "y" or answer == "Y":
                new_base_urls.append(url)

        # Update the list of base URLs with the selected URLs
        sinen.base_urls = new_base_urls
    elif(choice == 3):
        custom_url = input("Enter your custom url : ")
        sinen.base_urls = [custom_url]
    elif(choice == 0):
        # Exit the program
        exit()

    # List to store the extracted text
    texts = []
    
    # Loop through each base URL
    for base_url in sinen.base_urls:
        base_url = base_url.rstrip("/")
        
        # Get the hostname from the base URL
        hostname = urlsplit(base_url).hostname

        # Try to make a request to the website
        try:
            response = requests.get(base_url)
        except requests.exceptions.RequestException as e:
            # Print error message if the request fails
            print("Error while retrieving URL: ", e)
            # Continue to the next iteration if the request fails
            continue
        
        # Try to parse the HTML content of the website
        try:
            soup = BeautifulSoup(response.content, "html.parser")
        except Exception as e:
            # Print error message if parsing the HTML fails
            print("Error while parsing HTML: ", e)
            # Continue to the next iteration if parsing fails
            continue
        
        # Extract the text from the HTML
        print("\nCollecting text from", base_url, "...")
        text = soup.get_text()
        texts.append(text)
        
        # Find all 'a' tags in the soup
        links = soup.find_all("a")
        
        # Loop through each 'a' tag in the list
        for link in links:
            # Extract the "href" attribute from the 'a' tag
            relative_link = link.get("href")
            try:
                # Check if the relative link starts with "/" or "http"
                if relative_link.startswith("/") or relative_link.startswith("http"):
                    try:
                        # Create a full link by combining the base URL and the relative link
                        full_link = urljoin(base_url, relative_link)

                        # Make a request to the full link
                        page_response = requests.get(full_link)

                        # Use BeautifulSoup to parse the HTML
                        page_soup = BeautifulSoup(page_response.text, 'html.parser')

                        # Extract the text from the HTML
                        print("\nCollecting text from", full_link, "...")
                        page_text = page_soup.get_text()
                        texts.append(page_text)

                        # Save the text to a file
                        sinen.save_file(texts,hostname)

                    except Exception as e:
                        print(f"An error occurred while processing {full_link}: {e}")
                        continue
            except Exception as e:
                        print(f"An error occurred while processing {relative_link}: {e}")
                        continue
                
        # Save the extracted text to a file using the hostname as the file name
        sinen.save_file(texts, hostname) # This line saves the extracted texts in a file using the hostname as the file name and the output file type specified in the Sinen class.
        print("\n---The texts extracted from " + hostname + " have been saved.---\n") 
        # Clear the texts list
        texts.clear()     
        
        # Dictionary of options for the user to choose from
        option_list = {1: "Back",0: "Exit"} 
        # Prompts the user to choose an option from the dictionary
        choice = option_menu(option_list,"\nPlease choose an option: ")  
        if(choice == 1):
            main()
        elif(choice == 0):
            # Print a goodbye message and exit the program
            print("Thank you for using our program!\n")
            # Exit the program
            exit() 
            
if __name__ == "__main__":
    main()