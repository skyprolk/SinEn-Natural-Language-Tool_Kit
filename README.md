# SinEn - Sinhala Language Processing Toolkit

SinEn is an open-source natural language processing toolkit for Sinhala language, written in Python. The toolkit provides several tools for Sinhala language processing, including:

## SinEn Bad Words Detector
A machine learning model that can detect offensive or inappropriate words in Singlish text.

![SinEn Bad Words Detector](https://github.com/skyprolk/SinEn-Natural-Language-Tool_Kit/blob/main/img/1.png)

## SinEn Converter
A software tool that can convert Sinhala script to English transliteration.

![SinEn Converter](https://github.com/skyprolk/SinEn-Natural-Language-Tool_Kit/blob/main/img/2.png)

## SinEn Stemmer
A program that provides stemming functionality for Singlish words, reducing them to their base or root form.

![SinEn Stemmer](https://github.com/skyprolk/SinEn-Natural-Language-Tool_Kit/blob/main/img/3.png)

## SinEn Tokenizer
A program that can split Singlish sentences into individual words or tokens.

![SinEn Tokenizer](https://github.com/skyprolk/SinEn-Natural-Language-Tool_Kit/blob/main/img/4.png)

_In addition to these tools, the project also includes supporting tools for collecting and processing Sinhala text, including a **"Bag of Words Collector"** and **"Bag of Words Maker"**._

## Installation

To install SinEn, clone the repository and install the required packages using pip:
> git clone https://github.com/skyprolk/SinEn-Natural-Language-Tool_Kit.git <br />
> pip install art <br />
> pip install pygtrie <br />


## Usage

To use the SinEn toolkit, import the necessary modules in your Python code:

```python
# Import the SinEnStemmer class from the SinEn_Stemmer module
from sinen_stemmer import SinEnStemmer

# Import the SinEnTokenizer class from the SinEn_Tokenizer module
from sinen_tokenizer import SinEnTokenizer

# Create an instance of the SinEnStemmer class
stemmer = SinEnStemmer()

# Create an instance of the SinEnTokenizer class for splitting the text
tokenizer = SinEnTokenizer()

# Tokenize the text using the SinEnTokenizer object
text = "Mata rupiyal 100k wage mudalak denna puluwanda?"
text = tokenizer.tokenize(text)

# Stem each word in the tokenized text using the SinEnStemmer object
stemmed_words = []
for word in text:
    # Call the stem method to reduce the word to its base form
    stemmed_word = stemmer.stem(word)[0]
    stemmed_words.append(stemmed_word)

# Print the stemmed words to the console
print(stemmed_words) # Output : ['ma', 'rupiyal', 'mudala', 'dena', 'puluwan']
```
## Contributing

We welcome contributions to the SinEn project. If you would like to contribute, please open a pull request with your changes.

## License

SinEn is released under the Apache-2.0 license. **See [LICENSE](https://github.com/skyprolk/SinEn-Natural-Language-Tool_Kit/blob/main/LICENSE) for more information.**

## Credits
Developed & Scripted by _**#KNOiT**_ with _**Sky Production**_
