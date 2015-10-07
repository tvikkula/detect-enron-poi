#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
from nltk import tokenize
import string

def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated) 
        
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
        
        """


    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
        ### remove punctuation
        text_string = content[1].translate(string.maketrans("", ""), string.punctuation)

        ### Stem the words
        words = tokenize.word_tokenize(text_string)
        stemmer = SnowballStemmer('english')
        stemwords = [stemmer.stem(word) for word in words]
        words = ' '.join(stemwords)
    return words
