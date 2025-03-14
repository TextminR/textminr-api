import os

from dotenv import load_dotenv
from gensim.parsing.preprocessing import (
    preprocess_string,
    strip_punctuation,
    strip_numeric,
    remove_stopwords,
    strip_short,
    stem_text,
    strip_tags,
    strip_multiple_whitespaces,
    strip_non_alphanum,
    lower_to_unicode,
)
from tqdm import tqdm
from nltk.stem.snowball import SnowballStemmer
import nltk

load_dotenv()

folder = os.getenv("TEXT_FOLDER", "")
rawFolder = os.getenv("RAW_TEXT_FOLDER", "")

filters = [
    strip_multiple_whitespaces,
    # strip_punctuation,  # Remove punctuation
    strip_non_alphanum,
    strip_numeric,  # Remove numbers
    remove_stopwords,  # Remove stopwords
    strip_tags,
    strip_short,  # Remove words shorter than 3 characters
    lower_to_unicode
]

import spacy

nltk.download("stopwords")
germanStopwords = nltk.corpus.stopwords.words("german")
snowball = SnowballStemmer("german")

spc = spacy.load("de_core_news_md")
spc.max_length = 1_000_000_000


def oldStringToTokens(text, english=False):
    # tokenization + apply filters
    tokens = preprocess_string(text, filters)
    # stemming
    stemmed_tokens = (
        [stem_text(token) for token in tokens]
        if english
        else [snowball.stem(token) for token in tokens]
    )
    # german stopwords
    if not english:
        stemmed_tokens = [w for w in stemmed_tokens if w not in germanStopwords]
    return stemmed_tokens


def stringToTokens(text, english=False):
    # lemmatization
    doc = spc(text, disable=['ner', 'parser'])
    tokens = [x.lemma_ for x in doc]
    # tokenization
    tokens = [preprocess_string(s, filters) for s in tokens]
    # convert back to array of strings
    tokens = [x[0] for x in tokens if len(x) ]
    # german stopwords
    if not english:
        tokens = [w for w in tokens if w not in germanStopwords]

    return tokens


def convertFolder(folder, out, conversionFunction):
    files = os.listdir(folder)
    os.makedirs(out, exist_ok=True)
    for fileName in tqdm(files):
        with open(os.path.join(folder, fileName)) as fileIn:
            lines = " ".join(fileIn.readlines())
            tokens = conversionFunction(lines)
            outString = " ".join(tokens)
            with open(os.path.join(out, fileName), "w") as fileOut:
                fileOut.write(outString)


def iterateFolder(name):
    files = os.listdir(name)
    for fn in tqdm(files):
        with open(os.path.join(name, fn)) as file:
            file.readlines()


if __name__ == "__main__":
    convertFolder(rawFolder, folder, stringToTokens)
    # while True:
    #     text = input("enter string:")
    #     # print(stringToTokens(text))
    #     # print(newStringToTokens(text))
