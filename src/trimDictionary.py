from datetime import datetime
import os
from dotenv import load_dotenv
from gensim import corpora


load_dotenv()

rawFilename = os.getenv("RAW_DICT_FILENAME", "")

print(f"loading from {rawFilename}", datetime.now())

dictionary = corpora.Dictionary.load(rawFilename)
dictionary.filter_extremes(
    no_below=int(os.getenv("DICT_NO_BELOW", 1)),
    no_above=(float(os.getenv("DICT_NO_ABOVE", 1))),
)

filename = os.getenv("DICT_FILENAME", "")

print(f"saving to {filename}", datetime.now())

dictionary.save(filename)
dictionary.save_as_text(filename + ".dbg")

print("done", datetime.now())
