"""creates a dictionary from the folder"""

from datetime import datetime
import os
from dotenv import load_dotenv
from gensim import corpora
from tqdm import tqdm
import preprocessing

load_dotenv()

folder = os.getenv('TEXT_FOLDER', 'texts')
files = os.listdir(folder)


def createDictionary():

    english = False

    def generator():
        print('creating dictionary from folder: ' + folder)
        for i in tqdm(range(len(files))):
            with open(os.path.join(folder, files[i])) as file:
                tokens = file.readline().split(' ')
                yield tokens

    print(datetime.now(), 'starting dictionary')

    dictionary = corpora.Dictionary(generator())

    filename = os.getenv('RAW_DICT_FILENAME', '')

    print(datetime.now(), 'saving dictionary to: ' + filename)
    dictionary.save(filename)
    dictionary.save_as_text(filename + ".dbg")


def getDictionary():
    return dictionary

if __name__ == "__main__":
    createDictionary()
