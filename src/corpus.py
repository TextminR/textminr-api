import os
from dotenv import load_dotenv

load_dotenv()


folder = os.getenv('TEXT_FOLDER', '')
files = os.listdir(folder)

maxDocuments = int(os.getenv("MAX_DOCUMENTS", 0))
if maxDocuments != 0:
    files = files[:maxDocuments]

english = False

class FolderCorpus:
    def __init__(self, dictionary) -> None:
        self.dictionary = dictionary

    def __len__(self):
        return len(files)

    def __iter__(self):
        for i in range(len(files)):
            with open(os.path.join(folder, files[i])) as file:
                tokens = file.readline().split(' ')

                yield self.dictionary.doc2bow(tokens)
