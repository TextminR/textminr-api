
from datetime import datetime
import os


from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

folder = os.getenv("TEXT_FOLDER", "")
rawFolder = os.getenv("RAW_TEXT_FOLDER", "")
files = os.listdir(folder)

id2title = {}
id2topics = {}

# build id2title
print(f"reading all document titles...", datetime.now())
for filename in tqdm(files):
    with open(os.path.join(rawFolder, filename)) as file:
        title = file.readline().strip()
        id2title[filename] = title

print("finished setup", datetime.now())
