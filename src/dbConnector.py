import os
from typing import Any
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv()

SCROLL_SIZE = 5
OUTPUT_FOLDER = os.getenv("RAW_TEXT_FOLDER", "")


IP = os.getenv("ELASTIC_IP")
USER = os.getenv("ELASTIC_USER")
PASSWORD = os.getenv("ELASTIC_PASSWORD")
INDEX = "texts"

es = Elasticsearch(
    [IP],
    http_auth=(USER, PASSWORD),
    verify_certs=False,
)

if es.ping():
    print("Connected to Elasticsearch")
else:
    print("Could not connect to Elasticsearch")
    raise ConnectionError()


def scrollWithQuery(query) -> dict[str, Any]:
    out = {}

    data = es.search(index=INDEX, body=query, size=100, scroll="2m")
    scroll_id = data["_scroll_id"]
    hits = data["hits"]["hits"]

    while hits:
        for h in hits:
            out[h["_id"]] = h

        # continue scrolling
        data = es.scroll(scroll_id=scroll_id, scroll="2m")
        scroll_id = data["_scroll_id"]
        hits = data["hits"]["hits"]

    return out


def getDocument(id: str):
    return es.get(index=INDEX, id=id, source_excludes=["embeddings", "text"])


def downloadTexts():

    def extractText(textArr):
        out = ""
        for part in textArr:
            out += part["part"]
        return out

    query = {"_source": {"excludes": ["embeddings"]}}

    hits = scrollWithQuery(query)
    for hit in hits.values():
        source = hit["_source"]

        id = hit["_id"]
        title = source["title"]
        author = source["author"]
        textArr = source["text"]
        print(f"{title}")
        textStr = extractText(textArr)
        with open(os.path.join(OUTPUT_FOLDER, id), "w") as file:
            file.write(f"{title}\n{author}\n{textStr}")
