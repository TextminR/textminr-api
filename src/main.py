from collections import defaultdict
from dotenv import load_dotenv
from datetime import datetime
import os
import pandas as pd
import random

from cache import *
from routers.documents import getDocumentTopics
from dictionary import dictionary

from fastapi import FastAPI, Query

import dbConnector as db
from gensim.models import LdaModel, HdpModel, LsiModel

import preprocessing
from sentiment import getSentimentValue

load_dotenv()

app = FastAPI()


lda = LdaModel.load(os.getenv("LDA_FILENAME", ""))
lsi = LsiModel.load(os.getenv("LSI_FILENAME", ""))


def getDocumentsBetweenDates(
    startdate: datetime,
    enddate: datetime,
):
    query = {
        "_source": {
            "excludes": ["embeddings", "text"],
        },
        "query": {"range": {"year": {"gte": startdate.year, "lte": enddate.year}}},
    }
    return db.scrollWithQuery(query)


@app.get("/topicmodels/topics/{n}")
def getTopics(model: str, n: int = -1):
    if n == -1:
        result = getAllTopics(lda if model == "lda" else lsi, n)
        print(result)

        result2 = {
            int(topic[0]): {words[0]: float(words[1]) for words in topic[1]}
            for topic in result
        }

        return {"topics": result2}
    else:
        result = getWordsForTopic(lda if model == "lda" else lsi, n)
        print(result)
        result2 = {words[0]: float(words[1]) for words in result}
        return result2


@app.get("/topicmodels/count")
def getTopicCount(model: str):
    model = lda if model == "lda" else lsi
    return model.num_topics


@app.get("/documents")
def getDocuments():
    return id2title


@app.get("/documents/info/{id}")
def getDocument(id: str):
    if id in id2title:
        # get the topics
        topics = getDocumentTopics(id)

        hit = db.getDocument(id)["_source"]

        # get sentiment
        sentiment = {
            "positive": hit["probabilities"][2],
            "neutral": hit["probabilities"][1],
            "negative": hit["probabilities"][0],
        }

        return {
            "author": hit["author"],
            "location": hit["country"],
            "topics": topics,
            "sentiment": sentiment,
        }


@app.get("/topicmodels/plot")
def getTopicPlotData():
    p = pd.read_json("plot.json")
    p["topics"] = p.index
    p = p.drop(columns=["cluster"])
    p = p.rename(columns={"topics": "topic", "Freq": "size"})
    return p.to_dict(orient="records")


def getWordsForTopic(model, n):
    return model.show_topic(n)


def getAllTopics(model, n):
    return model.show_topics(n, formatted=False)


@app.get("/topicmodels/word")
def getTopicForWord(model: str, word: str):
    print(f"get topics for word {word}")
    model = lda if model == "lda" else lsi

    tokens = preprocessing.stringToTokens(word)
    bow = dictionary.doc2bow(tokens)
    print("bow", bow)
    if len(bow):
        foundWord = dictionary[bow[0][0]]
        print("word found: " + foundWord)

        result = model[bow]
        out = {x[0]: float(x[1]) for x in result}
        print(out)
        return out
    else:
        return {}


@app.get("/topicmodels/frequency")
def getWordFrequencies(
    startdate: datetime,
    enddate: datetime,
    topics: list[int] = Query(None),
    absolute: bool = True,
):
    print(
        f"get word freq for topics {topics}, dates {startdate} - {enddate}, absolute: {absolute}",
        datetime.now(),
    )
    hits = getDocumentsBetweenDates(startdate, enddate)

    print(f"got data len: {len(hits)}", datetime.now())

    documentTopics = {id: getDocumentTopics(id) for id in hits.keys()}
    # documentTopics = {id: id2topics[id] for id in hits.keys()}

    print("calculated topics", datetime.now())

    topicFrequency = defaultdict(lambda: defaultdict(float))
    for id, doc in documentTopics.items():
        for topic in topics:
            # check if the document belongs to this topic
            if topic in doc and doc[topic] > 0.05:
                year = hits[id]["_source"]["year"]
                date = datetime(year, 1, 1)

                topicFrequency[topic][date] += 1
    if not absolute:
        # find out the total works for a year
        totalWorks = defaultdict(int)
        for hit in hits.values():
            date = hit["_source"]["year"]
            # print(f"add work {hit['_id']} for year {date}")
            totalWorks[date] += 1

        # divide by the total works
        for id, freq in topicFrequency.items():
            for date2 in freq:
                print(
                    f"topic freq is {topicFrequency[id][date2]}, total works for {date2} is {totalWorks[date2.year]}"
                )
                topicFrequency[id][date2] /= totalWorks[date2.year] + 0.1

    # sort by date
    return {topicnum: dict(sorted(d.items())) for topicnum, d in topicFrequency.items()}


# # TODO author
# name geschlech gebtrs sterbedatum (erstes letztes werk)
# topic über zeit (linieniagramm)
# histogramm wv produziert hat
# auch wo der autor


@app.get("/authors/{name}")
def getAuthorData(name: str):
    return "author " + name


@app.get("/heatmap")
def getHeatmapData(startdate: datetime, enddate: datetime, topic: int):

    print(f"get heatmap {startdate} - {enddate} with topc {topic}", datetime.now())
    hits = getDocumentsBetweenDates(startdate, enddate)
    print(f"got data size {len(hits)}", datetime.now())

    topics = {id: getDocumentTopics(id) for id in hits.keys()}
    print("got topics", datetime.now())

    return [
        {
            "title": hit["_source"]["title"],
            "author": hit["_source"]["author"],
            "latitude": hit["_source"]["coordinates"][0],
            "longitude": hit["_source"]["coordinates"][1],
            "location": hit["_source"]["country"],
            "sentiment": hit["_source"]["sentiment"],
            "probabilities": {
                "positive": hit["_source"]["probabilities"][2],
                "neutral": hit["_source"]["probabilities"][1],
                "negative": hit["_source"]["probabilities"][0],
            },
            "topic_accuracy": getDocumentTopics(id)[topic],
            "id": id,
        }
        for id, hit in hits.items()
        if topic in topics[id] and topics[id][topic] >= 0.05
    ]


@app.get("/info")
def debug(id: str):
    # out = db.scrollWithQuery(
    #     {
    #         "_source": {
    #             "excludes": ["embeddings", "text"],
    #         },
    #         "query": {"term": {"author": id}},
    #     }
    # )
    # return out
    from dbConnector import es

    return es.get(id=id, index="texts", source_excludes=["embeddings", "text"])
