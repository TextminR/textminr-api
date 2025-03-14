import os
from corpus import FolderCorpus
from gensim import corpora
import time
from gensim.models import LdaModel, HdpModel, LsiModel
import logging
import warnings
from dotenv import load_dotenv

load_dotenv()


dictionary = corpora.Dictionary.load(os.getenv("DICT_FILENAME", ""))

c = FolderCorpus(dictionary)


# Enable logging to monitor progress
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)
warnings.filterwarnings("ignore", category=DeprecationWarning)


lda_model = None
lsi_model = None


# Initialize topic modeling algorithms
def lda_topic_modeling(corpus, dictionary, num_topics=10):
    # lda_model = LdaModel(
    #     corpus=corpus,
    #     id2word=dictionary,
    #     num_topics=num_topics,
    #     random_state=100,
    #     update_every=1,
    #     chunksize=100,
    #     passes=10,
    #     alpha="auto",
    #     per_word_topics=True,
    # )
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        chunksize=(int(os.getenv("LDA_CHUNKSIZE", 2000))),
        alpha="auto",
        eta="auto",
        iterations=(int(os.getenv("LDA_ITERATIONS", 50))),
        num_topics=num_topics,
        passes=int(os.getenv("LDA_PASSES", 20)),
        eval_every=None,
    )
    return lda_model


def hdp_topic_modeling(corpus, dictionary):
    hdp_model = HdpModel(corpus=corpus, id2word=dictionary)
    return hdp_model


def lsi_topic_modeling(corpus, dictionary, num_topics=10):
    lsi_model = LsiModel(
        corpus=corpus,
        id2word=dictionary,
        chunksize=(int(os.getenv("LSI_CHUNKSIZE", 20000))),
        onepass=False,
        power_iters=int(os.getenv("LSI_POWER_ITERS", 2)),
        num_topics=num_topics,
    )
    return lsi_model


# Set number of topics (for models that need it)
num_topics = 50

# Create models
def model():
    global lda_model
    global lsi_model
    # hdp_model = hdp_topic_modeling(c, dictionary)
    # hdp_model.save("hdp.model")
    num_topics = int(os.getenv("NUM_TOPICS", 0))
    start_time = time.time()
    print("modeling lda")
    lda_model = lda_topic_modeling(c, dictionary, num_topics)
    ldaName = os.getenv("LDA_FILENAME", "lda.model")
    lda_model.save(ldaName)
    end_time = time.time()
    elapsed_time = end_time - start_time

    with open(ldaName + ".runtime", "w") as file:
        file.write(f"{elapsed_time}")

    print("modeling lsi")
    startTime = time.time()
    lsi_model = lsi_topic_modeling(c, dictionary, num_topics)
    lsiName = os.getenv("LSI_FILENAME", "lsi.model")
    lsi_model.save(lsiName)
    endTime = time.time()
    elapsed = endTime - startTime
    with open(lsiName + ".runtime", "w") as file:
        file.write(f"{elapsed}")

    # hdp_model = hdp_topic_modeling(c2, dictionary)
    #
    # for i, model in enumerate([lda_model, lsi_model]):
    #     print(f"model {i}")
    #     print("topics")
    #
    #     for idx, topic in model.print_topics(-1):
    #         print(f"Topic {idx}: {topic}\n")
    #
    #     print("in sample test:")
    #     for j, a in enumerate(c):
    #         print(j, files[j], model[a])
    #
    #     input("next?")


model()

# print("\nHDP Topics:")
# for idx, topic in enumerate(hdp_model.print_topics(-1)[:num_topics]):
#     print(f"Topic {idx}: {topic}\n")
#
# print("\nLSI Topics:")
# for idx, topic in lsi_model.print_topics(-1):
#     print(f"Topic {idx}: {topic}\n")
#
