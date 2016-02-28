import re
from nltk import word_tokenize
from nltk.corpus import reuters
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

#10788 documents
#7769 total train documents
#3019 total test documents
#http://www.godbole.net/shantanu/pubs/multilabelsvm-pakdd04.pdf

cachedStopWords = stopwords.words("english")


def tokenize(text):
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(text))
    words = [word for word in words if word not in cachedStopWords]
    tokens = (list(map(lambda token: PorterStemmer().stem(token), words)))
    p = re.compile('[a-zA-Z]+')
    filtered_tokens = list(filter(lambda token: p.match(token) and
                                  len(token) >= min_length, tokens))
    return filtered_tokens


def tf_idf(docs):
    tfidf = TfidfVectorizer(tokenizer=tokenize, min_df=3, max_df=0.90,
                            max_features=1000, use_idf=True, sublinear_tf=True)
    tfidf.fit(docs)
    return tfidf

train_docs = []
train_cats = []
test_docs = []
test_cats = []

cats = reuters.categories()

for doc_id in reuters.fileids():
    if doc_id.startswith("train"):
        train_docs.append(reuters.raw(doc_id))
        train_cats.append(
            [cats.index(cat) for cat in reuters.categories(doc_id)])
    else:
        test_docs.append(reuters.raw(doc_id))
        test_cats.append(
            [cats.index(cat) for cat in reuters.categories(doc_id)])
representer = tf_idf(train_docs)


def get_train_set():
    return representer.transform(train_docs),\
        MultiLabelBinarizer().fit_transform(train_cats)


def get_validation_set():
    return 0


def get_test_set():
    return representer.transform(test_docs),\
        MultiLabelBinarizer().fit_transform(test_cats)
