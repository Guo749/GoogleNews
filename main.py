import gensim
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import ssl
from nltk.tokenize import word_tokenize

# Preprocess the paragraphs and tokenize the words
def preprocess_paragraph(paragraph):
    paragraph = paragraph.lower()
    tokens = word_tokenize(paragraph)
    print(tokens)
    return tokens

if __name__ == '__main__':
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download('punkt')

    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('/Users/wenchaoguo/Desktop/Google/09_GoogleNews/GoogleNews-vectors-negative300.bin', binary=True)

    # Input paragraphs
    # paragraph1 = "The computer wouldn't start. She banged on the side and tried again. Nothing. She lifted it up and dropped it to the table. Still nothing. She banged her closed fist against the top. It was at this moment she saw the irony of trying to fix the machine with violence.h"
    # paragraph2 = "Generating random paragraphs can be an excellent way for writers to get their creative flow going at the beginning of the day. The writer has no idea what topic the random paragraph will be about when it appears. This forces the writer to use creativity to complete one of three common writing challenges. The writer can use the paragraph as the first one of a short story and build upon it. A second option is to use the random paragraph somewhere in a short story they create. The third option is to have the random paragraph be the ending paragraph in a short story. No matter which of these challenges is undertaken, the writer is forced to use creativity to incorporate the paragraph into their writing."
    paragraph1 = "Definitely not revelent is the first paragraph"
    token1 = preprocess_paragraph(paragraph1)
    paragraph2 = "Hello world who am I"
    token2 = preprocess_paragraph(paragraph2)

    # Preprocess and convert paragraphs into vectors using Word2Vec
    vector1 = sum(word2vec_model[token] for token in token1 / len(token1))
    vector2 = sum(word2vec_model[token] for token in token2 / len(token2))

    # Calculate the cosine similarity between the two vectors
    similarity = cosine_similarity(vector1.reshape(1, -1), vector2.reshape(1, -1))[0][0]

    print("Cosine Similarity:", similarity)
