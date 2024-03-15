import nltk
from nltk.corpus import wordnet
import random
import time


nltk.download('punkt')
nltk.download('brown')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

REPAIR_CUES = [("no", 1), ("no wait", 2), ("no sorry", 2), ("I meant", 2), ("I mean", 2), ("sorry", 1),
               ("I am sorry", 3), ("no i meant to say", 5), ("actually no", 2), ("wait", 1),
               ("well I actually mean", 4), ("well I actually meant", 4),
               ("wait a minute", 3),
               ("no wait a minute", 4)]


'''Connective words and phrased can be used to connect sentences in a logical flow for different purposes
etc. addition, contrast, emphasis, illustration, comparison, etc.
To create restarts, pair of sentences are splitted and combined which can accidentaly lead to a fluent
sequence, if the splitted sentence is a connective word or phrase.
'''
CONNECTIVE_STOPWORDS = ('and', 'also', 'as well as', 'further', 'furthermore', 'too', 'moreover',
                        'in addition', 'besides', 'next', 'finally', 'last', 'lastly', 'at last', 'now', 'subsequently', 'then',
                        'when', 'soon', 'thereafter', 'after a short time', 'the next week', 'the next month',
                        'the next day', 'a minute later', 'in the meantime', 'meanwhile', 'on the following day',
                        'at length', 'ultimately', 'presently', 'first', 'second', 'finally', 'hence', 'next', 'then',
                        'from here on', 'to begin with', 'last of all', 'after', 'before', 'as soon as', 'in the end',
                        'gradually', 'above', 'behind', 'below', 'beyond', 'here', 'there', 'on the other side',
                        'in the background', 'directly ahead', 'at this point',
                        'for example', 'to illustrate', 'for instance', 'to be specific', 'such as', 'furthermore',
                        'just as important', 'similarly', 'in the same way',
                        'as a result', 'hence', 'so', 'accordingly', 'as a consequence',
                        'consequently', 'thus', 'since', 'therefore', 'for this reason', 'because of this',
                        'to this end', 'for this purpose', 'with this in mind', 'for this reason', 'for these reasons',
                        'like', 'in the same manner', 'in the same way', 'as so', 'similarly', 'but', 'in contrast',
                        'conversely', 'however', 'still', 'nevertheless', 'but clearly'
                        'nonetheless', 'yet', 'and yet', 'on the other hand', 'on the contrary', 'or', 'or that'
                        'in spite of this', 'actually', 'in fact', 'you know', 'and then'
                        'in summary', 'to sum up', 'to repeat', 'briefly', 'in short', 'finally', 'on the whole',
                        'therefore', 'as I have said',  'so what', 'well', '''it's''', ''''that's''', 'i guess'
                        'in conclusion', 'as you can see', 'i have', 'but that', 'but then', 'so that', 'even', 'so but', 'now'
                        ,'and there', 'no', 'so what', 'probably', 'and probably', 'but well', 'i mean',
                        'and so', 'and then', 'and this', 'and that', '''that 's''', 'but just', 'but now', 'i know', 'so when',
                        'so what', 'just', 'well good', 'well if', 'well and', 'well that', 'well there','yeah well', 'well actually', 'and well',
                        'because', 'because then', 'because now', 'because that', 'true yeah', 'true', 'well then',
                        'basically', 'so basically', 'and also', 'well still', 'still', 'yeah there', 'but anyway',
                        'anyway', 'also somewhat', 'somewhat', 'well usually', 'usually', 'well what', 'i think',
                        'so there', 'so this', 'so that', 'yeah but', 'i guess', 'well sometimes', 'really', 'always',
                        'yeah always', 'well just'
                        )


none_tuple = (None, None, None, None, None)


def extract_syns_ants(word, pos):
    synsets = wordnet.synsets(word, pos=getattr(wordnet, pos))

    synonyms = [lemma.name() for synset in synsets for lemma in synset.lemmas()]
    antonyms = [lemma.antonyms()[0].name() for synset in synsets for lemma in synset.lemmas() if lemma.antonyms()]

    return synonyms, antonyms


def extract_hyponyms(word, pos):

    # Extract synsets of word
    synsets = wordnet.synsets(word, pos=getattr(wordnet, pos))

    if len(synsets) > 0:
        candidate_synsets = [synset for synset in synsets]

        # Extract all the hypernyms of each synset
        hypernyms = [synset.hypernyms()[0] for synset in candidate_synsets if len(synset.hypernyms()) > 0]

        if 0 < len(hypernyms) < 10:

            hyponyms = []
            for hypernym in hypernyms:
                print(hypernym.name())
                if hypernym.name() == 'restrain.v.01':
                    return None
                elif hypernym.name() != 'restrain.v.01' or hypernym.name() != 'inhibit.v.04':
                    # Find hyponyms with the same hypernym
                    candidate_hyponyms = list(
                        set([w for s in hypernym.closure(lambda s: s.hyponyms()) for w in s.lemma_names()]))
                    # Keep up to 20 hyponyms, randomly, for each synset
                    if len(candidate_hyponyms) >= 10:
                        candidate_hyponyms = random.sample(candidate_hyponyms, 10)

                    for candidate_hyponym in candidate_hyponyms:
                        hyponyms.append(candidate_hyponym)

            return hyponyms


def are_same(lst):
    return all(x.lower() == lst[0].lower() for x in lst)


def extract_pos_format(candidate_pos):
    if candidate_pos is not None:
        if candidate_pos == 'NOUN':
            possible_tag_format = ['NN', 'NNS']
        elif candidate_pos == 'VERB':
            possible_tag_format = ['VB', 'VBD', 'VBG', 'VBN', 'VBZ', 'VBP']
        elif candidate_pos == "ADJ":
            possible_tag_format = ['JJ', 'JJR', 'JJS']
        else:
            print("You need to specify a valid POS identifier. Supported POS: NOUN, VERB or ADJ")
            return None
    else:
        possible_tag_format = ['NN', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBZ', 'VBP', 'JJ', 'JJR', 'JJS']

    return possible_tag_format


def revert_pos_format(pos):
    if pos in ['NN', 'NNS']:
        candidate_pos = 'NOUN'
    elif pos in ['VB', 'VBD', 'VBG', 'VBN', 'VBZ', 'VBP']:
        candidate_pos = 'VERB'
    elif pos in ['JJ', 'JJR', 'JJS']:
        candidate_pos = 'ADJ'
    else:
        candidate_pos = ''

    return candidate_pos


def check_disfluency_validity(existing_annotations):

    if existing_annotations is None:
        print("Warning! You try to pass a disfluent sentence without token-based annotations. Please specify "
              "token-based annotations")
        return False
    if all(i == "D" for i in existing_annotations):
        print("Warning! You try to pass a disfluent sentence which consists only of disfluent tokens."
              "We need at least one fluent token to generate a disfluency.")
        return False
    else:
        return True
