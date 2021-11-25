import nltk
import string
from itertools import chain
from scipy.spatial import distance

temp = string.punctuation+'\u2026'+'\xbb'+'\xab'+'\xba'+'“'+'’'+'”'+'‘'+'–'
phorbidden = list(temp.replace('-','').replace('_',''))
def asphorbidden(token):
    return any(x in token for x in phorbidden)


def candidate_chunks(text_pos,chunker):
    chunks = chunker.parse(text_pos)
    subtrees = chunks.subtrees(filter=lambda t: t.label() == 'NP')
    temp = [[(x,tag) for x,tag in subtree.leaves()] for subtree in subtrees]
    return temp
