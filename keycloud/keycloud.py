import nltk
import numpy,scipy
from collections import Counter,defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

from .utils import *

class KeyCloud():
    def __init__(self,lan = 'en'):

        self.lan = lan

        if lan == 'en':
            from nltk.stem.porter import PorterStemmer
            self.stemer = PorterStemmer()
            self.stopwords = nltk.corpus.stopwords.words('english')

        elif lan == 'pt':
            from nltk.stem import RSLPStemmer
            self.stemer = RSLPStemmer()
            self.stopwords = nltk.corpus.stopwords.words('portuguese')
            self.stopwords = [w for w in self.stopwords if w not in ['de','do','da','dos','das']]
        else:
            raise Exception('Language not suported')



    def read_docs(self,docs_pos):
        self.docs_pos = docs_pos

    def stem_key(self,keyphrase):
        """Return a stemmed keyphrase."""
        return '_'.join(['-'.join([self.stemer.stem(t2) for t2 in  t.split('-') if len(t2)>=1]) for t in keyphrase.split('_')])


    def calc_candidates(self,min_doc = 1,min_corpus = 1,stop_candidates=[],
                            grammar = """ NP: {<JJ>* <NN.*>+}"""):


        """Calc raw candidates
        ----------
        min : minimum occurrences within a single doc
        min_docs : minimum occurrences in the corpus
        stop_candidates : custom stop candidates
        grammar : pos grammar to extract candidates
        """
        # custom stop candidates uniformization
        stop_candidates = ['_'.join(cand.split()).lower() for cand in stop_candidates]
        stop_candidates_st = [self.stem_key(cand) for cand in stop_candidates]

        doc_size = []
        chunker = nltk.RegexpParser(grammar) # chunker
        stem2reg_temp = defaultdict(list) #dictionary of stem to regular form
        corpus_counter = defaultdict(int) #dictionary to count # of candidates in all documents
        self.tf = defaultdict(lambda: defaultdict(int)) # main candidates counter per document
        temp_docs_candidates = []
        for i,doc in enumerate(self.docs_pos):

            sents = doc.split('_|_')

            doc_size.append(len([x for sent in sents for x in sent.split()]))
            doc_cand = []
            for sent in sents:
                tk_tags = [(x.split('__')[0].lower(),x.split('__')[1]) for x in sent.split()]
                sent_new = []
                for tk,tag in tk_tags:
                    if tk.isdigit():
                        sent_new.append((tk,'PP'))
                    elif asphorbidden(tk):
                        sent_new.append((tk,'PP'))
                    elif tk in self.stopwords:
                        sent_new.append((tk,'PP'))
                    elif len(tk) < 2:
                        sent_new.append((tk,'PP'))
                    else:
                        sent_new.append((tk,tag))
                if sent_new:
                    candidates = candidate_chunks(text_pos = sent_new,
                                                    chunker = chunker)
                    sent_cand = []
                    for cand in candidates:
                        candidate = '_'.join([x for x,_ in cand])
                        candidate_st = self.stem_key(candidate)
                        if candidate in stop_candidates:
                            continue
                        if candidate_st in stop_candidates_st:
                            continue
                        stem2reg_temp[candidate_st].append(candidate)
                        if len(cand) > 1:
                            for x,_ in cand:
                                stem2reg_temp[self.stem_key(x)].append(x)
                        self.tf[i][candidate_st] += 1
                        sent_cand.append(candidate_st)
                    if sent_cand:
                        doc_cand.append(sent_cand)
            for cand,count in self.tf[i].items():
                corpus_counter[cand] += count
            temp_docs_candidates.append(doc_cand)

        self.candidates = defaultdict(int) # candidate dictionary
        self.docs_candidates = []
        self.docs_unigrams = []
        for i,doc_cand in enumerate(temp_docs_candidates):
            unigrams = []
            temp = []
            for sent in doc_cand:
                sent_new = [cand for cand in sent if self.tf[i][cand] >= min_doc and corpus_counter[cand] >= min_corpus]
                if sent_new:
                    for cand in sent_new:
                        self.candidates[cand] = 1
                        for tk in cand.split('_'):
                            unigrams.append(tk)
                    temp.append(' '.join(sent_new).strip())
            doc_cand = ' _|_ '.join(temp)
            self.docs_candidates.append(doc_cand)
            self.docs_unigrams.append(' '.join(unigrams))

        self.candidates = list(self.candidates.keys())

        self.stem2reg = {}
        for cand,listas in stem2reg_temp.items():
            self.stem2reg[cand] = sorted(Counter(listas).items(), key = lambda x:x[1],reverse = True)[0][0]
        self.docs_size =  doc_size


    def scores_tfidf(self,n_keys = 10):
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\S\S+")
        tfidf = vectorizer.fit_transform(self.docs_candidates).toarray()
        vocabulary = vectorizer.vocabulary_
        predicted_keys = []
        for i,doc in enumerate(self.docs_candidates):
            candidates = Counter(doc.split())
            del(candidates['_|_'])
            scores = []
            for cand in candidates.keys():
                try:
                    scores.append((cand,tfidf[i,vocabulary[cand]]))
                except KeyError:
                    scores.append((cand,0))

            if  n_keys == 'dynamic':
                n_keys = int(2.5*numpy.log(self.docs_size[i]))
                scores = sorted(scores,key = lambda x:x[1],reverse=True)[:n_keys]
            elif n_keys == 'all':
                scores = sorted(scores,key = lambda x:x[1],reverse=True)
            else:
                scores = sorted(scores,key = lambda x:x[1],reverse=True)[:n_keys]
            predicted_keys.append(scores)
        return predicted_keys


    def scores_h1(self,n_keys = 10):
        import warnings
        vectorizer = CountVectorizer(token_pattern=r"(?u)\S\S+")
        count_tokens = vectorizer.fit_transform(self.docs_unigrams).toarray()
        vocabulary = vectorizer.vocabulary_
        voc_size = len(vocabulary.items())
        n_docs = len(self.docs_candidates)
        tf_norm = numpy.zeros(count_tokens.shape)
        for i in range(n_docs):
            try:
                tf_norm[i,:] = count_tokens[i,:]/count_tokens[i,:].sum()
            except Warning:
                print("Warning! ",i,count_tokens[i,:].sum())

        idf = {}
        for i in range(voc_size):
            temp = numpy.count_nonzero(count_tokens[:,i])
            idf[i] = numpy.log(n_docs/(1+temp))

        predicted_keys = []
        for i,doc in enumerate(self.docs_candidates):
            cand_pos = defaultdict(int)
            pos = 0
            for sent in doc.split('_|_'):
                for cand in sent.split():
                    pos += 1
                    if not cand_pos[cand]:
                        cand_pos[cand] = pos
            pos_max = pos+1 # to avoid last candidate having 0 score
            score_pos = {cand:(1-pos/pos_max)**self.tf[i][cand] for cand,pos in cand_pos.items()}
            scores = []
            for cand in score_pos.keys():
                length_sc = 0
                sc = 0
                for tk in cand.split('_'):
                    length_sc += 1
                    try:
                        sc += tf_norm[i,vocabulary[tk]]
                    except KeyError:
                        continue
                if length_sc > 1:
                    length_sc = 2
                try:
                    scores.append((cand,sc*score_pos[cand]*length_sc))
                except KeyError:

                    scores.append((cand,0))

            if  n_keys == 'dynamic':
                n_keys = int(2.5*numpy.log(self.docs_size[i]))
                scores = sorted(scores,key = lambda x:x[1],reverse=True)[:n_keys]
            elif n_keys == 'all':
                scores = sorted(scores,key = lambda x:x[1],reverse=True)
            else:
                scores = sorted(scores,key = lambda x:x[1],reverse=True)[:n_keys]

            predicted_keys.append(scores)
        return predicted_keys

    def scores_h2(self,n_keys = 10):
        vectorizer = CountVectorizer(token_pattern=r"(?u)\S\S+")
        count_tokens = vectorizer.fit_transform(self.docs_unigrams).toarray()
        vocabulary = vectorizer.vocabulary_
        voc_size = len(vocabulary.items())
        n_docs = len(self.docs_candidates)
        tf_norm = numpy.zeros(count_tokens.shape)
        for i in range(n_docs):
            try:
                tf_norm[i,:] = count_tokens[i,:]/count_tokens[i,:].sum()
            except Warning:
                print("Warning! ",i)
        idf = {}
        for i in range(voc_size):
            temp = numpy.count_nonzero(count_tokens[:,i])
            idf[i] = numpy.log(n_docs/(1+temp))

        predicted_keys = []
        for i,doc in enumerate(self.docs_candidates):
            cand_pos = defaultdict(int)
            pos = 0
            for sent in doc.split('_|_'):
                for cand in sent.split():
                    pos += 1
                    if not cand_pos[cand]:
                        cand_pos[cand] = pos
            pos_max = pos+1 # to avoid last candidate having 0 score
            score_pos = {cand:(1-pos/pos_max)**self.tf[i][cand] for cand,pos in cand_pos.items()}
            scores = []
            for cand in score_pos.keys():
                length_sc = 0
                sc = 0
                for tk in cand.split('_'):
                    length_sc += 1
                    try:
                        sc += tf_norm[i,vocabulary[tk]] * idf[vocabulary[tk]]
                    except KeyError:
                        continue
                if length_sc > 1:
                    length_sc = 2
                try:
                    scores.append((cand,sc*score_pos[cand]*length_sc))
                except KeyError:

                    scores.append((cand,0))

            if  n_keys == 'dynamic':
                n_keys = int(2.5*numpy.log(self.docs_size[i]))
                scores = sorted(scores,key = lambda x:x[1],reverse=True)[:n_keys]
            elif n_keys == 'all':
                scores = sorted(scores,key = lambda x:x[1],reverse=True)
            else:
                scores = sorted(scores,key = lambda x:x[1],reverse=True)[:n_keys]

            predicted_keys.append(scores)
        return predicted_keys


    def calc_scores_emb(self,n_keys = 10,sent2vec_model = ''):
        import sent2vec
        model = sent2vec.Sent2vecModel()
        model.load_model(sent2vec_model)

        predicted_keys = []
        for i,doc in enumerate(self.docs_candidates):
            sents = doc.split('_|_')
            unigrams = [tk for sent in sents for cand in sent.split() for tk in cand.split('_')]
            singe_doc = ' '.join([self.stem2reg[tk] for tk in unigrams])
            emb_doc = model.embed_sentence(singe_doc)

            candidates = Counter(doc.split())
            del(candidates['_|_'])

            emb_cand = {}
            for cand in candidates.keys():
                cand = self.stem2reg[cand]
                cand = ' '.join([tk for tk in cand.split('_')])
                emb_cand[self.stem_key('_'.join(cand.split()))] = model.embed_sentence(cand)
            distances = []
            for cand,vec_cand in emb_cand.items():
                d = scipy.spatial.distance.cosine(vec_cand,emb_doc)
                distances.append((cand,d))

            if  n_keys == 'dynamic':
                n_keys = int(2.5*numpy.log(self.docs_size[i]))
                scores = sorted(distances,key = lambda x:x[1],reverse=False)[:n_keys]
            elif n_keys == 'all':
                scores = sorted(distances,key = lambda x:x[1],reverse=False)
            else:
                scores = sorted(distances,key = lambda x:x[1],reverse=False)[:n_keys]


            predicted_keys.append(scores)
        return predicted_keys


    def keys_weights(self,n_keys):
        predicted_keys = self.scores_h2(n_keys = 'all')
        weights = defaultdict(int)
        for rank in predicted_keys:
            for cand,r in rank:
                weights[self.stem2reg[cand]] += r
        scores = sorted(weights.items(), key = lambda x:x[1],reverse = True)[:n_keys]
        maximo = scores[0][1]
        return {k:sc/maximo for k,sc in scores}


    def generate_keycloud(self,output_file,n_keys=25,**kwargs):
        from wordcloud import WordCloud
        # from matplotlib import cm

        keys_weights = self.keys_weights(n_keys)

        # color_rescale = {k:(w*0.5+0.5) for k,w in keys_weights.items()}
        # def colof_f(*args, **kwargs):
        #     word = args[0]
        #     w = color_rescale[word]
        #     color = list(cm.get_cmap('YlOrBr')(w))[:3]
        #     color = tuple([int(x*255) for x in color])
        #     return color
        # wc = WordCloud(**kwargs,color_func=colof_f).generate_from_frequencies(keys_weights)
        wc = WordCloud(
                **kwargs,
                color_func=lambda *args, **kwargs: (114,40,5)
                ).generate_from_frequencies(keys_weights)

        wc.to_file(output_file)
        return keys_weights
