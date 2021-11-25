# keycloud

KeyCloud is a simple keyphrase extractor and a keyphrase-cloud generator.

Is suitable to work with text corpora. It identifies each document keyphrases and generates a keyphrase-cloud that informs about the most scored keyphrases within the corpus.

KeyCloud contains several models:

tf-idf: A  baseline model that uses tf-idf information to score the keyphrase candidates.

H1: A model based in simple heuristic, based in each document features only, to score the keyphrase candidates. (working paper)

H2: The same as H1 with idf information added.

EmbedRank s2v. A model based in the [EmbedRank model](https://arxiv.org/abs/1801.04470). 

If you use it please cite the following paper:

Won, Miguel, Bruno Martins, and Filipa Raimundo. *Automatic extraction of relevant keyphrases for the study of issue competition.* Proceedings of the 20th international conference on computational linguistics and intelligent text processing, Berkeley, La Rochelle, France. 2019.
