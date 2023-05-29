from math import log10

from .pagerank_weighted import pagerank_weighted_scipy as _pagerank
from .preprocessing.textcleaner import clean_text_by_sentences as _clean_text_by_sentences
from .commons import build_graph as _build_graph
from .commons import remove_unreachable_nodes as _remove_unreachable_nodes
import re

def _get_ngrams(n, text):
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set

# import math
def _get_word_ngrams(n, sentences):

    assert len(sentences) > 0
    assert n > 0

    words = sum(sentences, [])

    return _get_ngrams(n, words)

def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def greedy_selection(doc_sent_list, abstract_sent_list):

    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    abstract = abstract_sent_list
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(s).split() for s in doc_sent_list]
    # evaluated_1grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    # reference_1grams = _get_word_ngrams(2, [abstract])
    evaluated_2grams = [_get_word_ngrams(3, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(3, [abstract])

    rouge_score = []

    for i in range(len(sents)):
        c = [i]
        # candidates_1 = [evaluated_1grams[idx] for idx in c]
        # candidates_1 = set.union(*map(set, candidates_1))
        candidates_2 = [evaluated_2grams[idx] for idx in c]
        candidates_2 = set.union(*map(set, candidates_2))
        # rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
        rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
        # rouge_score.append(rouge_1 + rouge_2)
        rouge_score.append(rouge_2)

    return rouge_score


#######


def _set_graph_edge_weights(graph):
    for sentence_1 in graph.nodes():
        for sentence_2 in graph.nodes():

            edge = (sentence_1, sentence_2)
            if sentence_1 != sentence_2 and not graph.has_edge(edge):
                similarity = _get_similarity(sentence_1, sentence_2)
                if similarity != 0:
                    graph.add_edge(edge, similarity)

    # Handles the case in which all similarities are zero.
    # The resultant summary will consist of random sentences.
    if all(graph.edge_weight(edge) == 0 for edge in graph.edges()):
        _create_valid_graph(graph)


def _create_valid_graph(graph):
    nodes = graph.nodes()

    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i == j:
                continue

            edge = (nodes[i], nodes[j])

            if graph.has_edge(edge):
                graph.del_edge(edge)

            graph.add_edge(edge, 1)


def _get_similarity(s1, s2):
    words_sentence_one = s1.split()
    words_sentence_two = s2.split()
    # n_greams = greedy_selection([s1], [s2])
    common_word_count = _count_common_words(words_sentence_one, words_sentence_two)

    log_s1 = log10(len(words_sentence_one))
    log_s2 = log10(len(words_sentence_two))

    if log_s1 + log_s2 == 0:
        return 0

    return common_word_count / (log_s1 + log_s2)
    # return n_greams[0]


def _count_common_words(words_sentence_one, words_sentence_two):
    return len(set(words_sentence_one) & set(words_sentence_two))


def _format_results(extracted_sentences, split, score):
    if score:
        return [(sentence.text, sentence.score) for sentence in extracted_sentences]
    if split:
        return [sentence.text for sentence in extracted_sentences]
    return "\n".join([sentence.text for sentence in extracted_sentences])


def _add_scores_to_sentences(sentences, scores):
    for sentence in sentences:
        # Adds the score to the object if it has one.
        if sentence.token in scores:
            sentence.score = scores[sentence.token]
        else:
            sentence.score = 0


def _get_sentences_with_word_count(sentences, words):
    """ Given a list of sentences, returns a list of sentences with a
    total word count similar to the word count provided.
    """
    word_count = 0
    selected_sentences = []
    # Loops until the word count is reached.
    for sentence in sentences:
        words_in_sentence = len(sentence.text.split())

        # Checks if the inclusion of the sentence gives a better approximation
        # to the word parameter.
        if abs(words - word_count - words_in_sentence) > abs(words - word_count):
            return selected_sentences

        selected_sentences.append(sentence)
        word_count += words_in_sentence

    return selected_sentences




def summarize(text_list, split=False, scores=False):

    # Gets a list of processed sentences.
    sentences, sentences_ = _clean_text_by_sentences(text_list)

    # Creates the graph and calculates the similarity coefficient for every pair of nodes.
    graph = _build_graph([sentence.token for sentence in sentences])
    _set_graph_edge_weights(graph)

    # Remove all nodes with all edges weights equal to zero.
    _remove_unreachable_nodes(graph)

    # PageRank cannot be run in an empty graph.
    if len(graph.nodes()) == 0:
        return [] if split else ""

    # Ranks the tokens using the PageRank algorithm. Returns dict of sentence -> score
    pagerank_scores = _pagerank(graph)

    # Adds the summa scores to the sentence objects.
    _add_scores_to_sentences(sentences_, pagerank_scores)

    # Extracts the most important sentences with the selected criterion.
    # sentences.sort(key=lambda s: s.score, reverse=True)


    # Sorts the extracted sentences by apparition order in the original text.
    # extracted_sentences.sort(key=lambda s: s.index)

    return _format_results(sentences_, split, scores)


def get_graph(text, language="english"):
    sentences = _clean_text_by_sentences(text, language)

    graph = _build_graph([sentence.token for sentence in sentences])
    _set_graph_edge_weights(graph)

    return graph
