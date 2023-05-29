from math import log10
import re
from .pagerank_weighted import pagerank_weighted_scipy as _pagerank
from .preprocessing.textcleaner import clean_text_by_sentences as _clean_text_by_sentences
from .commons import build_graph as _build_graph
from .commons import remove_unreachable_nodes as _remove_unreachable_nodes

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

    common_word_count = _count_common_words(words_sentence_one, words_sentence_two)

    log_s1 = log10(len(words_sentence_one))
    log_s2 = log10(len(words_sentence_two))

    if log_s1 + log_s2 == 0:
        return 0

    return common_word_count / (log_s1 + log_s2)


def _count_common_words(words_sentence_one, words_sentence_two):
    return len(set(words_sentence_one) & set(words_sentence_two))  # 交集，单词复现数量


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



def summarize(text_list, similarity, device):

    # 句子列表
    sentences = _clean_text_by_sentences(text_list)
    # 计算每对节点的相似系数。
    graph = _build_graph([sentence.token for sentence in sentences])
    _set_graph_edge_weights(graph)

    # 删除所有边权重等于零的所有节点。
    # _remove_unreachable_nodes(graph)

    # PageRank算法对token进行排序
    scores = _pagerank(graph, similarity, device)

    return scores


