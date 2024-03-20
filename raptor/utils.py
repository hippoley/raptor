import logging
import re
from typing import Dict, List, Set

import numpy as np
import tiktoken  # 假设是自定义的或外部的tokenization库
from scipy import spatial

# from .tree_structures import Node
from raptor.tree_structures import Node


logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


def reverse_mapping(layer_to_nodes: Dict[int, List[Node]]) -> Dict[Node, int]:
    """
    反向映射，将层到节点的映射转换为节点到层的映射。
    在构建RAPTOR树时，这有助于快速确定节点所在的层级。

    参数:
        layer_to_nodes (Dict[int, List[Node]]): 层到节点的映射字典。

    返回:
        Dict[Node, int]: 节点到层的映射字典。
    """
    node_to_layer = {}
    for layer, nodes in layer_to_nodes.items():
        for node in nodes:
            node_to_layer[node.index] = layer
    return node_to_layer


def split_text(
        text: str, tokenizer: tiktoken.get_encoding("cl100k_base"), max_tokens: int, overlap: int = 0
):
    """
    将输入文本基于tokenizer和允许的最大token数分割成更小的文本块。
    这是RAPTOR系统中文本预处理的关键步骤，确保每个文本块适合后续的嵌入和聚类处理。

    参数:
        text (str): 需要分割的文本。
        tokenizer (CustomTokenizer): 用于分割文本的tokenizer。
        max_tokens (int): 允许的最大token数。
        overlap (int, optional): 文本块之间重叠的token数。默认为0。

    返回:
        List[str]: 分割后的文本块列表。
    """
    # 使用多个分隔符将文本分割成句子
    delimiters = [".", "!", "?", "\n"]
    regex_pattern = "|".join(map(re.escape, delimiters))
    sentences = re.split(regex_pattern, text)

    # 为每个句子计算token数
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence, token_count in zip(sentences, n_tokens):
        # 如果句子为空或只包含空白，则跳过
        if not sentence.strip():
            continue

        # 如果句子过长，将其分割成更小的部分
        if token_count > max_tokens:
            sub_sentences = re.split(r"[,;:]", sentence)
            sub_token_counts = [len(tokenizer.encode(" " + sub_sentence)) for sub_sentence in sub_sentences]

            sub_chunk = []
            sub_length = 0

            for sub_sentence, sub_token_count in zip(sub_sentences, sub_token_counts):
                if sub_length + sub_token_count > max_tokens:
                    chunks.append(" ".join(sub_chunk))
                    sub_chunk = sub_chunk[-overlap:] if overlap > 0 else []
                    sub_length = sum(sub_token_counts[max(0, len(sub_chunk) - overlap):len(sub_chunk)])

                sub_chunk.append(sub_sentence)
                sub_length += sub_token_count

            if sub_chunk:
                chunks.append(" ".join(sub_chunk))

        # 如果将句子添加到当前块会超过最大tokens，开始一个新的块
        elif current_length + token_count > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:] if overlap > 0 else []
            current_length = sum(n_tokens[max(0, len(current_chunk) - overlap):len(current_chunk)])
            current_chunk.append(sentence)
            current_length += token_count

        # 否则，将句子添加到当前块
        else:
            current_chunk.append(sentence)
            current_length += token_count

    # 如果最后一个块不为空，则添加到chunks列表
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def distances_from_embeddings(query_embedding: List[float], embeddings: List[List[float]],
                              distance_metric: str = "cosine") -> List[float]:
    """
    计算查询嵌入与嵌入列表之间的距离。

    参数:
    - query_embedding (List[float]): 查询嵌入。
    - embeddings (List[List[float]]): 与查询嵌入比较的嵌入列表。
    - distance_metric (str, 可选): 用于计算距离的度量，默认为'cosine'。

    返回:
    - List[float]: 查询嵌入与嵌入列表之间的距离列表。
    """
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }

    if distance_metric not in distance_metrics:
        raise ValueError(
            f"Unsupported distance metric '{distance_metric}'. Supported metrics are: {list(distance_metrics.keys())}")

    distances = [distance_metrics[distance_metric](query_embedding, embedding) for embedding in embeddings]

    return distances


def get_node_list(node_dict: Dict[int, Node]) -> List[Node]:
    """
    将节点索引字典转换为节点列表。

    参数:
    - node_dict (Dict[int, Node]): 节点索引到节点的字典。

    返回:
    - List[Node]: 节点列表。
    """
    indices = sorted(node_dict.keys())
    node_list = [node_dict[index] for index in indices]
    return node_list


def get_embeddings(node_list: List[Node], embedding_model: str) -> List:
    """
    从节点列表中提取嵌入。

    参数:
    - node_list (List[Node]): 节点列表。
    - embedding_model (str): 被使用的嵌入模型的名称。

    返回:
    - List: 节点嵌入列表。
    """
    return [node.embeddings[embedding_model] for node in node_list]


def get_children(node_list: List[Node]) -> List[Set[int]]:
    """
    从节点列表中提取子节点。

    参数:
    - node_list (List[Node]): 节点列表。

    返回:
    - List[Set[int]]: 节点子节点索引的列表。
    """
    return [node.children for node in node_list]


def get_text(node_list: List[Node]) -> str:
    """
    通过连接节点列表中的文本生成单一文本字符串。

    参数:
    - node_list (List[Node]): 节点列表。

    返回:
    - str: 连接后的文本。
    """
    text = ""
    for node in node_list:
        text += f"{' '.join(node.text.splitlines())}"
        text += "\n\n"
    return text


def indices_of_nearest_neighbors_from_distances(distances: List[float]) -> np.ndarray:
    """
    根据距离返回最近邻居的索引，按距离升序排序。

    参数:
    - distances (List[float]): 嵌入之间的距离列表。

    返回:
    - np.ndarray: 按升序距离排序的索引数组。
    """
    return np.argsort(distances)


if __name__ == "__main__":
    # 假设的Tokenizer实例化（请替换为真实的Tokenizer实例）
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # 测试 reverse_mapping 函数
    layer_to_nodes_example = {0: [Node("Node 1", 1, set(), {})], 1: [Node("Node 2", 2, set(), {})]}
    node_to_layer = reverse_mapping(layer_to_nodes_example)
    print("测试 reverse_mapping 函数:", node_to_layer)

    # 测试 split_text 函数
    test_text = "这是一个长文本，需要被切分成多个小块。每个小块的最大token数量限制为10。"
    chunks = split_text(test_text, tokenizer, 10, 2)
    print("测试 split_text 函数，切分结果：", chunks)

    # 测试 distances_from_embeddings 函数
    query_embedding = [0.0, 1.0]
    embeddings = [[1.0, 0.0], [0.5, 0.5]]
    distances = distances_from_embeddings(query_embedding, embeddings)
    print("测试 distances_from_embeddings 函数，距离结果：", distances)

    # 测试 get_node_list 函数
    node_dict_example = {1: Node("Node 1", 1, set(), {}), 2: Node("Node 2", 2, set(), {})}
    node_list = get_node_list(node_dict_example)
    print("测试 get_node_list 函数，节点列表：", [node.text for node in node_list])

    # 测试 get_embeddings 函数
    # 此处仅模拟嵌入向量
    node_list_example = [Node("Node 1", 1, set(), {'embedding_model': [0.1, 0.2]}),
                         Node("Node 2", 2, set(), {'embedding_model': [0.2, 0.3]})]
    embeddings = get_embeddings(node_list_example, 'embedding_model')
    print("测试 get_embeddings 函数，嵌入列表：", embeddings)

    # 测试 get_children 函数
    children = get_children(node_list_example)
    print("测试 get_children 函数，子节点集合：", children)

    # 测试 get_text 函数
    text_result = get_text(node_list_example)
    print("测试 get_text 函数，文本结果：", text_result)

    # 注意：具体测试实现可能需要根据函数的实际逻辑进行调整。
