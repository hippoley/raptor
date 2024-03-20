from typing import Dict, List, Set

class Node:
    """
    表示分层树结构中的一个节点。
    """

    def __init__(self, text: str, index: int, children: Set[int], embeddings) -> None:
        # 节点包含的文本内容
        self.text = text
        # 节点的索引
        self.index = index
        # 子节点的索引集合
        self.children = children
        # 节点文本的嵌入向量
        self.embeddings = embeddings


class Tree:
    """
    表示整个分层树结构。
    """

    def __init__(
        self, all_nodes, root_nodes, leaf_nodes, num_layers, layer_to_nodes
    ) -> None:
        # 包含树中所有节点的字典，键为节点索引
        self.all_nodes = all_nodes
        # 根节点的集合
        self.root_nodes = root_nodes
        # 叶节点的集合
        self.leaf_nodes = leaf_nodes
        # 树的层数
        self.num_layers = num_layers
        # 每层包含的节点，键为层次索引
        self.layer_to_nodes = layer_to_nodes
