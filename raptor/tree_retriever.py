import logging
import os
from typing import Dict, List, Set

import tiktoken
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .EmbeddingModels import BaseEmbeddingModel, OpenAIEmbeddingModel
from .Retrievers import BaseRetriever
from .tree_structures import Node, Tree
from .utils import (distances_from_embeddings, get_children, get_embeddings,
                    get_node_list, get_text,
                    indices_of_nearest_neighbors_from_distances,
                    reverse_mapping)

# 配置日志
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

class TreeRetrieverConfig:
    def __init__(
        self,
        tokenizer=None,  # 分词器，默认使用tiktoken的"cl100k_base"编码
        threshold=None,  # 阈值，用于筛选相关节点，默认0.5
        top_k=None,  # 选择最相关的前k个节点，默认5
        selection_mode=None,  # 选择模式，"top_k"或"threshold"，默认为"top_k"
        context_embedding_model=None,  # 上下文嵌入模型名称，默认为"OpenAI"
        embedding_model=None,  # 嵌入模型实例，默认使用OpenAIEmbeddingModel
        num_layers=None,  # 树的层数，可选参数
        start_layer=None,  # 开始检索的层级，可选参数
    ):
        # 如果没有指定分词器，则使用tiktoken的"cl100k_base"编码
        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("cl100k_base")
        self.tokenizer = tokenizer

        # 设置阈值，默认为0.5
        if threshold is None:
            threshold = 0.5
        if not isinstance(threshold, float) or not (0 <= threshold <= 1):
            raise ValueError("threshold must be a float between 0 and 1")
        self.threshold = threshold

        # 设置top_k，默认为5
        if top_k is None:
            top_k = 5
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be an integer and at least 1")
        self.top_k = top_k

        # 设置选择模式，默认为"top_k"
        if selection_mode is None:
            selection_mode = "top_k"
        if not isinstance(selection_mode, str) or selection_mode not in [
            "top_k",
            "threshold",
        ]:
            raise ValueError(
                "selection_mode must be a string and either 'top_k' or 'threshold'"
            )
        self.selection_mode = selection_mode

        # 设置上下文嵌入模型，默认为"OpenAI"
        if context_embedding_model is None:
            context_embedding_model = "OpenAI"
        if not isinstance(context_embedding_model, str):
            raise ValueError("context_embedding_model must be a string")
        self.context_embedding_model = context_embedding_model

        # 设置嵌入模型，默认为OpenAIEmbeddingModel
        if embedding_model is None:
            embedding_model = OpenAIEmbeddingModel()
        if not isinstance(embedding_model, BaseEmbeddingModel):
            raise ValueError(
                "embedding_model must be an instance of BaseEmbeddingModel"
            )
        self.embedding_model = embedding_model

        # 设置层数，默认值可以为空
        if num_layers is not None:
            if not isinstance(num_layers, int) or num_layers < 0:
                raise ValueError("num_layers must be an integer and at least 0")
        self.num_layers = num_layers

        # 设置起始层，默认值可以为空
        if start_layer is not None:
            if not isinstance(start_layer, int) or start_layer < 0:
                raise ValueError("start_layer must be an integer and at least 0")
        self.start_layer = start_layer

    # 日志配置信息
    def log_config(self):
        config_log = """
        TreeRetrieverConfig:
            Tokenizer: {tokenizer}
            Threshold: {threshold}
            Top K: {top_k}
            Selection Mode: {selection_mode}
            Context Embedding Model: {context_embedding_model}
            Embedding Model: {embedding_model}
            Num Layers: {num_layers}
            Start Layer: {start_layer}
        """.format(
            tokenizer=self.tokenizer,
            threshold=self.threshold,
            top_k=self.top_k,
            selection_mode=self.selection_mode,
            context_embedding_model=self.context_embedding_model,
            embedding_model=self.embedding_model,
            num_layers=self.num_layers,
            start_layer=self.start_layer,
        )
        return config_log


class TreeRetriever(BaseRetriever):
    """
    TreeRetriever 类负责从树结构中检索信息。它使用嵌入模型来评估查询与树中节点的相关性，并返回最相关的信息。
    """

    def __init__(self, config, tree) -> None:
        """
        初始化 TreeRetriever 实例。

        参数:
        - config: TreeRetrieverConfig 实例，包含检索配置。
        - tree: Tree 实例，表示要检索的树结构。

        异常:
        - 如果 tree 不是 Tree 的实例，则抛出 ValueError。
        - 如果配置的 num_layers 超出树的层数加1，则抛出 ValueError。
        - 如果配置的 start_layer 超过树的层数，则抛出 ValueError。
        """
        # 验证 tree 是否为 Tree 实例
        if not isinstance(tree, Tree):
            raise ValueError("tree 必须是 Tree 的实例")

        # 验证 num_layers 和 start_layer 的合法性
        if config.num_layers is not None and config.num_layers > tree.num_layers + 1:
            raise ValueError("配置中的 num_layers 必须小于等于树的层数加1")
        if config.start_layer is not None and config.start_layer > tree.num_layers:
            raise ValueError("配置中的 start_layer 必须小于等于树的层数")

        self.tree = tree
        self.num_layers = config.num_layers if config.num_layers is not None else tree.num_layers + 1
        self.start_layer = config.start_layer if config.start_layer is not None else tree.num_layers
        if self.num_layers > self.start_layer + 1:
            raise ValueError("num_layers 必须小于等于 start_layer + 1")

        self.tokenizer = config.tokenizer
        self.top_k = config.top_k
        self.threshold = config.threshold
        self.selection_mode = config.selection_mode
        self.embedding_model = config.embedding_model
        self.context_embedding_model = config.context_embedding_model

        # 创建节点索引到层次的映射
        self.tree_node_index_to_layer = reverse_mapping(self.tree.layer_to_nodes)

        logging.info("TreeRetriever 初始化成功")

    def create_embedding(self, text: str) -> List[float]:
        """
        为给定文本生成嵌入向量。

        参数:
        - text: 要生成嵌入向量的文本。

        返回:
        - 生成的嵌入向量列表。
        """
        # 使用配置的嵌入模型为文本创建嵌入向量
        return self.embedding_model.create_embedding(text)

    # 后续方法按照上述格式进行翻译和解释

    def retrieve_information_collapse_tree(self, query: str, top_k: int, max_tokens: int) -> str:
        """
        基于查询从树中检索最相关的信息，并将树折叠以简化结果。此方法适用于大规模信息检索，其中信息被压缩成更易于处理的形式。

        参数:
        - query: 查询文本。
        - top_k: 考虑的最相关节点的数量。
        - max_tokens: 结果中的最大令牌数。

        返回:
        - 使用最相关节点创建的上下文字符串。
        """
        query_embedding = self.create_embedding(query)  # 为查询生成嵌入向量
        selected_nodes = []  # 初始化选中节点列表

        node_list = get_node_list(self.tree.all_nodes)  # 获取树中所有节点的列表
        embeddings = get_embeddings(node_list, self.context_embedding_model)  # 获取节点嵌入向量
        distances = distances_from_embeddings(query_embedding, embeddings)  # 计算距离
        indices = indices_of_nearest_neighbors_from_distances(distances)  # 获取最近邻节点的索引

        total_tokens = 0
        for idx in indices[:top_k]:  # 仅考虑 top_k 个节点
            node = node_list[idx]
            node_tokens = len(self.tokenizer.encode(node.text))
            if total_tokens + node_tokens > max_tokens:  # 检查是否超过最大令牌数限制
                break
            selected_nodes.append(node)
            total_tokens += node_tokens

        context = get_text(selected_nodes)  # 从选中的节点创建上下文
        return selected_nodes, context

    def retrieve_information(self, current_nodes: List[Node], query: str, num_layers: int) -> str:
        """
        从当前节点列表中基于查询检索最相关的信息。此方法在每个层次上选择最相关的节点，并在层与层之间进行迭代。

        参数:
        - current_nodes: 当前层的节点列表。
        - query: 查询文本。
        - num_layers: 要遍历的层数。

        返回:
        - 使用最相关节点创建的上下文字符串。
        """
        query_embedding = self.create_embedding(query)  # 为查询生成嵌入向量
        selected_nodes = []  # 初始化选中节点列表

        node_list = current_nodes
        for layer in range(num_layers):
            embeddings = get_embeddings(node_list, self.context_embedding_model)  # 获取当前层节点嵌入
            distances = distances_from_embeddings(query_embedding, embeddings)  # 计算距离
            indices = indices_of_nearest_neighbors_from_distances(distances)  # 获取最近邻节点的索引

            if self.selection_mode == "threshold":
                best_indices = [index for index in indices if distances[index] > self.threshold]
            elif self.selection_mode == "top_k":
                best_indices = indices[:self.top_k]

            nodes_to_add = [node_list[idx] for idx in best_indices]  # 选择最相关的节点
            selected_nodes.extend(nodes_to_add)

            if layer != num_layers - 1:  # 准备下一层的节点列表
                child_nodes = []
                for index in best_indices:
                    child_nodes.extend(node_list[index].children)
                child_nodes = list(dict.fromkeys(child_nodes))  # 去除重复值
                node_list = [self.tree.all_nodes[i] for i in child_nodes]

        context = get_text(selected_nodes)  # 从选中的节点创建上下文
        return selected_nodes, context

    def retrieve(
            self,
            query: str,
            start_layer: int = None,
            num_layers: int = None,
            top_k: int = 10,
            max_tokens: int = 3500,
            collapse_tree: bool = True,
            return_layer_information: bool = False,
    ) -> str:
        """
        执行查询并返回最相关的信息。

        参数:
        - query: 查询文本。
        - start_layer: 开始的层级，如果未指定，则使用配置中的start_layer。
        - num_layers: 遍历的层数，如果未指定，则使用配置中的num_layers。
        - top_k: 最相关节点的数量。
        - max_tokens: 结果中的最大令牌数。
        - collapse_tree: 是否折叠树以检索信息，默认为True。
        - return_layer_information: 是否返回层级信息，默认为False。

        返回:
        - 查询结果。如果return_layer_information为True，则还包括层级信息。
        """

        if not isinstance(query, str):
            raise ValueError("query must be a string")

        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be an integer and at least 1")

        if not isinstance(collapse_tree, bool):
            raise ValueError("collapse_tree must be a boolean")

        # Set defaults
        start_layer = self.start_layer if start_layer is None else start_layer
        num_layers = self.num_layers if num_layers is None else num_layers

        if not isinstance(start_layer, int) or not (
            0 <= start_layer <= self.tree.num_layers
        ):
            raise ValueError(
                "start_layer must be an integer between 0 and tree.num_layers"
            )

        if not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError("num_layers must be an integer and at least 1")

        if num_layers > (start_layer + 1):
            raise ValueError("num_layers must be less than or equal to start_layer + 1")

        if collapse_tree:
            logging.info(f"Using collapsed_tree")
            selected_nodes, context = self.retrieve_information_collapse_tree(
                query, top_k, max_tokens
            )
        else:
            layer_nodes = self.tree.layer_to_nodes[start_layer]
            selected_nodes, context = self.retrieve_information(
                layer_nodes, query, num_layers
            )

        if return_layer_information:

            layer_information = []

            for node in selected_nodes:
                layer_information.append(
                    {
                        "node_index": node.index,
                        "layer_number": self.tree_node_index_to_layer[node.index],
                    }
                )

            return context, layer_information

        return context
