import copy
import logging
import os
from abc import abstractclassmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Dict, List, Optional, Set, Tuple

import openai
import tiktoken
from tenacity import retry, stop_after_attempt, wait_random_exponential

from raptor.EmbeddingModels import BaseEmbeddingModel, OpenAIEmbeddingModel
from raptor.SummarizationModels import (BaseSummarizationModel,
                                        GPT3TurboSummarizationModel)
from raptor.tree_structures import Node, Tree
from raptor.utils import (distances_from_embeddings, get_children, get_embeddings,
                          get_node_list, get_text,
                          indices_of_nearest_neighbors_from_distances, split_text)
import os

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class TreeBuilderConfig:
    def __init__(
            self,
            tokenizer=None,  # 分词器
            max_tokens=None,  # 最大令牌数
            num_layers=None,  # 层数
            threshold=None,  # 阈值
            top_k=None,  # top_k值
            selection_mode=None,  # 选择模式
            summarization_length=None,  # 摘要长度
            summarization_model=None,  # 摘要模型
            embedding_models=None,  # 嵌入模型
            cluster_embedding_model=None,  # 聚类嵌入模型
    ):
        # 如果没有提供分词器，默认使用'cl100k_base'
        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("cl100k_base")
        self.tokenizer = tokenizer

        # 如果没有提供最大令牌数，默认为100
        if max_tokens is None:
            max_tokens = 100
        # 确保最大令牌数是一个正整数
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens 必须是至少为1的整数")
        self.max_tokens = max_tokens

        # 如果没有提供层数，默认为5
        if num_layers is None:
            num_layers = 5
        # 确保层数是一个正整数
        if not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError("num_layers 必须是至少为1的整数")
        self.num_layers = num_layers

        # 如果没有提供阈值，默认为0.5
        if threshold is None:
            threshold = 0.5
        # 确保阈值是0到1之间的数字
        if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
            raise ValueError("threshold 必须是0和1之间的数字")
        self.threshold = threshold

        # 如果没有提供top_k值，默认为5
        if top_k is None:
            top_k = 5
        # 确保top_k值是一个正整数
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k 必须是至少为1的整数")
        self.top_k = top_k

        # 如果没有提供选择模式，默认为"top_k"
        if selection_mode is None:
            selection_mode = "top_k"
        # 确保选择模式是'top_k'或'threshold'
        if selection_mode not in ["top_k", "threshold"]:
            raise ValueError("selection_mode 必须是 'top_k' 或 'threshold'")
        self.selection_mode = selection_mode

        # 如果没有提供摘要长度，默认为100
        if summarization_length is None:
            summarization_length = 100
        self.summarization_length = summarization_length

        # 如果没有提供摘要模型，默认使用GPT3TurboSummarizationModel
        if summarization_model is None:
            summarization_model = GPT3TurboSummarizationModel()
        # 确保摘要模型是BaseSummarizationModel的实例
        if not isinstance(summarization_model, BaseSummarizationModel):
            raise ValueError("summarization_model 必须是 BaseSummarizationModel 的实例")
        self.summarization_model = summarization_model

        # 如果没有提供嵌入模型，默认使用OpenAI嵌入模型
        if embedding_models is None:
            embedding_models = {"OpenAI": OpenAIEmbeddingModel()}
        # 确保嵌入模型是字典格式，键为模型名称，值为实例
        if not isinstance(embedding_models, dict):
            raise ValueError("embedding_models 必须是模型名称和实例的字典")
        # 确保所有嵌入模型都是BaseEmbeddingModel的实例
        for model in embedding_models.values():
            if not isinstance(model, BaseEmbeddingModel):
                raise ValueError("所有嵌入模型必须是 BaseEmbeddingModel 的实例")
        self.embedding_models = embedding_models

        # 如果没有提供聚类嵌入模型，默认为"OpenAI"
        if cluster_embedding_model is None:
            cluster_embedding_model = "OpenAI"
        # 确保聚类嵌入模型是嵌入模型字典的键
        if cluster_embedding_model not in self.embedding_models:
            raise ValueError("cluster_embedding_model 必须是 embedding_models 字典的键")
        self.cluster_embedding_model = cluster_embedding_model

    def log_config(self):
        config_log = """
        TreeBuilderConfig 日志:
            分词器: {tokenizer}
            最大令牌数: {max_tokens}
            层数: {num_layers}
            阈值: {threshold}
            Top K值: {top_k}
            选择模式: {selection_mode}
            摘要长度: {summarization_length}
            摘要模型: {summarization_model}
            嵌入模型: {embedding_models}
            聚类嵌入模型: {cluster_embedding_model}
        """.format(
            tokenizer=self.tokenizer,
            max_tokens=self.max_tokens,
            num_layers=self.num_layers,
            threshold=self.threshold,
            top_k=self.top_k,
            selection_mode=self.selection_mode,
            summarization_length=self.summarization_length,
            summarization_model=self.summarization_model,
            embedding_models=self.embedding_models,
            cluster_embedding_model=self.cluster_embedding_model,
        )
        return config_log


class TreeBuilder:
    """
    TreeBuilder 类负责使用摘要模型和嵌入模型构建层次化的文本抽象结构，即“树”。
    """

    def __init__(self, config) -> None:
        """使用指定的配置初始化分词器、最大令牌数、层数、top-k值、阈值和选择模式。"""

        self.tokenizer = config.tokenizer
        self.max_tokens = config.max_tokens
        self.num_layers = config.num_layers
        self.top_k = config.top_k
        self.threshold = config.threshold
        self.selection_mode = config.selection_mode
        self.summarization_length = config.summarization_length
        self.summarization_model = config.summarization_model
        self.embedding_models = config.embedding_models
        self.cluster_embedding_model = config.cluster_embedding_model

        logging.info(
            f"成功使用配置初始化 TreeBuilder: {config.log_config()}"
        )

    def create_node(
            self, index: int, text: str, children_indices: Optional[Set[int]] = None
    ) -> Tuple[int, Node]:
        """使用给定的索引、文本和（可选的）子节点索引创建一个新节点。

        参数:
            index (int): 新节点的索引。
            text (str): 与新节点关联的文本。
            children_indices (Optional[Set[int]]): 表示新节点子节点的索引集。
                如果未提供，则使用空集。

        返回:
            Tuple[int, Node]: 包含索引和新创建节点的元组。
        """
        if children_indices is None:
            children_indices = set()

        embeddings = {
            model_name: model.create_embedding(text)
            for model_name, model in self.embedding_models.items()
        }
        return (index, Node(text, index, children_indices, embeddings))

    def create_embedding(self, text) -> List[float]:
        """
        使用指定的嵌入模型为给定文本生成嵌入。

        参数:
            text (str): 需要生成嵌入的文本。

        返回:
            List[float]: 生成的嵌入列表。
        """
        return self.embedding_models[self.cluster_embedding_model].create_embedding(
            text
        )

    def summarize(self, context, max_tokens=150) -> str:
        """
        使用指定的摘要模型为输入上下文生成摘要。

        参数:
            context (str, 可选): 要摘要的上下文。
            max_tokens (int, 可选): 生成摘要的最大令牌数。默认值为150。

        返回:
            str: 生成的摘要。
        """
        return self.summarization_model.summarize(context, max_tokens)

    def get_relevant_nodes(self, current_node, list_nodes) -> List[Node]:
        """
        基于嵌入空间的余弦距离，从节点列表中检索与当前节点最相关的top-k个节点。

        参数:
            current_node (Node): 当前节点。
            list_nodes (List[Node]): 节点列表。

        返回:
            List[Node]: 最相关的top-k个节点列表。
        """
        embeddings = get_embeddings(list_nodes, self.cluster_embedding_model)
        distances = distances_from_embeddings(
            current_node.embeddings[self.cluster_embedding_model], embeddings
        )
        indices = indices_of_nearest_neighbors_from_distances(distances)

        if self.selection_mode == "threshold":
            best_indices = [
                index for index in indices if distances[index] > self.threshold
            ]

        elif self.selection_mode == "top_k":
            best_indices = indices[: self.top_k]

        nodes_to_add = [list_nodes[idx] for idx in best_indices]

        return nodes_to_add

    def multithreaded_create_leaf_nodes(self, chunks: List[str]) -> Dict[int, Node]:
        """使用多线程从给定的文本块列表中创建叶节点。

        参数:
            chunks (List[str]): 转换为叶节点的文本块列表。

        返回:
            Dict[int, Node]: 映射节点索引到对应叶节点的字典。
        """
        with ThreadPoolExecutor() as executor:
            future_nodes = {
                executor.submit(self.create_node, index, text): (index, text)
                for index, text in enumerate(chunks)
            }

            leaf_nodes = {}
            for future in as_completed(future_nodes):
                index, node = future.result()
                leaf_nodes[index] = node

        return leaf_nodes

    def build_from_text(self, text: str, use_multithreading: bool = True) -> Tree:
        """从输入文本构建黄金树结构，可选择使用多线程。

        参数:
            text (str): 输入文本。
            use_multithreading (bool, 可选): 创建叶节点时是否使用多线程。
                默认值: True。

        返回:
            Tree: 黄金树结构。
        """
        chunks = split_text(text, self.tokenizer, self.max_tokens)

        logging.info("创建叶节点")

        if use_multithreading:
            leaf_nodes = self.multithreaded_create_leaf_nodes(chunks)
        else:
            leaf_nodes = {}
            for index, text in enumerate(chunks):
                __, node = self.create_node(index, text)
                leaf_nodes[index] = node

        layer_to_nodes = {0: list(leaf_nodes.values())}

        logging.info(f"创建了 {len(leaf_nodes)} 个叶节点嵌入")

        logging.info("构建所有节点")

        all_nodes = copy.deepcopy(leaf_nodes)

        root_nodes = self.construct_tree(all_nodes, all_nodes, layer_to_nodes)

        tree = Tree(all_nodes, root_nodes, leaf_nodes, self.num_layers, layer_to_nodes)

        return tree

    @abstractclassmethod
    def construct_tree(
            self,
            current_level_nodes: Dict[int, Node],
            all_tree_nodes: Dict[int, Node],
            layer_to_nodes: Dict[int, List[Node]],
            use_multithreading: bool = True,
    ) -> Dict[int, Node]:
        """
        通过迭代地总结相关节点的组并在每一步更新 current_level_nodes 和 all_tree_nodes 字典，
        逐层构建分层的树结构。

        参数:
            current_level_nodes (Dict[int, Node]): 当前的节点集合。
            all_tree_nodes (Dict[int, Node]): 所有节点的字典。
            use_multithreading (bool): 是否使用多线程加快过程。

        返回:
            Dict[int, Node]: 最终的根节点集合。
        """
        pass

        # logging.info("使用类似 Transformer 的 TreeBuilder")

        # def process_node(idx, current_level_nodes, new_level_nodes, all_tree_nodes, next_node_index, lock):
        #     relevant_nodes_chunk = self.get_relevant_nodes(
        #         current_level_nodes[idx], current_level_nodes
        #     )

        #     node_texts = get_text(relevant_nodes_chunk)

        #     summarized_text = self.summarize(
        #         context=node_texts,
        #         max_tokens=self.summarization_length,
        #     )

        #     logging.info(
        #         f"节点文本长度: {len(self.tokenizer.encode(node_texts))}, 摘要文本长度: {len(self.tokenizer.encode(summarized_text))}"
        #     )

        #     next_node_index, new_parent_node = self.create_node(
        #         next_node_index,
        #         summarized_text,
        #         {node.index for node in relevant_nodes_chunk}
        #     )

        #     with lock:
        #         new_level_nodes[next_node_index] = new_parent_node

        # for layer in range(self.num_layers):
        #     logging.info(f"构建第 {layer} 层: ")

        #     node_list_current_layer = get_node_list(current_level_nodes)
        #     next_node_index = len(all_tree_nodes)

        #     new_level_nodes = {}
        #     lock = Lock()

        #     if use_multithreading:
        #         with ThreadPoolExecutor() as executor:
        #             for idx in range(0, len(node_list_current_layer)):
        #                 executor.submit(process_node, idx, node_list_current_layer, new_level_nodes, all_tree_nodes, next_node_index, lock)
        #                 next_node_index += 1
        #             executor.shutdown(wait=True)
        #     else:
        #         for idx in range(0, len(node_list_current_layer)):
        #             process_node(idx, node_list_current_layer, new_level_nodes, all_tree_nodes, next_node_index, lock)

        #     layer_to_nodes[layer + 1] = list(new_level_nodes.values())
        #     current_level_nodes = new_level_nodes
        #     all_tree_nodes.update(new_level_nodes)

        # return new_level_nodes


if __name__ == "__main__":

    # 假设有一个测试文本
    test_text = "这是一个用于测试嵌入模型的文本。"

    # 初始化配置
    config = TreeBuilderConfig(
        tokenizer=tiktoken.get_encoding("cl100k_base"),
        max_tokens=100,
        num_layers=5,
        threshold=0.5,
        top_k=5,
        selection_mode="top_k",
        summarization_length=100,
        summarization_model=GPT3TurboSummarizationModel(),
        embedding_models={
            "OpenAI": OpenAIEmbeddingModel(),
        },
        cluster_embedding_model="OpenAI",
    )

    # 创建 TreeBuilder 实例
    tree_builder = TreeBuilder(config)

    # 创建一个节点
    index, node = tree_builder.create_node(0, test_text)
    print(f"创建的节点: {node.text}, 嵌入: {node.embeddings}")

    # 生成嵌入
    embeddings = tree_builder.create_embedding(test_text)
    print(f"生成的嵌入: {embeddings}")

    # 生成摘要
    summarized_text = tree_builder.summarize(test_text)
    print(f"生成的摘要: {summarized_text}")

    # 假设我们有一个上下文和一组节点，我们想要找到最相关的节点
    # 这里我们简单地使用单个节点和它自己作为上下文进行测试
    context = test_text
    list_nodes = [node]
    relevant_nodes = tree_builder.get_relevant_nodes(node, list_nodes)
    print(f"找到的相关节点: {[n.text for n in relevant_nodes]}")

    # 从文本构建树状结构
    tree = tree_builder.build_from_text(test_text)
    print(f"构建的树的根节点文本: {tree.root_node.text if tree.root_node else '无'}")
