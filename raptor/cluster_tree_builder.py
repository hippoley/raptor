import logging
import pickle
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Dict, List, Set

# 导入聚类算法、树构建器配置和树结构相关的模块
from .cluster_utils import ClusteringAlgorithm, RAPTOR_Clustering
from .tree_builder import TreeBuilder, TreeBuilderConfig
from .tree_structures import Node, Tree
from .utils import (distances_from_embeddings, get_children, get_embeddings,
                    get_node_list, get_text,
                    indices_of_nearest_neighbors_from_distances, split_text)

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class ClusterTreeConfig(TreeBuilderConfig):
    def __init__(
            self,
            reduction_dimension=10,
            clustering_algorithm=RAPTOR_Clustering,  # 默认使用 RAPTOR 聚类
            clustering_params={},  # 以字典形式传递额外参数
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.reduction_dimension = reduction_dimension
        self.clustering_algorithm = clustering_algorithm
        self.clustering_params = clustering_params

    """
      ClusterTreeConfig 定义了构建聚类树所需的配置参数。

      继承自 TreeBuilderConfig，添加了特定于聚类树构建的参数，如降维维度和聚类算法。

      属性:
      - reduction_dimension (int): 用于UMAP降维的目标维度。
      - clustering_algorithm (ClusteringAlgorithm): 聚类算法类，用于执行节点聚类。
      - clustering_params (dict): 传递给聚类算法的额外参数。
    """

    def log_config(self):
        # 记录并返回聚类树配置的摘要
        base_summary = super().log_config()
        cluster_tree_summary = f"""
        Reduction Dimension: {self.reduction_dimension}
        Clustering Algorithm: {self.clustering_algorithm.__name__}
        Clustering Parameters: {self.clustering_params}
        """
        return base_summary + cluster_tree_summary


class ClusterTreeBuilder(TreeBuilder):
    """
        聚类树构建器，专门负责根据聚类算法结果逐层构建树结构。

        该类扩展了TreeBuilder基类，引入了特定于聚类的配置和方法，
        使得可以根据数据的嵌入向量动态构建层次化的聚类树

        Attributes:
            reduction_dimension (int): 使用UMAP算法进行降维时的目标维度。
            clustering_algorithm (ClusteringAlgorithm): 实现了ClusteringAlgorithm接口的聚类算法类。
            clustering_params (dict): 聚类算法的额外参数。

        Methods:
            construct_tree: 接收当前层的节点，通过聚类算法和递归细分，构建出树的层级结构。
     """

    def __init__(self, config) -> None:
        super().__init__(config)

        if not isinstance(config, ClusterTreeConfig):
            raise ValueError("config must be an instance of ClusterTreeConfig")
        self.reduction_dimension = config.reduction_dimension
        self.clustering_algorithm = config.clustering_algorithm
        self.clustering_params = config.clustering_params

        logging.info(
            f"Successfully initialized ClusterTreeBuilder with Config {config.log_config()}"
        )

    def construct_tree(
            self,
            current_level_nodes: Dict[int, Node],
            all_tree_nodes: Dict[int, Node],
            layer_to_nodes: Dict[int, List[Node]],
            use_multithreading: bool = False,
    ) -> Dict[int, Node]:
        """
           递归构建聚类树的核心方法。该方法按层处理节点，每层的节点基于上一层的聚类结果生成。

           使用指定的聚类算法对当前层的节点进行聚类，基于聚类结果创建新的节点，这些新节点成为下一层的父节点。
           直至满足停止条件：聚类无法进一步细分或聚类内的文本总长度低于预设的最大长度限制。
           这个过程可以选择性地通过多线程来加速

           Parameters:
               current_level_nodes: 当前层的节点字典，键为节点 ID，值为节点对象。
               all_tree_nodes: 包含所有已经生成的树节点的字典。
               layer_to_nodes: 记录每一层及其对应节点的字典。
               use_multithreading: 是否使用多线程加速树的构建过程，默认为False。

           Returns:
               更新后包含所有树节点的字典。

           Note:
               此方法的递归构建逻辑体现了聚类树构建过程的层次化特性，
               每一层的构建都是基于上一层聚类结果的进一步细化。
        """
        logging.info("Using Cluster TreeBuilder")

        next_node_index = len(all_tree_nodes)  # 初始化下一个可用的节点索引

        def process_cluster(
                cluster, new_level_nodes, next_node_index, summarization_length, lock
        ):
            """
            处理单个聚类，生成摘要文本并创建新的父节点。

            Parameters:
                cluster: 当前正在处理的聚类，包含该聚类中所有节点的列表。
                new_level_nodes: 当前层新生成的节点字典。
                next_node_index: 用于分配给新节点的索引。
                summarization_length: 生成摘要的最大长度。
                lock: 线程锁，用于在多线程环境中保护共享资源。
            """

            # 1. 获取聚类中所有文本，并生成摘要
            node_texts = get_text(cluster)
            summarized_text = self.summarize(
                context=node_texts,
                max_tokens=summarization_length,
            )

            logging.info(
                f"Node Texts Length: {len(self.tokenizer.encode(node_texts))}, Summarized Text Length: {len(self.tokenizer.encode(summarized_text))}"
            )

            # 2. 创建新的父节点并将其添加到new_level_nodes字典中
            __, new_parent_node = self.create_node(
                next_node_index, summarized_text, {node.index for node in cluster}
            )

            with lock:  # 确保线程安全
                new_level_nodes[next_node_index] = new_parent_node

        # 开始构建每一层
        for layer in range(self.num_layers):
            # 存储当前层新生成的所有节点
            new_level_nodes = {}
            logging.info(f"Constructing Layer {layer}")

            # 检查当前层节点数量，确定是否继续构建
            node_list_current_layer = get_node_list(current_level_nodes)
            if len(node_list_current_layer) <= self.reduction_dimension + 1:
                # 更新层数，准备停止构建
                self.num_layers = layer
                logging.info(
                    f"Stopping Layer construction: Cannot Create More Layers. Total Layers in tree: {layer}"
                )
                # 节点数量过少，无法继续聚类，停止构建
                break

            # 执行聚类，生成新的父节点
            clusters = self.clustering_algorithm.perform_clustering(
                node_list_current_layer,
                self.cluster_embedding_model,
                reduction_dimension=self.reduction_dimension,
                **self.clustering_params,
            )

            lock = Lock()

            summarization_length = self.summarization_length
            logging.info(f"Summarization Length: {summarization_length}")

            # 根据聚类结果，使用多线程或单线程处理每个聚类
            if use_multithreading:
                with ThreadPoolExecutor() as executor:
                    for cluster in clusters:
                        executor.submit(
                            process_cluster,  # 处理单个聚类的函数
                            cluster,
                            new_level_nodes,
                            next_node_index,
                            summarization_length,
                            lock,
                        )
                        next_node_index += 1
                    executor.shutdown(wait=True)

            else:
                # 单线程处理每个聚类
                for cluster in clusters:
                    process_cluster(
                        cluster,
                        new_level_nodes,
                        next_node_index,
                        summarization_length,
                        lock,
                    )
                    next_node_index += 1

            # 更新树结构，准备下一层的构建，为新的聚类创建父节点
            layer_to_nodes[layer + 1] = list(new_level_nodes.values())
            current_level_nodes = new_level_nodes
            all_tree_nodes.update(new_level_nodes)

            tree = Tree(
                all_tree_nodes,
                layer_to_nodes[layer + 1],
                layer_to_nodes[0],
                layer + 1,
                layer_to_nodes,
            )

        # 构建完毕，返回包含所有节点的树结构
        return current_level_nodes
