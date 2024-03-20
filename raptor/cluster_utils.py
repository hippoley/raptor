import logging
import random
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import tiktoken
import umap
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits
threadpool_limits(limits=1)  # 限制线程池使用单个线程

# Initialize logging
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

from raptor.tree_structures import Node
# Import necessary methods from other modules
from raptor.utils import get_embeddings

# Set a random seed for reproducibility
RANDOM_SEED = 224
random.seed(RANDOM_SEED)


def global_cluster_embeddings(
        embeddings: np.ndarray,
        dim: int,
        n_neighbors: Optional[int] = None,
        metric: str = "cosine",
) -> np.ndarray:
    """
      对所有文本块嵌入向量执行全局降维，是后续聚类步骤的预处理，旨在减少数据复杂性并提高处理速度。

      输入参数:
          :param embeddings: 原始的嵌入向量数组。
          :param dim: 目标降维后的维度。
          :param n_neighbors: 用于UMAP算法的邻居数量，如果未指定，则根据数据规模自动确定。
          :param metric: 计算距离时使用的度量方法，默认为"cosine"。

      返回:
          reduced_embeddings: 降维后的嵌入向量。
    """

    # 1. 如果未指定n_neighbors，根据数据规模自动计算一个值
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)

    # 2. 使用UMAP算法进行降维
    # UMAP算法将嵌入向量从原始维度降低到指定的dim维度
    # UMAP通过考虑每个点的n_neighbors个邻居来保留局部邻域结构
    # 而使用metric参数指定的距离度量方法来评估点之间的距离
    reduced_embeddings = umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)
    return reduced_embeddings


def local_cluster_embeddings(
        embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    """
        对特定聚类内的文本块嵌入向量执行局部降维，有助于进一步揭示数据的内在结构，为更细致的聚类提供便利

        输入参数:
            :param embeddings: 聚类内的嵌入向量数组。
            :param dim: 目标降维后的维度。
            :param num_neighbors: 用于UMAP算法的邻居数量，默认值为10。
            :param metric: 计算距离时使用的度量方法，默认为"cosine"。

        返回:
            reduced_embeddings: 聚类内降维后的嵌入向量。
    """

    # 1. 使用UMAP算法进行局部降维
    # 使用UMAP算法但是通常采用不同的n_neighbors值，来更精细地处理局部数据结构
    reduced_embeddings = umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)

    return reduced_embeddings


def get_optimal_clusters(
        embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
) -> int:
    """
      寻找最适合数据的聚类数量的过程。

      输入参数:
          :param embeddings: 数据点的嵌入向量。
          :param max_clusters: 尝试的最大聚类数量。
          :param random_state: 随机状态，确保结果的可复现性。

      返回:
          optimal_clusters: 最优聚类数量。
    """

    # 1. 设置聚类数量范围:
    # 确定了一个可能的聚类数量的范围, 从1到max_clusters（不超过数据点的数量）
    # 为后续寻找最优聚类数量提供了一个搜索范围
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)

    # 2. 计算每种聚类数量的BIC值：
    # 对于给定范围内的每一个聚类数量，使用GaussianMixture模型进行拟合，并计算对应的BIC值
    # BIC值反映了模型复杂度和数据拟合度的权衡，较小的BIC值表明模型在简洁性和解释性之间达到了较好的平衡
    bics = []
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))

    # 3. 选择最优聚类数量
    # 所有计算出的BIC值中，选择最小的那个对应的聚类数量作为最优数量。
    # 这一步是基于模型选择的标准进行的，旨在找到最适合当前数据结构的聚类数量
    optimal_clusters = n_clusters[np.argmin(bics)]
    return optimal_clusters


def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    """
       使用GMM对数据进行聚类。

       输入参数:
           :param embeddings: 数据点的嵌入向量。
           :param threshold: 用于确定数据点聚类归属的概率阈值。
           :param random_state: 随机状态，确保结果的可复现性。

       返回:
           labels: 每个数据点的聚类标签。
           n_clusters: 聚类的数量。
    """

    # 1. 获取最优聚类数量
    # 使用get_optimal_clusters函数确定数据最适合的聚类数量。
    n_clusters = get_optimal_clusters(embeddings)

    # 2. 拟合GMM模型并预测聚类概率
    # 使用最优聚类数量初始化GaussianMixture模型，然后对数据进行拟合
    # 通过predict_proba方法，为每个数据点计算其属于各个聚类的概率
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)

    # 3. 应用概率阈值，生成聚类标签：
    # 对每个数据点的聚类概率, 根据给定的阈值，决定每个点属于哪个或哪些聚类
    # 如果数据点属于某个聚类的概率超过阈值，该数据点则被认为属于该聚类。
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters


def perform_clustering(
        embeddings: np.ndarray,
        dim: int,
        threshold: float,
        verbose: bool = False
) -> List[np.ndarray]:
    """
    执行聚类的主函数。

    输入参数:
        :param embeddings: 包含所有文本块嵌入向量的NumPy数组。
        :param dim: 降维后的目标维度，用于全局聚类。
        :param threshold: 聚类概率阈值，用于GMM聚类。
        :param verbose: 是否输出详细日志。

    返回:
        all_local_clusters: 每个文本块的局部聚类索引列表。
    """

    # 1. 全局聚类: 通过UMAP算法对嵌入向量进行降维。
    reduced_embeddings_global = global_cluster_embeddings(embeddings, dim)

    # 2. 使用GMM算法基于降维后的嵌入向量进行全局聚类，并获取聚类结果及聚类数量
    global_clusters, n_global_clusters = GMM_cluster(reduced_embeddings_global, threshold)

    if verbose:
        logging.info(f"Global Clusters: {n_global_clusters}")

    """
    构建树结构
    -   这部分对应的代码是在全局聚类后初始化局部聚类列表
    -   并为每个文本块分配一个局部聚类的索引。
    """
    # 3. 初始化局部聚类列表：用于存储每个文本块的局部聚类索引
    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    """
    递归聚类
    -   递归聚类的过程在遍历每个全局聚类的循环中体现。
    -   如果一个全局聚类包含的文本块数量超过了某个阈值，它将被进一步细分为局部聚类
    """
    # 4. 遍历每个全局聚类进行进一步处理
    for i in range(n_global_clusters):
        #  获取当前全局聚类中所有文本块的嵌入向量
        global_cluster_embeddings_ = embeddings[
            np.array([i in gc for gc in global_clusters])
        ]

        if verbose:
            logging.info(f"Nodes in Global Cluster {i}: {len(global_cluster_embeddings_)}")

        if len(global_cluster_embeddings_) == 0:
            continue

        # 检查和递归聚类逻辑
        if len(global_cluster_embeddings_) <= dim + 1:
            # 如果全局聚类中的文本块数量较少，则不进行进一步聚类
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            # 如果文本块数量较多，对这些嵌入向量进行局部降维，并再次聚类
            reduced_embeddings_local = local_cluster_embeddings(global_cluster_embeddings_, dim)
            local_clusters, n_local_clusters = GMM_cluster(reduced_embeddings_local, threshold)

        if verbose:
            logging.info(f"Local Clusters in Global Cluster {i}: {n_local_clusters}")

        """
        构建树状检索系统
        -   树状检索系统的构建是通过上述聚类过程的递归应用实现的。
        -   每个局部聚类实际上是树结构中的一个节点，
        -   而全局聚类和局部聚类的层次关系构建了树的层次结构。
        """
        # 更新局部聚类列表
        for j in range(n_local_clusters):
            # 找到属于当前局部聚类的文本块索引
            local_cluster_embeddings_ = global_cluster_embeddings_[
                np.array([j in lc for lc in local_clusters])
            ]
            indices = np.where(
                (embeddings == local_cluster_embeddings_[:, None]).all(-1)
            )[1]
            for idx in indices:
                # 更新每个文本块的局部聚类索引
                all_local_clusters[idx] = np.append(all_local_clusters[idx], j + total_clusters)

        total_clusters += n_local_clusters

    if verbose:
        logging.info(f"Total Clusters: {total_clusters}")

    # 通过更新 all_local_clusters 列表来记录每个文本块的聚类索引。
    # 这个列表最终会包含所有文本块的聚类信息，
    # 从而形成一个多层次的树状结构，其中每个节点代表一个聚类。
    return all_local_clusters


class ClusteringAlgorithm(ABC):
    @abstractmethod
    def perform_clustering(self, embeddings: np.ndarray, **kwargs) -> List[List[int]]:
        pass


class RAPTOR_Clustering(ClusteringAlgorithm):
    def perform_clustering(
            nodes: List[Node],
            embedding_model_name: str,
            max_length_in_cluster: int = 3500,
            tokenizer=tiktoken.get_encoding("cl100k_base"),
            reduction_dimension: int = 10,
            threshold: float = 0.1,
            verbose: bool = False,
    ) -> List[List[Node]]:

        """
        执行节点的聚类，并在聚类内文本总长度超过限制时递归地重新聚类。

        参数说明:
        - nodes: 节点列表，每个节点包含文本及其嵌入向量。
        - embedding_model_name: 使用的嵌入模型名称，用于从节点中提取嵌入向量。
        - max_length_in_cluster: 单个聚类允许的最大文本长度，超出则触发递归聚类。
        - tokenizer: 用于计算文本长度的编码器。
        - reduction_dimension: 降维到的目标维数，用于UMAP算法。
        - threshold: GMM聚类中使用的概率阈值。
        - verbose: 是否输出过程信息，用于调试。

        返回值:
        - 返回一个列表，包含按聚类分组的节点列表。
        """

        # 1.从节点中提取嵌入向量
        embeddings = np.array([node.embeddings[embedding_model_name] for node in nodes])

        # 2.使用UMAP和GMM对嵌入向量进行全局聚类
        clusters = perform_clustering(
            embeddings, dim=reduction_dimension, threshold=threshold
        )

        # 3. 初始化用于存储最终聚类结果的列表
        node_clusters = []

        # 4. 遍历每个聚类，根据聚类标签将节点分组
        for label in np.unique(np.concatenate(clusters)):
            # 根据聚类标签，找出属于当前聚类的节点
            indices = [i for i, cluster in enumerate(clusters) if label in cluster]

            # 将对应的节点添加到node_clusters列表中
            cluster_nodes = [nodes[i] for i in indices]

            # 如果聚类中只有一个节点，则直接将其作为一个聚类结果
            if len(cluster_nodes) == 1:
                node_clusters.append(cluster_nodes)
                continue

            # 计算当前聚类中所有文本的总长度
            total_length = sum(
                [len(tokenizer.encode(node.text)) for node in cluster_nodes]
            )

            # 如果文本总长度超过最大限制，对当前聚类递归聚类
            if total_length > max_length_in_cluster:
                # 输出递归聚类的信息（如果启用了详细模式）
                if verbose:
                    logging.info(
                        f"reclustering cluster with {len(cluster_nodes)} nodes"
                    )
                # 递归调用perform_clustering对当前聚类进行细分
                node_clusters.extend(
                    RAPTOR_Clustering.perform_clustering(
                        cluster_nodes, embedding_model_name, max_length_in_cluster
                    )
                )
            else:
                # 如果未超出长度限制，将满足条件的聚类直接添加到最终结果中
                node_clusters.append(cluster_nodes)

        # 5. 返回根据聚类分组的节点列表，形成多层次结构
        return node_clusters


# 生成模拟嵌入向量
embeddings = np.random.rand(100, 128)  # 假设有100个128维的嵌入向量


def main():
    print("开始全局降维...")
    global_reduced = global_cluster_embeddings(embeddings, dim=50)

    print("开始局部降维...")
    local_reduced = local_cluster_embeddings(global_reduced[:10], dim=5)  # 假设对前10个嵌入向量进行局部降维

    print("寻找最优聚类数量...")
    optimal_clusters = get_optimal_clusters(embeddings)

    print("使用GMM进行聚类...")
    labels, n_clusters = GMM_cluster(embeddings, threshold=0.5)

    print("执行聚类...")
    clusters = perform_clustering(embeddings, dim=50, threshold=0.5, verbose=False)

    print("使用RAPTOR聚类算法进行聚类...")
    # 创建节点列表，每个节点包含一个模拟嵌入向量
    nodes = [Node(text=str(i), index=i, children=set(), embeddings={"default": emb}) for i, emb in
             enumerate(embeddings)]
    raptor_clusters = RAPTOR_Clustering().perform_clustering(nodes=nodes, embedding_model_name="default",
                                                             max_length_in_cluster=5000)

    print(f"完成聚类，共有{len(raptor_clusters)}个聚类。")


if __name__ == "__main__":
    main()