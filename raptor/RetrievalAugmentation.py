import logging
import pickle

from .cluster_tree_builder import ClusterTreeBuilder, ClusterTreeConfig
from .EmbeddingModels import BaseEmbeddingModel
from .QAModels import BaseQAModel, GPT3TurboQAModel
from .SummarizationModels import BaseSummarizationModel
from .tree_builder import TreeBuilder, TreeBuilderConfig
from .tree_retriever import TreeRetriever, TreeRetrieverConfig
from .tree_structures import Node, Tree

# Define a dictionary to map supported tree builders to their respective configs
supported_tree_builders = {"cluster": (ClusterTreeBuilder, ClusterTreeConfig)}

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class RetrievalAugmentationConfig:
    """
    配置检索增强系统，包括树构建器、树检索器、问答模型、嵌入模型和摘要模型的配置。
    """
    def __init__(
        self,
        tree_builder_config=None,  # 用于树构建的配置，如果未提供，则将使用默认值。
        tree_retriever_config=None,  # 用于树检索的配置，如果未提供，则将使用默认值。
        qa_model=None,  # 用于问答任务的模型，如果未提供，则将使用GPT3TurboQAModel。
        embedding_model=None,  # 用于生成文档嵌入的模型，如果未提供，则需指定。
        summarization_model=None,  # 用于生成摘要的模型，如果未提供，则需指定。
        tree_builder_type="cluster",  # 树构建类型，目前支持"cluster"。
        # 以下参数为TreeRetrieverConfig和TreeBuilderConfig的新参数
        tr_tokenizer=None,  # 树检索器使用的分词器。
        tr_threshold=0.5,  # 树检索器使用的阈值。
        tr_top_k=5,  # 树检索器在每一层选择的最高k个节点。
        tr_selection_mode="top_k",  # 树检索器的选择模式。
        tr_context_embedding_model="OpenAI",  # 树检索器使用的上下文嵌入模型。
        tr_embedding_model=None,  # 树检索器使用的嵌入模型。
        tr_num_layers=None,  # 树检索器使用的层数。
        tr_start_layer=None,  # 树检索器的起始层。
        tb_tokenizer=None,  # 树构建器使用的分词器。
        tb_max_tokens=100,  # 树构建器处理的最大令牌数。
        tb_num_layers=5,  # 树构建器使用的层数。
        tb_threshold=0.5,  # 树构建器使用的阈值。
        tb_top_k=5,  # 树构建器在每一层选择的最高k个节点。
        tb_selection_mode="top_k",  # 树构建器的选择模式。
        tb_summarization_length=100,  # 生成摘要的长度。
        tb_summarization_model=None,  # 用于生成摘要的模型。
        tb_embedding_models=None,  # 树构建器使用的嵌入模型集合。
        tb_cluster_embedding_model="OpenAI",  # 树构建器用于聚类的嵌入模型。
    ):
        # 验证树构建类型是否支持
        if tree_builder_type not in supported_tree_builders:
            raise ValueError(f"tree_builder_type must be one of {list(supported_tree_builders.keys())}")

        # 验证问答模型是否为BaseQAModel的实例
        if qa_model is not None and not isinstance(qa_model, BaseQAModel):
            raise ValueError("qa_model must be an instance of BaseQAModel")

        # 如果提供了嵌入模型，并且该模型不是BaseEmbeddingModel的实例，则抛出错误。
        # 这确保了嵌入模型具有必要的接口和功能，以便于系统正确使用。
        if embedding_model is not None and not isinstance(
                embedding_model, BaseEmbeddingModel
        ):
            raise ValueError(
                "embedding_model must be an instance of BaseEmbeddingModel"
            )

        # 如果同时提供了嵌入模型和tb_embedding_models，则抛出错误。
        # 这是为了避免配置冲突，确保系统的一致性和预期行为。
        elif embedding_model is not None:
            if tb_embedding_models is not None:
                raise ValueError(
                    "Only one of 'tb_embedding_models' or 'embedding_model' should be provided, not both."
                )
            # 如果只提供了embedding_model，则将其注册到系统中，并且用于聚类和上下文嵌入的默认模型名称设置为"EMB"。
            tb_embedding_models = {"EMB": embedding_model}
            tr_embedding_model = embedding_model
            tb_cluster_embedding_model = "EMB"
            tr_context_embedding_model = "EMB"

        # 如果提供了摘要模型，并且该模型不是BaseSummarizationModel的实例，则抛出错误。
        # 这确保了摘要模型具有生成摘要所需的接口和功能。
        if summarization_model is not None and not isinstance(
                summarization_model, BaseSummarizationModel
        ):
            raise ValueError(
                "summarization_model must be an instance of BaseSummarizationModel"
            )

        # 如果同时提供了摘要模型和tb_summarization_model，则抛出错误。
        # 这是为了避免配置上的冲突，确保系统使用一致且预期的摘要生成策略。
        elif summarization_model is not None:
            if tb_summarization_model is not None:
                raise ValueError(
                    "Only one of 'tb_summarization_model' or 'summarization_model' should be provided, not both."
                )
            tb_summarization_model = summarization_model

        # 设置树构建器配置。根据提供的树构建类型，使用对应的构建器类和配置类。
        tree_builder_class, tree_builder_config_class = supported_tree_builders[
            tree_builder_type
        ]
        # 如果未提供树构建配置，则使用默认值创建一个新的配置实例。
        if tree_builder_config is None:
            tree_builder_config = tree_builder_config_class(
                tokenizer=tb_tokenizer,
                max_tokens=tb_max_tokens,
                num_layers=tb_num_layers,
                threshold=tb_threshold,
                top_k=tb_top_k,
                selection_mode=tb_selection_mode,
                summarization_length=tb_summarization_length,
                summarization_model=tb_summarization_model,
                embedding_models=tb_embedding_models,
                cluster_embedding_model=tb_cluster_embedding_model,
            )

        # 验证提供的树构建配置是否为正确的类实例。
        elif not isinstance(tree_builder_config, tree_builder_config_class):
            raise ValueError(
                f"tree_builder_config must be a direct instance of {tree_builder_config_class} for tree_builder_type '{tree_builder_type}'"
            )

        # 设置树检索配置。如果未提供，则创建一个新的配置实例。
        if tree_retriever_config is None:
            tree_retriever_config = TreeRetrieverConfig(
                tokenizer=tr_tokenizer,
                threshold=tr_threshold,
                top_k=tr_top_k,
                selection_mode=tr_selection_mode,
                context_embedding_model=tr_context_embedding_model,
                embedding_model=tr_embedding_model,
                num_layers=tr_num_layers,
                start_layer=tr_start_layer,
            )

        # 验证提供的树检索配置是否为TreeRetrieverConfig的实例。
        elif not isinstance(tree_retriever_config, TreeRetrieverConfig):
            raise ValueError(
                "tree_retriever_config must be an instance of TreeRetrieverConfig"
            )

        # 将创建的配置分配给实例，完成系统配置的初始化。
        self.tree_builder_config = tree_builder_config
        self.tree_retriever_config = tree_retriever_config
        self.qa_model = qa_model or GPT3TurboQAModel()
        self.tree_builder_type = tree_builder_type

    def log_config(self):
        config_summary = """
        检索增强配置:
            {tree_builder_config}

            {tree_retriever_config}

            问答模型: {qa_model}
            树构建器类型: {tree_builder_type}
        """.format(
            tree_builder_config=self.tree_builder_config.log_config(),  # 树构建器配置的日志信息
            tree_retriever_config=self.tree_retriever_config.log_config(),  # 树检索器配置的日志信息
            qa_model=self.qa_model,  # 使用的问答模型
            tree_builder_type=self.tree_builder_type,  # 选择的树构建器类型
        )
        return config_summary


class RetrievalAugmentation:
    """
    一个结合了TreeBuilder和TreeRetriever类的检索增强类。这个类允许向树中添加文档，检索信息，并回答问题。
    """

    def __init__(self, config=None, tree=None):
        """
        使用指定的配置初始化RetrievalAugmentation实例。
        参数:
            config (RetrievalAugmentationConfig): RetrievalAugmentation实例的配置。
            tree: 树实例或者指向已序列化树文件的路径。
        """
        # 如果未提供配置，则使用默认配置
        if config is None:
            config = RetrievalAugmentationConfig()
        # 确保提供的配置是RetrievalAugmentationConfig的实例
        if not isinstance(config, RetrievalAugmentationConfig):
            raise ValueError("config必须是RetrievalAugmentationConfig的一个实例")

        # 检查树是否为路径（指向已序列化树的路径）
        if isinstance(tree, str):
            try:
                with open(tree, "rb") as file:
                    self.tree = pickle.load(file)
                if not isinstance(self.tree, Tree):
                    raise ValueError("加载的对象不是Tree的一个实例")
            except Exception as e:
                raise ValueError(f"从{tree}加载树失败: {e}")
        elif isinstance(tree, Tree) or tree is None:
            self.tree = tree
        else:
            raise ValueError("tree必须是Tree的一个实例，一个指向已序列化Tree的路径，或者None")

        # 根据配置创建树构建器和树检索器
        tree_builder_class = supported_tree_builders[config.tree_builder_type][0]
        self.tree_builder = tree_builder_class(config.tree_builder_config)

        self.tree_retriever_config = config.tree_retriever_config
        self.qa_model = config.qa_model

        # 如果树已经存在，则创建一个树检索器实例
        if self.tree is not None:
            self.retriever = TreeRetriever(self.tree_retriever_config, self.tree)
        else:
            self.retriever = None

        logging.info("RetrievalAugmentation实例成功初始化，配置如下: {}".format(config.log_config()))

    def add_documents(self, docs):
        """
        向树中添加文档并创建一个TreeRetriever实例。
        参数:
            docs (str): 要添加到树中的输入文本。
        """
        if self.tree is not None:
            user_input = input("警告: 即将覆盖现有的树。你是想调用'add_to_existing'吗？ (y/n): ")
            if user_input.lower() == "y":
                # 如果用户选择是，应调用add_to_existing(docs)
                return

        # 从提供的文档构建树，并为其创建一个树检索器实例
        self.tree = self.tree_builder.build_from_text(text=docs)
        self.retriever = TreeRetriever(self.tree_retriever_config, self.tree)

    def retrieve(self, question, start_layer=None, num_layers=None, top_k=10, max_tokens=3500, collapse_tree=True, return_layer_information=True):
        """
        使用TreeRetriever实例检索信息并回答问题。
        参数:
            question (str): 要回答的问题。
            start_layer (int): 开始检索的层级。默认为self.start_layer。
            num_layers (int): 要遍历的层数。默认为self.num_layers。
            top_k (int): 返回的最多结果数量。默认为10。
            max_tokens (int): 最大令牌数。默认为3500。
            collapse_tree (bool): 是否在检索时折叠树。默认为True。
            return_layer_information (bool): 是否返回层级信息。默认为True。
        返回:
            str: 可以找到答案的上下文。
        异常:
            ValueError: 如果TreeRetriever实例未被初始化。
        """
        if self.retriever is None:
            raise ValueError("TreeRetriever实例未被初始化。请首先调用'add_documents'。")

        # 调用retriever的retrieve方法，根据提供的参数检索信息
        return self.retriever.retrieve(question, start_layer, num_layers, top_k, max_tokens, collapse_tree, return_layer_information)

    def answer_question(self, question, top_k=10, start_layer=None, num_layers=None, max_tokens=3500, collapse_tree=True, return_layer_information=False):
        """
        使用TreeRetriever实例检索信息并回答问题。
        参数:
            question (str): 要回答的问题。
            start_layer (int): 开始检索的层级。默认为self.start_layer。
            num_layers (int): 要遍历的层数。默认为self.num_layers。
            max_tokens (int): 最大令牌数。默认为3500。
            collapse_tree (bool): 是否在检索时折叠树。默认为True。
            return_layer_information (bool): 是否返回层级信息。默认为False。
        返回:
            str: 对问题的答案。
        异常:
            ValueError: 如果TreeRetriever实例未被初始化。
        """
        # 首先使用retriever检索相关信息
        context, layer_information = self.retrieve(question, start_layer, num_layers, top_k, max_tokens, collapse_tree, True)

        # 然后使用qa_model根据检索到的信息回答问题
        answer = self.qa_model.answer_question(context, question)

        # 根据参数决定是否返回层级信息
        if return_layer_information:
            return answer, layer_information

        return answer

    def save(self, path):
        """
        将树结构保存到文件。
        参数:
            path (str): 保存文件的路径。
        异常:
            ValueError: 如果没有树可保存。
        """
        if self.tree is None:
            raise ValueError("没有树可保存。")
        # 将树序列化并保存到指定路径
        with open(path, "wb") as file:
            pickle.dump(self.tree, file)
        logging.info(f"树成功保存到{path}")
