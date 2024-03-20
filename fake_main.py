import logging
from raptor.RetrievalAugmentation import RetrievalAugmentation, RetrievalAugmentationConfig
from raptor.EmbeddingModels import OpenAIEmbeddingModel
from raptor.EmbeddingModels import SBertEmbeddingModel
from raptor.QAModels import GPT3TurboQAModel
from raptor.SummarizationModels import GPT3TurboSummarizationModel
from raptor.tree_structures import Tree
import os


def load_document(file_path):
    """加载并返回文件内容"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def configure():
    """配置并返回RAPTOR系统所需的各项配置"""
    # 初始化SBert嵌入模型
    # embedding_model = SBertEmbeddingModel()
    embedding_model = OpenAIEmbeddingModel()
    qa_model = GPT3TurboQAModel()
    summarization_model = GPT3TurboSummarizationModel()

    # 初始化检索增强配置，具体参数根据需求设置
    config = RetrievalAugmentationConfig(
        qa_model=qa_model,
        embedding_model=embedding_model,
        summarization_model=summarization_model,
        tree_builder_type='cluster',
        # 添加其他需要的配置参数
    )

    return config

def main():
    logging.basicConfig(level=logging.INFO)

    # 配置RAPTOR
    config = configure()
    print("配置完成。")

    # 实例化检索增强对象
    retrieval_augmentation = RetrievalAugmentation(config)
    print("检索增强对象已实例化。")

    # 加载文档
    document_text = load_document('demo/sample1.txt')
    print("文档加载完成。")

    # 向RAPTOR树添加文档
    retrieval_augmentation.add_documents(document_text)
    print("文档已添加到RAPTOR树。")

    # 演示信息检索
    question = "灰姑娘的背景故事是怎么样的"
    answer, layer_info = retrieval_augmentation.answer_question(question, return_layer_information=True)

    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print(f"Layer Information: {layer_info}")

if __name__ == "__main__":
    main()
