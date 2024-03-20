import logging
from abc import ABC, abstractmethod

import torch
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_random_exponential
import os
from transformers import AutoModel, AutoTokenizer


logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class BaseEmbeddingModel(ABC):
    """
       抽象基类定义了嵌入模型必须实现的方法。
    """
    @abstractmethod
    def create_embedding(self, text):
        pass


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    """
        使用OpenAI API生成文本嵌入的具体实现。
    """
    def __init__(self, model="text-embedding-ada-002"):
        """
            初始化OpenAI客户端和模型。

            :param model: 使用的OpenAI嵌入模型。
        """
        self.client = OpenAI()
        self.model = model

    @retry(wait=wait_random_exponential(min=5, max=20), stop=stop_after_attempt(6))
    def create_embedding(self, text):
        """
        调用OpenAI API生成文本的嵌入向量。
        :param text: 输入的文本字符串。
        :return: 文本的向量表示。
        """
        text = text.replace("\n", " ")
        return (
            self.client.embeddings.create(input=[text], model=self.model)
            .data[0]
            .embedding
        )


class SBertEmbeddingModel(BaseEmbeddingModel):
    """
       使用Sentence-BERT模型生成文本嵌入的具体实现。
    """
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"):
        """
            初始化SBERT模型。
            :param model_name: 模型的名称或路径。
        """
        self.model = SentenceTransformer(model_name)


    def create_embedding(self, text):
        return self.model.encode(text)


class BGEEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="BAAI/bge-base-zh-v1.5"):
        """
        初始化BGE模型和分词器。
        :param model_name: 模型的名称或路径。
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def create_embedding(self, text):
        """
        使用BGE模型生成文本的嵌入向量。
        :param text: 输入的文本字符串。
        :return: 文本的向量表示。
        """
        # 将文本编码为模型输入格式
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        # 获取模型的最后一层隐藏状态
        with torch.no_grad():
            outputs = self.model(**inputs)
        # 取平均池化后的嵌入向量
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.detach().numpy()


if __name__ == "__main__":

    # 假定已有一个测试文本
    test_text = "这是一个用于测试嵌入模型的文本。"

    # 使用 SBertEmbeddingModel 生成嵌入向量
    sbert_model = SBertEmbeddingModel()
    embedding_sbert = sbert_model.create_embedding(test_text)
    print("测试 SBertEmbeddingModel，嵌入向量结果：", embedding_sbert)

    # 使用 OpenAIEmbeddingModel 生成嵌入向量
    openai_model = OpenAIEmbeddingModel()
    embedding_openai = openai_model.create_embedding(test_text)
    print("测试 OpenAIEmbeddingModel，嵌入向量结果：", embedding_openai)

    # 使用 BGEEmbeddingModel 生成嵌入向量
    bge_model = BGEEmbeddingModel()
    embedding_bge = bge_model.create_embedding(test_text)
    print("测试 BGEEmbeddingModel，嵌入向量结果：", embedding_bge)