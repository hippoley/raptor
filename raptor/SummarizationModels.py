import logging
import os
from abc import ABC, abstractmethod

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class BaseSummarizationModel(ABC):
    """
    摘要生成模型的抽象基类。定义了所有摘要模型必须实现的接口。
    """

    @abstractmethod
    def summarize(self, context, max_tokens=150):
        """
        根据给定的上下文生成摘要。

        参数:
        - context (str): 需要摘要的上下文。
        - max_tokens (int): 生成摘要的最大令牌数，默认为150。

        返回:
        - 摘要文本。
        """
        pass


class GPT3TurboSummarizationModel(BaseSummarizationModel):
    """
    使用OpenAI的GPT-3.5 Turbo模型进行摘要生成的实现。
    """

    def __init__(self, model="gpt-3.5-turbo"):
        """
        初始化GPT-3.5 Turbo模型。

        参数:
        - model (str): 使用的模型标识，默认为"gpt-3.5-turbo"。
        """
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):
        """
        重试策略装饰器，确保在请求失败时自动重试。

        参数:
        - context (str): 需要摘要的上下文。
        - max_tokens (int): 生成摘要的最大令牌数，默认为500。
        - stop_sequence (str): 指示摘要结束的序列，可选。

        返回:
        - 摘要文本。
        """
        try:
            client = OpenAI()

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Write a summary of the following, including as many key details as possible: {context}:",
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return e


class GPT3SummarizationModel(BaseSummarizationModel):
    """
    使用OpenAI的GPT-3 Davinci模型进行摘要生成的实现。
    """

    def __init__(self, model="text-davinci-003"):
        """
        初始化GPT-3 Davinci模型。

        参数:
        - model (str): 使用的模型标识，默认为"text-davinci-003"。
        """
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):
        """
        生成摘要的具体实现，使用OpenAI GPT-3 Davinci模型。

        参数:
        - context (str): 需要摘要的上下文。
        - max_tokens (int): 生成摘要的最大令牌数，默认为500。
        - stop_sequence (str): 指示摘要结束的序列，可选。

        返回:
        - 摘要文本。
        """
        try:
            client = OpenAI()

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Write a summary of the following, including as many key details as possible: {context}:",
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return e
