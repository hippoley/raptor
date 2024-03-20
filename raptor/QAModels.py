import logging
import os

from openai import OpenAI


import getpass
from abc import ABC, abstractmethod

import torch
from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import T5ForConditionalGeneration, T5Tokenizer


class BaseQAModel(ABC):
    """
    所有问答模型的抽象基类。定义了问答模型必须实现的方法。
    """
    @abstractmethod
    def answer_question(self, context, question):
        """
        根据提供的上下文和问题生成答案。

        参数:
        - context (str): 提问的上下文信息。
        - question (str): 提出的问题。

        返回:
        - (str): 答案的文本字符串。
        """
        pass

class GPT3QAModel(BaseQAModel):
    """
    使用OpenAI的GPT-3模型进行问答的具体实现类。
    """
    def __init__(self, model="text-davinci-003"):
        """
        初始化GPT-3模型和OpenAI客户端。

        参数:
        - model (str): 指定使用的GPT-3模型版本，例如"text-davinci-003"。
        """
        self.model = model
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question, max_tokens=150, stop_sequence=None):
        """
        使用GPT-3模型生成对给定问题的答案。

        参数:
        - context (str): 提供的上下文信息。
        - question (str): 需要回答的问题。
        - max_tokens (int): 在生成答案时使用的最大令牌数，默认150。
        - stop_sequence (str): 生成答案时的停止序列，可选。

        返回:
        - (str): GPT-3模型根据上下文和问题生成的答案。
        """
        try:
            response = self.client.completions.create(
                prompt=f"根据以下信息{context}。请回答问题：{question}，如果可能，尽量在5-7个词以内。",
                temperature=0,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop_sequence,
                model=self.model,
            )
            return response.choices[0].text.strip()
        except Exception as e:
            logging.error(e)
            return ""


class GPT3TurboQAModel(BaseQAModel):
    """
    使用OpenAI的GPT-3.5 Turbo模型进行问答的实现类。
    """
    def __init__(self, model="gpt-3.5-turbo"):
        """
        初始化GPT-3 Turbo模型和OpenAI客户端。

        参数:
        - model (str): 指定使用的GPT-3 Turbo模型版本。
        """
        self.model = model
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question, max_tokens=150, stop_sequence=None):
        """
        使用GPT-3 Turbo模型根据上下文生成问题的答案。

        参数:
        - context (str): 提供的上下文信息。
        - question (str): 需要回答的问题。
        - max_tokens (int): 在生成答案时使用的最大令牌数，默认为150。
        - stop_sequence (str): 生成答案时的停止序列，可选。

        返回:
        - (str): GPT-3 Turbo模型根据上下文和问题生成的答案。
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Given the following information: {context}. Please answer the question: {question}, if possible, in less than 5-7 words."
                    },
                ],
                max_tokens=max_tokens,
                temperature=0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(e)
            return ""


class GPT4QAModel(BaseQAModel):
    def __init__(self, model="gpt-4"):
        """
        Initializes the GPT-3 model with the specified model version.

        Args:
            model (str, optional): The GPT-3 model version to use for generating summaries. Defaults to "text-davinci-003".
        """
        self.model = model
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _attempt_answer_question(
        self, context, question, max_tokens=150, stop_sequence=None
    ):
        """
        Generates a summary of the given context using the GPT-3 model.

        Args:
            context (str): The text to summarize.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.
            stop_sequence (str, optional): The sequence at which to stop summarization. Defaults to None.

        Returns:
            str: The generated summary.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are Question Answering Portal"},
                {
                    "role": "user",
                    "content": f"Given Context: {context} Give the best full answer amongst the option to question {question}",
                },
            ],
            temperature=0,
        )

        return response.choices[0].message.content.strip()

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question, max_tokens=150, stop_sequence=None):

        try:
            return self._attempt_answer_question(
                context, question, max_tokens=max_tokens, stop_sequence=stop_sequence
            )
        except Exception as e:
            print(e)
            return e


class UnifiedQAModel(BaseQAModel):
    """
    使用allenai的UnifiedQA模型进行问答的实现类。
    """
    def __init__(self, model_name="allenai/unifiedqa-v2-t5-3b-1363200"):
        """
        初始化UnifiedQA模型及其分词器。

        参数:
        - model_name (str): 指定使用的UnifiedQA模型名称，例如"allenai/unifiedqa-v2-t5-3b-1363200"。
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def run_model(self, input_string, **generator_args):
        """
        使用UnifiedQA模型生成答案。

        参数:
        - input_string (str): 结合了问题和上下文的输入字符串。

        返回:
        - (str): 模型生成的答案。
        """
        input_ids = self.tokenizer.encode(input_string, return_tensors="pt").to(self.device)
        res = self.model.generate(input_ids, **generator_args)
        return self.tokenizer.batch_decode(res, skip_special_tokens=True)

    def answer_question(self, context, question):
        """
        根据上下文和问题使用UnifiedQA模型生成答案。

        参数:
        - context (str): 提供的上下文信息。
        - question (str): 需要回答的问题。

        返回:
        - (str): 根据上下文和问题使用UnifiedQA模型生成的答案。
        """
        input_string = f"{question} \\n {context}"
        output = self.run_model(input_string)
        return output[0]
