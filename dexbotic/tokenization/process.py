from typing import Dict, List, Sequence

import transformers
import numpy as np

from dexbotic.constants import DEFAULT_IMAGE_TOKEN
from dexbotic.data.dataset.tokenization import Tokenization
from dexbotic.tokenization import tokenization as tokenization_lib


def _process(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False,
        chat_template: str = "dexbotic",
):
    if chat_template == "dexbotic":
        return tokenization_lib.tokenize_dexbotic(
            sources=sources,
            tokenizer=tokenizer,
            has_image=has_image,
            chat_template=chat_template,
        )
    else:
        raise ValueError(f"Unsupported chat template: {chat_template}")


def llava_multi_image_map_fn(conversations):
    messages = conversations

    for msg in messages:
        if DEFAULT_IMAGE_TOKEN in msg['value']:
            # move the image token to the beginning of the sentence
            msg['value'] = msg['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
            msg['value'] = DEFAULT_IMAGE_TOKEN + '\n' + msg['value']
            msg['value'] = msg['value'].strip()

    return conversations


def process_data_item(
    conversations: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    chat_template: str,
    has_image: bool
) -> Dict:
    conversations = llava_multi_image_map_fn(conversations)
    text_dict = _process(
        sources=[conversations],
        tokenizer=tokenizer,
        has_image=has_image,
        chat_template=chat_template,
    )
    data_dict = dict(input_ids=text_dict["input_ids"][0], labels=text_dict["labels"][0])
    return data_dict


class LLMTokenization(Tokenization):
    def __init__(self, tokenizer, data_args):
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __call__(self, conversations: List[Dict], has_image: bool) -> Dict:
        data_dict = process_data_item(
            conversations=conversations,
            tokenizer=self.tokenizer,
            chat_template=self.data_args.chat_template,
            has_image=has_image,
        )
        return data_dict


class Pi0Tokenization(Tokenization):
    def __init__(self, tokenizer: transformers.GemmaTokenizer, *args, **kwargs):
        self.tokenizer = tokenizer
        self._max_len = tokenizer.model_max_length

    def __call__(self, conversations: List[Dict], **kwargs):
        prompt = conversations[0]["value"]
        cleaned_prompt = prompt.strip().replace('\n', ' ').replace('_', ' ')
        tokens = self.tokenizer.sp_model.encode(cleaned_prompt, add_bos=True) + self.tokenizer.sp_model.encode("\n")
        tokens = tokens[: self._max_len]
        tokens += [0] * (self._max_len - len(tokens))
        return {"input_ids": np.asarray(tokens), "labels": np.asarray(tokens)}
