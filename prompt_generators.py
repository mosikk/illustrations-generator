import requests
from typing import Any, Callable, Dict, Optional

SUMMARIZER_URL = 'https://api.aicloud.sbercloud.ru/public/v2/summarizator/predict'


# 1. Get summary (by Sber summarizer)
def _create_summarization_request(text: str, **kwargs) -> Dict[str, Any]:
    req = dict()
    req['instances'] = list()
    req['instances'].append(dict())
    req['instances'][0]['text'] = text
    for key, value in kwargs.items():
        req['instances'][0][key] = value
    return req


def _get_summary(text: str, **kwargs) -> Optional[str]:
    request = _create_summarization_request(text, **kwargs)
    response = requests.post(SUMMARIZER_URL, json=request).json()

    if response['comment'] != 'Ok!':
        raise RuntimeError(f"Can't get text summary: {response['comment']}")

    return response['prediction_best']['bertscore']


# 2. Chat GPT
def _get_chatgpt_prompt(text: str, **kwargs) -> Optional[str]:
    # Coming soon
    pass


# 3. Grammatical parsing
def _get_grammar_prompt(text: str, **kwargs) -> Optional[str]:
    # Coming soon
    pass


# public function: get prompt generator
def get_prompt_generator(type: str) -> Optional[Callable]:
    if type == 'Summary':
        return _get_summary
    else:
        raise RuntimeError(f"Can't load prompt generator: {type}")
