import os
import openai
import requests
from typing import Any, Callable, Dict, Optional

OPENAI_ENGINE = 'text-davinci-003'
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
    openai.api_key = os.environ.get('OPENAI_KEY')
    request = f"{text}\nCreate a short prompt to generate image for this text."
    response = openai.Completion.create(
        engine=OPENAI_ENGINE,
        prompt=request,
        max_tokens=77,
    )
    print(response)
    return response.choices[0]['text'].replace('\n', '').replace('\t', '')


# 3. Grammatical parsing
def _get_grammar_prompt(text: str, **kwargs) -> Optional[str]:
    sentences = text.split('.')[0]
    openai.api_key = os.environ.get('OPENAI_KEY')
    sentence_parts_requests = ('subject', 'predicate', 'action place')
    sentence_parts_responses = []
    sentence = sentences[0]
    for sentence_part in sentence_parts_requests:
        request = f"{sentence}.\n Find {sentence_part} in this sentence. Write just the answer"
        response = openai.Completion.create(
            engine=OPENAI_ENGINE,
            prompt=request,
            max_tokens=10,
        )
        sentence_parts_responses.append(
            response.choices[0]['text'].lower().replace(sentence_part, '') \
                .replace('\n', '').replace('\t', '').replace(':', '').replace('.', ' ')
        )
        print(sentence_parts_responses[-1])
    return '. '.join(sentence_parts_responses)


# public function: get prompt generator
def get_prompt_generator(type: str) -> Optional[Callable]:
    if type == 'Summary':
        return _get_summary
    elif type == 'ChatGPT':
        return _get_chatgpt_prompt
    elif type == 'Grammar parsing':
        return _get_grammar_prompt
    else:
        raise RuntimeError(f"Can't load prompt generator: {type}")
