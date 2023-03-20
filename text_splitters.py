from typing import List


# 1. Split text into n equal parts
def split_text(
        text: str,
        n: int,
) -> List[str]:
    sentences = text.replace('\n', '').replace('\t', '').split('.')
    sample_size = len(sentences) // n
    text_splitted = []
    start_id = 0
    for i in range(n):
        if i == n - 1:
            # last part may be a bit longer
            text_splitted.append('.'.join(sentences[start_id:]))
        else:
            text_splitted.append('.'.join(sentences[start_id:start_id + sample_size]))
        start_id += sample_size
    return text_splitted
