import json
from pathlib import Path

def main():
    # 定义 JSON 文件所在目录和要处理的文件名列表
    json_dir = Path('./json')
    files = ['train', 'dev', 'test']

    # 定义要输出的中文和英文语料文件路径
    ch_corpus_path = Path('./corpus.ch')
    en_corpus_path = Path('./corpus.en')

    # 用于暂存中文和英文句子的列表
    ch_lines = []
    en_lines = []

    # 遍历三个数据集文件
    for file_name in files:
        json_file = json_dir / f'{file_name}.json'
        
        # 检查文件是否存在
        if not json_file.exists():
            raise FileNotFoundError(f'Error: {json_file} does not exist. Please check your data files.')

        # 读取 JSON 文件内容
        with json_file.open('r', encoding='utf-8') as f:
            corpus = json.load(f)

        # 将英文和中文句子分别处理后加入列表（去除首尾空白）
        for en_text, ch_text in corpus:
            en_lines.append(en_text.strip() + '\n')
            ch_lines.append(ch_text.strip() + '\n')

        print(f'Loaded {len(corpus)} sentence pairs from {json_file}')

    # 将英文句子写入英文语料文件
    with en_corpus_path.open('w', encoding='utf-8') as f:
        f.writelines(en_lines)

    # 将中文句子写入中文语料文件
    with ch_corpus_path.open('w', encoding='utf-8') as f:
        f.writelines(ch_lines)

    # 打印加载完成的信息
    print(f'Total English sentences: {len(en_lines)}')
    print(f'Total Chinese sentences: {len(ch_lines)}')

if __name__ == "__main__":
    main()
