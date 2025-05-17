import sentencepiece as spm

def train(input_file, vocab_size, model_name, model_type, character_coverage):
    """
    使用 SentencePiece 训练分词器模型。
    
    Args:
        input_file (str): 输入语料文件路径，一行一句
        vocab_size (int): 词表大小，比如 32000
        model_name (str): 输出模型名前缀
        model_type (str): 分词模型类型，比如 'bpe' 或 'unigram'
        character_coverage (float): 字符覆盖率，中文一般用0.9995，英文用1.0
    """
    cmd = (
        f"--input={input_file} "
        f"--model_prefix={model_name} "
        f"--vocab_size={vocab_size} "
        f"--model_type={model_type} "
        f"--character_coverage={character_coverage} "
        "--pad_id=0 "
        "--unk_id=1 "
        "--bos_id=2 "
        "--eos_id=3"
    )
    spm.SentencePieceTrainer.Train(cmd)

def run():
    """
    分别训练英文和中文的分词器。
    """
    # 英文分词器训练
    en_input = '../data/corpus.en'
    en_vocab_size = 32000
    en_model_name = 'eng'
    en_model_type = 'bpe'  
    en_character_coverage = 1.0
    train(en_input, en_vocab_size, en_model_name, en_model_type, en_character_coverage)
    
    # 中文分词器训练
    ch_input = '../data/corpus.ch' 
    ch_vocab_size = 32000
    ch_model_name = 'chn'
    ch_model_type = 'bpe'
    ch_character_coverage = 0.9995
    train(ch_input, ch_vocab_size, ch_model_name, ch_model_type, ch_character_coverage)

def test():
    """
    用于测试训练好的分词器效果。
    """
    sp = spm.SentencePieceProcessor()
    text = "美国老头特朗普今日给我按摩"

    sp.Load('./chn.model')
    print(sp.EncodeAsPieces(text))
    print(sp.EncodeAsIds(text))
    a = [313, 29475, 29335, 277, 7420, 24760, 29728, 29897]
    print(sp.decode_ids(a))

if __name__ == '__main__':
    # run()
    test()
