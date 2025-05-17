from pathlib import Path
import logging
import sentencepiece as spm
import config

def set_logger(log_path):
    """
    使用 logger 代替初学者常用的 print 调试，统一管理训练日志。

    该函数定义了全局 logger，
    可以同时将训练过程信息输出到控制台和日志文件，便于实时监控和后续回顾。
    
    Args:
        log_path (str or Path): 日志文件保存路径
    """
    log_path = Path(log_path)
    
    # 如果log文件存在就清空它
    log_path.write_text('')

    # 获取全局logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 清除可能存在的handlers
    logger.handlers.clear()

    # 设置文件Handler，确保训练信息写入日志文件
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # 设置终端Handler，确保训练过程实时输出到屏幕
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    
    # 将文件Handler和终端Handler加入logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def chinese_tokenizer_load():
    """
    加载中文SentencePiece分词器。
    
    Returns:
        spm.SentencePieceProcessor: 加载好的中文分词器实例
    """
    sp_chn = spm.SentencePieceProcessor()
    sp_chn.Load(config.chn_tokenizer)
    return sp_chn

def english_tokenizer_load():
    """
    加载英文SentencePiece分词器。
    
    Returns:
        spm.SentencePieceProcessor: 加载好的英文分词器实例
    """
    sp_eng = spm.SentencePieceProcessor()
    sp_eng.Load(config.eng_tokenizer)
    return sp_eng