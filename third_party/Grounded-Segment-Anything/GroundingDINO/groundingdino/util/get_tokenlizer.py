from transformers import AutoTokenizer, BertModel, RobertaModel
import os
from pathlib import Path

DEFAULT_BERT_PATH = str(Path(__file__).resolve().parents[5] / "third_party" / "bert-base-uncased")


def _resolve_local_bert_path(bert_base_uncased_path=None):
    return (
        bert_base_uncased_path
        or os.environ.get("GROUNDINGDINO_BERT_PATH")
        or DEFAULT_BERT_PATH
    )



def get_tokenlizer(text_encoder_type, bert_base_uncased_path=None):
    """Tokenizer loader that prefers a local model path."""
    
    if text_encoder_type == "bert-base-uncased":
        local_path = _resolve_local_bert_path(bert_base_uncased_path)
        
        if not os.path.isdir(local_path):
            raise FileNotFoundError(
                f"本地BERT模型未找到: {local_path}\n"
                f"请下载BERT模型到该路径，或使用以下命令下载:\n"
                f"python -c \"from transformers import AutoTokenizer; "
                f"AutoTokenizer.from_pretrained('bert-base-uncased').save_pretrained('{local_path}')\""
            )
        
        print(f"Using local BERT tokenizer from: {local_path}")
        return AutoTokenizer.from_pretrained(local_path, local_files_only=True)
    
    # Only local bert-base-uncased is supported.
    raise ValueError(f"Only local bert-base-uncased is supported, got: {text_encoder_type}")


def get_pretrained_language_model(text_encoder_type, bert_base_uncased_path=None):
    """Language-model loader that prefers a local model path."""
    
    if text_encoder_type == "bert-base-uncased":
        local_path = _resolve_local_bert_path(bert_base_uncased_path)
        
        if not os.path.isdir(local_path):
            raise FileNotFoundError(
                f"本地BERT模型未找到: {local_path}\n"
                f"请下载BERT模型到该路径，或使用以下命令下载:\n"
                f"python -c \"from transformers import BertModel; "
                f"BertModel.from_pretrained('bert-base-uncased').save_pretrained('{local_path}')\""
            )
        
        print(f"Using local BERT model from: {local_path}")
        return BertModel.from_pretrained(local_path, local_files_only=True)
    
    # Only local bert-base-uncased is supported.
    raise ValueError(f"Only local bert-base-uncased is supported, got: {text_encoder_type}")