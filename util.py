def apply_llama2_training_format(prompt, system_prompt="You are a helpful AI agent."):
    """
    Returns a correctly formatted LLaMA-2 chat prompt.
    """
    return f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"

def apply_llama3_training_format(prompt, system_prompt="You are a helpful AI agent."):
    """
    Returns a correctly formatted LLaMA-3 chat prompt.
    """

    BEGIN = "<|begin_of_text|>"
    START = "<|start_header_id|>"
    END = "<|end_header_id|>"
    EOT = "<|eot_id|>"

    return (
        f"{BEGIN}"
        f"{START}system{END}\n"
        f"{system_prompt}{EOT}\n"
        f"{START}user{END}\n"
        f"{prompt}{EOT}\n"
        f"{START}assistant{END}\n"
    )
def apply_mistral_chat_format(text):
    return f"<s>[INST] {text.strip()} [/INST]"

def apply_gemma2_chat_format(prompt: str) -> str:
    return (
        "<bos>"
        "<start_of_turn>user\n"
        f"{prompt}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
def apply_qwen2_chat_format(prompt: str) -> str:
    return (
        "<|im_start|>user\n"
        f"{prompt}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

