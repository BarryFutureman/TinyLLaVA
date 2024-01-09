from llava.model.language_model.llava_llama import LlamaForCausalLM
from transformers import BitsAndBytesConfig, AutoTokenizer
import torch


# Base Model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


base_model_name = "C:\Files\PycharmProjects\TinyLLaVA\merged\TinyLLaVA_1.1B_align_consolidated"

model = LlamaForCausalLM.from_pretrained(base_model_name,
                                         torch_dtype=torch.float16,
                                         quantization_config=bnb_config,
                                         local_files_only=True,
                                         # cache_dir="cache/models",
                                         )

tokenizer = AutoTokenizer.from_pretrained(base_model_name, cache_dir="cache/models")


def make_prompt(entry):
    return f"\n<|user|>\n{entry}\n<|assistant|>\n"


def run_model(text_prompt):
    model_input = tokenizer(
        text_prompt,
        return_tensors="pt").to("cuda")

    input_length = len(model_input['input_ids'][0])

    with torch.no_grad():
        generation_kwargs = {
            "min_length": -1,
            "temperature": 1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
            "max_new_tokens": 100,
        }

        full_tokens = model.generate(**model_input, **generation_kwargs)[0]
        decoded_tokens = tokenizer.decode(full_tokens[input_length:], skip_special_tokens=True)
    return decoded_tokens


if __name__ == '__main__':
    context = "<|system|>\nThe following is a conversation between a user and a helpful assistant.\n"
    while True:
        print(context)
        context += make_prompt(input("You: "))
        response = run_model(context)
        context += response[:-1]



