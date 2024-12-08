from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Specify the model name
model_name = "google/t5-v1_1-xxl"
save_directory = "/usr3/hcontant/pixart-project-recent/ckpts/t5"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Save the tokenizer and model locally
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

print(f"Model and tokenizer saved to {save_directory}")
