from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
print("Caricamento del modello...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
print("Modello caricato con successo!")

# Configura la GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
