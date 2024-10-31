import json
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Carica il tokenizer e il modello GPT-2 Medium
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
tokenizer.pad_token = tokenizer.eos_token  # Imposta il token di fine sequenza come token di padding
model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
model.to("cuda" if torch.cuda.is_available() else "cpu")

# Crea la classe del dataset per il fine-tuning
class QADataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        # Carica le domande e risposte
        with open(file_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Prendi il prompt e la risposta dal dataset
        prompt = self.data[idx]["prompt"]
        response = self.data[idx]["response"]
        
        # Crea il testo da dare in input al modello
        text = f"Question: {prompt}\nAnswer: {response}"
        
        # Codifica il testo e la risposta
        encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        
        # Gli input_ids e gli attention_mask sono le features di input
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids}

# Inizializza il dataset
dataset = QADataset("qa_dataset.json", tokenizer)

# Configura i parametri di addestramento
training_args = TrainingArguments(
    output_dir="./qa_model",
    per_device_train_batch_size=1,
    num_train_epochs=10,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10
)

# Inizializza il Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# Avvia il fine-tuning
trainer.train()

# Salva il modello addestrato
model.save_pretrained("./fine_tuned_qa_model")
tokenizer.save_pretrained("./fine_tuned_qa_model")
