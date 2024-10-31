from transformers import pipeline

# Carica il modello fine-tuned dal percorso specificato
qa_model = pipeline("text-generation", model="C:/Users/Fabio/Desktop/GPT-medium2/fine_tuned_qa_model", tokenizer="C:/Users/Fabio/Desktop/GPT-medium2/fine_tuned_qa_model")

def chat():
    print("Chatbot attivo! Chiedimi qualsiasi cosa sulla storia. Digita 'exit' per uscire.")
    
    while True:
        # Ottieni la domanda dall'utente
        question = input("You: ")
        
        # Condizione per uscire dalla chat
        if question.lower() == "exit":
            print("Chatbot: Grazie per aver parlato con me! A presto!")
            break
        
        # Prompt con struttura Q&A fissa
        prompt = f"{question}\nAnswer:"
        
        # Genera la risposta
        output = qa_model(prompt, max_length=50, num_return_sequences=1, truncation=True)
        
        # Estrai la risposta generata
        answer = output[0]["generated_text"].replace(prompt, "").strip()
        
        # Mostra la risposta
        print("Chatbot:", answer)

# Avvia la chat
chat()
