
from Classifiers.Classifier import Classifier

# Import libraries
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
from tqdm import tqdm

class DeviceDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, device):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)
        self.device = device

    def __iter__(self):
        for batch in super().__iter__():
            yield tuple(t.to(self.device) for t in batch)

class RoBERTa(Classifier):
    def __init__(self, config: dict):
        super().__init__(config)
        if self.model == None:
            self.build_roberta_classifier()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def load_model(self, path: str) -> int:
        """
        loads the model
        :param path: path to load the model
        """
        self.model = RobertaForSequenceClassification.from_pretrained(path)
        self.tokenizer = RobertaTokenizer.from_pretrained(path)
        return 1
    
    def train(self, X: np.array, y: np.array) -> int:
        """
        Trains the RoBERTa Classifier
        :param X: training data
        :param y: training labels
        :return: history of the training
        """
        X = X.tolist()
        # Tokenize input texts
        tokenized_texts = self.tokenizer(X, padding=True, truncation=True, return_tensors="pt")

        # Create TensorDataset for input and labels
        input_ids = tokenized_texts['input_ids']
        attention_mask = tokenized_texts['attention_mask']
        labels = torch.tensor(y)
        dataset = TensorDataset(input_ids, attention_mask, labels)

        # Set device (CPU or GPU)
        print('this is the device: ', self.device)

        # Create DataLoader
        batch_size = self.config['batch_size']
        dataloader = DeviceDataLoader(dataset, batch_size=batch_size, shuffle=True, device=self.device)
        dataloader_with_progress = tqdm(dataloader, desc="Training")

        # Set optimizer and learning rate
        optimizer = AdamW(self.model.parameters(), lr=self.config['lr'])

        loss_fn = torch.nn.CrossEntropyLoss()

        # Training loop
        num_epochs = self.config['epochs']
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            for batch in dataloader_with_progress:
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits.float()
                labels = labels.long()
                
                loss = loss_fn(logits, labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        dataloader_with_progress.close()
        
        return 1
    
    def build_roberta_classifier(self) -> int:
        """
        Builds the roberta model
        The model is built using the transformers library
        """
        model_name = 'roberta-base'
        self.model = RobertaForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        return 1
    
    def save(self, path: str) -> int:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        return 1

    def predict(self, X: np.array) -> np.array:
        self.model.eval()

        predictions = []
        for input_text in tqdm(X, desc="Predicting"):
            tokenized_input = self.tokenizer(input_text, return_tensors='pt')
            tokenized_input = {key: value.to(self.device) for key, value in tokenized_input.items()}

            with torch.no_grad():
                outputs = self.model(**tokenized_input)

            probas = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probas).item()
            predictions.append(predicted_class)

        predictions = [-1 if p == 0 else 1 for p in predictions]
        return np.array(predictions)

