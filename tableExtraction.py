from PIL import Image
import pytesseract
import numpy as np

import torch
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD


import torch
from transformers import BertTokenizer, BertForSequenceClassification


# class KeyValueClassifier(Module):
#     def __init__(self):
#         super(KeyValueClassifier, self).__init__()

#         self.cnn_layers = Sequential(
#             # Defining a 2D convolution layer
#             Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
#             BatchNorm2d(4),
#             ReLU(inplace=True),
#             MaxPool2d(kernel_size=2, stride=2),
#             # Defining another 2D convolution layer
#             Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
#             BatchNorm2d(4),
#             ReLU(inplace=True),
#             MaxPool2d(kernel_size=2, stride=2),
#         )

#         self.linear_layers = Sequential(
#             Linear(4 * 7 * 7, 10)
#         )

#     # Defining the forward pass
#     def forward(self, x):
#         x = self.cnn_layers(x)
#         x = x.view(x.size(0), -1)
#         x = self.linear_layers(x)
#         return x


def extract_text_from_image(image):
    # Tesseract expects images, so we need to convert our PDF to image
    # You can use the pdf2image library for this
    text = pytesseract.image_to_string(image)
    
    return text


# Load the BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# The text to be classified and its true label
text = extract_text_from_image("page_0.png")
labels_gold = torch.tensor([0]).unsqueeze(0)  # assuming it's binary classification

# Convert the text to a list of tokens and truncate if necessary
tokens = tokenizer(text=text, return_tensors="pt", truncation=True, max_length=512)


# Predict the class of the text
outputs = model(**tokens, labels=labels_gold)

# Get the logits
logits = outputs.logits

# Get the predicted label
predicted_label = torch.argmax(logits, dim=-1)

# Calculate the accuracy
accuracy = (predicted_label == labels_gold).sum().item() / len(predicted_label)

print("Accuracy:", accuracy)

