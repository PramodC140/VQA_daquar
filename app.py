# -*- coding: utf-8 -*-
"""
VQA Flask API with CORS support
"""

import torch
import os
from pathlib import Path
import numpy as np
from PIL import Image
from torch import nn
from torchvision import transforms, models
from transformers import BertTokenizer, BertModel
import json
from flask import Flask, request, jsonify
import werkzeug.utils
from flask_cors import CORS

class VQAModel_trained(nn.Module):
    def __init__(self, num_answers):
        super(VQAModel_trained, self).__init__()
        self.cnn = models.resnet50(pretrained=True)  # Pre-trained ResNet-50
        self.cnn.fc = nn.Identity()  # Remove classification layer

        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.fc1 = nn.Linear(2048 + 768, 1024)
        self.fc2 = nn.Linear(1024, num_answers)
        self.dropout = nn.Dropout(0.5)

    def forward(self, images, input_ids, attention_mask):
        image_features = self.cnn(images)
        outputs = self.bert(input_ids, attention_mask)
        question_features = outputs.last_hidden_state[:, 0, :]
        combined_features = torch.cat((image_features, question_features), dim=1)
        x = self.fc1(combined_features)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model
device = torch.device('cpu')
model_path = "/home/narendra/Desktop/streamlit/best_model.pth"  # Update this path as needed
loaded_model = VQAModel_trained(num_answers=582)
checkpoint = torch.load(model_path, map_location=device)
loaded_model.load_state_dict(checkpoint['model_state_dict'])
loaded_model.to(device)
loaded_model.eval()

# Initialize tokenizer and transform
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create upload directory
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def vqa_prediction(question, image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    inputs = tokenizer(question, return_tensors="pt", padding="max_length", truncation=True, max_length=30)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    with torch.no_grad():
        output = loaded_model(image, input_ids, attention_mask)
    
    predicted_idx = torch.argmax(output, dim=1).item()

    file_path = Path("answer_space.txt") 
    with file_path.open() as f:
        answer_space = f.read().splitlines()
    
    predicted_answer = answer_space[predicted_idx]
    return predicted_answer

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    if 'question' not in request.form:
        return jsonify({'error': 'No question provided'}), 400
    
    # Get the question
    question = request.form['question']
    
    # Save the uploaded image
    image_file = request.files['image']
    filename = werkzeug.utils.secure_filename(image_file.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image_file.save(image_path)
    
    # Make prediction
    try:
        answer = vqa_prediction(question, image_path)
        return jsonify({
            'question': question,
            'answer': answer
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add endpoint to match React frontend's expected API
@app.route('/answer', methods=['POST'])
def answer():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    if 'message' not in request.form:  # React frontend sends 'message' instead of 'question'
        return jsonify({'error': 'No message provided'}), 400
    
    # Get the question from the message field
    question = request.form['message']
    
    # Save the uploaded image
    image_file = request.files['image']
    filename = werkzeug.utils.secure_filename(image_file.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image_file.save(image_path)
    
    # Make prediction
    try:
        answer = vqa_prediction(question, image_path)
        return jsonify({
            'answer': answer  # React frontend expects 'answer' field
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)