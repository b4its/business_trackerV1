# train_ai_assistant.py

import os
import json
import torch
import numpy as np
import joblib
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer, util

# ------------------
# Model Prediksi Bisnis (MLP multi-output)
# ------------------

class BisnisAssistantModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # modal, profit, rugi
        )

    def forward(self, x):
        return self.model(x)

# ------------------
# Load dan Normalisasi Dataset
# ------------------
X, y = [], []

normal_dir = "data/normal"
norm_dir = "data/normData"
os.makedirs(norm_dir, exist_ok=True)

for label_folder in os.listdir(normal_dir):
    label_path = os.path.join(normal_dir, label_folder)
    if not os.path.isdir(label_path):
        continue

    for file in os.listdir(label_path):
        if not file.endswith(".json"):
            continue
        with open(os.path.join(label_path, file)) as f:
            data = json.load(f)

        out_file = os.path.join(norm_dir, f"dataset_{label_folder}_{file}")
        with open(out_file, "w") as out:
            json.dump(data, out, indent=2)

        for item in data:
            pemasukan = item["total_pemasukan"]
            pengeluaran = item["total_pengeluaran"]
            waktu = datetime.fromisoformat(item["waktu"])
            jam = waktu.hour / 24.0

            X.append([pemasukan, pengeluaran, jam])
            modal = item["modal_awal"]
            rugi = item["rugi"]
            profit = pemasukan - pengeluaran if rugi == 0 else 0
            y.append([modal, profit, rugi])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_x.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

joblib.dump(scaler_x, os.path.join(norm_dir, "scaler_x.pkl"))
joblib.dump(scaler_y, os.path.join(norm_dir, "scaler_y.pkl"))

with open(os.path.join(norm_dir, "normalization_stats.json"), "w") as f:
    json.dump({"x_max": scaler_x.data_max_.tolist(), "y_max": scaler_y.data_max_.tolist()}, f, indent=2)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# ------------------
# Training Model Prediksi
# ------------------
model = BisnisAssistantModel()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

X_tensor = torch.tensor(X_train)
y_tensor = torch.tensor(y_train)

for epoch in range(1000):
    model.train()
    output = model(X_tensor)
    loss = loss_fn(output, y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/assistV1.pth")

# ------------------
# Text Paraphrasing menggunakan Sentence-BERT
# ------------------
with open("data/textData.json") as f:
    text_data = json.load(f)

# Intent pool: kumpulan template pertanyaan
intent_templates = [
    "Berapa keuntungan saya hari ini?",
    "Keuntungan saya minggu ini berapa?",
    "Apakah saya untung kemarin?",
    "Tolong hitungkan profit saya",
    "Saya untung atau tidak?",
    "Laba saya hari ini?",
    "Berapakah profit hari ini?"
]

# Load model pretrained
model_sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2')
template_embeddings = model_sbert.encode(intent_templates, convert_to_tensor=True)

augmented_data = []

for entry in text_data:
    question = entry["text"]
    intent = entry["intent"]

    try:
        question_embedding = model_sbert.encode(question, convert_to_tensor=True)
        cos_scores = util.cos_sim(question_embedding, template_embeddings)[0]
        top_indices = cos_scores.argsort(descending=True)[:3]

        for idx in top_indices:
            augmented_data.append({
                "text": intent_templates[idx],
                "intent": intent
            })
    except Exception as e:
        print(f"Skip paraphrase for '{question}' due to error: {e}")
        continue

# Gabungkan original + augmented
text_data.extend(augmented_data)
with open("data/augmentedText.json", "w") as f:
    json.dump(text_data, f, indent=2)

# print("✅ Model prediksi bisnis dilatih dan pertanyaan diparafrase menggunakan Sentence-BERT.")
