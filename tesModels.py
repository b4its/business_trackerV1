
import os
import json
import torch
import joblib
import numpy as np
from datetime import datetime
from transformers import pipeline
from ctrain import BisnisAssistantModel

# --- Load MLP Model ---
model = BisnisAssistantModel()
model.load_state_dict(torch.load("model/assistV1.pth"))
model.eval()

scaler_x = joblib.load("data/normData/scaler_x.pkl")
scaler_y = joblib.load("data/normData/scaler_y.pkl")

# --- Load BERT Intent Classifier ---
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# --- Mapping intent ke field numerik ---
INTENT_MAPPING = {
    "tanya_modal": "modal",
    "tanya_profit": "profit",
    "tanya_rugi": "rugi"
}

# --- Prediksi MLP ---
def predict_from_data(pemasukan, pengeluaran, jam_float):
    input_data = np.array([[pemasukan, pengeluaran, jam_float]], dtype=np.float32)
    input_scaled = scaler_x.transform(input_data)
    with torch.no_grad():
        pred_scaled = model(torch.tensor(input_scaled)).numpy()
    pred = scaler_y.inverse_transform(pred_scaled)[0]
    return {
        "modal": pred[0],
        "profit": pred[1],
        "rugi": pred[2]
    }

# --- Intent detection dari pertanyaan user ---
def detect_intent(user_question):
    result = classifier(user_question, list(INTENT_MAPPING.keys()))
    intent = result["labels"][0]
    return INTENT_MAPPING[intent]

# --- Ambil data transaksi terbaru dari data/normData/*.json ---
def ambil_data_terbaru_from_norm():
    latest_item = None
    latest_time = None

    norm_dir = "data/normData"
    for file in os.listdir(norm_dir):
        if not file.endswith(".json") or file == "normalization_stats.json":
            continue

        with open(os.path.join(norm_dir, file)) as f:
            data = json.load(f)

        for item in data:
            try:
                waktu = datetime.fromisoformat(item["waktu"])
                if latest_time is None or waktu > latest_time:
                    latest_time = waktu
                    latest_item = item
            except Exception:
                continue

    if latest_item:
        pemasukan = latest_item["total_pemasukan"]
        pengeluaran = latest_item["total_pengeluaran"]
        jam = datetime.fromisoformat(latest_item["waktu"]).hour / 24.0
        return pemasukan, pengeluaran, jam
    else:
        raise ValueError("Tidak ada data transaksi valid ditemukan")

# --- Main Loop ---
if __name__ == "__main__":
    print("Business Tracker Siap, silahkan tanyakan sesuatu...")
    while True:
        try:
            user_input = input("\nAnda: ").strip()
            if user_input.lower() in ["exit", "quit", "keluar"]:
                print("Sampai jumpa!")
                break

            # Langkah 1: Deteksi intent dari pertanyaan
            target_field = detect_intent(user_input)

            # Langkah 2: Ambil data transaksi terbaru dari normData
            pemasukan, pengeluaran, jam = ambil_data_terbaru_from_norm()

            # Langkah 3: Prediksi
            hasil = predict_from_data(pemasukan, pengeluaran, jam)

            # Langkah 4: Jawaban
            print(f"Jawaban: {target_field.capitalize()} Anda diperkirakan sebesar Rp {hasil[target_field]:,.0f}")

        except Exception as e:
            print(f"Terjadi kesalahan: {e}")
