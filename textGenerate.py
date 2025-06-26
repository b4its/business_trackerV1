import json
import random
import os
from sentence_transformers import SentenceTransformer, util

# Load model SBERT (compatible dengan TF, meskipun via PyTorch interface)
model = SentenceTransformer("all-mpnet-base-v2")

# Intent template
intent_templates = {
    "tanya_profit": [
        "Berapa keuntungan saya hari ini?",
        "Keuntungan saya minggu ini berapa?",
        "Apakah saya untung kemarin?",
        "Tolong hitungkan profit saya",
        "Saya untung atau tidak?",
        "Laba saya hari ini?",
        "Berapakah profit hari ini?"
    ],
    "tanya_rugi": [
        "Saya rugi berapa?",
        "Kerugian hari ini berapa?",
        "Apakah saya mengalami kerugian?",
        "Tolong hitungkan rugi saya",
        "Rugi saya hari ini berapa?",
        "Apakah ini minus?",
        "Berapa saya kehilangan uang?"
    ],
    "tanya_modal": [
        "Modal awal saya berapa?",
        "Berapa modal saya tadi pagi?",
        "Modal yang saya pakai hari ini?",
        "Saya mulai dengan modal berapa?",
        "Tolong hitung modal saya",
        "Berapa investasi awal saya?",
        "Saya pakai modal berapa hari ini?"
    ]
}

# Contoh kandidat template variasi (manual & bisa diperluas)
paraphrase_variants = {
    "tanya_profit": [
        "Saya dapat untung hari ini?",
        "Berapa laba saya saat ini?",
        "Profit saya sekarang berapa?",
        "Hari ini ada keuntungan?",
        "Berapa besar keuntungan hari ini?",
        "Berapa saya mendapat pemasukan bersih?"
    ],
    "tanya_rugi": [
        "Saya mengalami kerugian?",
        "Berapa minus saya hari ini?",
        "Hari ini saya rugi atau tidak?",
        "Ada pengeluaran lebih besar hari ini?",
        "Saya kehilangan uang berapa?",
        "Seberapa besar kerugian saya?"
    ],
    "tanya_modal": [
        "Berapa dana awal saya hari ini?",
        "Modal saya di awal berapa?",
        "Saya mulai dengan berapa uang?",
        "Investasi saya di awal berapa?",
        "Modal saya sekarang berapa?",
        "Modal kerja hari ini berapa?"
    ]
}

# Generate dataset
dataset = []

for intent, base_templates in intent_templates.items():
    candidates = paraphrase_variants[intent]
    for template in base_templates:
        template_embedding = model.encode(template, convert_to_tensor=True)
        candidates_embedding = model.encode(candidates, convert_to_tensor=True)

        cos_scores = util.pytorch_cos_sim(template_embedding, candidates_embedding)[0]
        top_k = min(5, len(candidates))
        top_indices = cos_scores.topk(k=top_k).indices

        dataset.append({"text": template, "intent": intent})  # asli

        for idx in top_indices:
            paraphrased = candidates[idx]
            dataset.append({
                "text": paraphrased,
                "intent": intent
            })

# Save to file
os.makedirs("data", exist_ok=True)
with open("data/textData.json", "w") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print("Dataset intent + paraphrase SBERT berhasil dibuat: data/textData.json")
