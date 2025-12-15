import json
import pandas as pd
from sentence_transformers import util, SentenceTransformer
import numpy as np

# --- Thresholds dari Notebook 2 (diambil dari logika) ---
NON_RELEVANT_SIM_THRESHOLD = 0.2
MIN_LENGTH_FOR_SCORE = 5

# --- MODEL CACHING ---
def load_embedder_model():
    """Memuat model SentenceTransformer untuk scoring."""
    try:
        return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') 
    except Exception as e:
        print(f"Error loading SentenceTransformer: {e}")
        return None

# --- FUNGSI RELEVANSI (DARI NOTEBOOK 2) ---

def is_non_relevant(text: str) -> bool:
    """Mengecek apakah transkrip cenderung tidak relevan/kosong."""
    t = text.strip().lower()
    if len(t) == 0:
        return True

    if len(t.split()) <= 2: # Transkrip yang sangat pendek
        return True

    # Frasa umum yang menunjukkan ketidakmampuan menjawab
    non_answers = [
        "i don't know", "i dont know", "no idea",
        "i have no idea", "not sure", "i can't answer",
        "i cannot answer", "i don't understand",
        "i dont understand", 
    ]
    if any(na in t for na in non_answers):
        return True
    
    return False

# --- FUNGSI CONFIDENCE SCORE (DARI NOTEBOOK 1) ---

def compute_confidence_score(transcript: str, text_confidence: int) -> int:
    """
    Menghitung skor kepercayaan (confidence score) akhir.
    text_confidence adalah output dari Whisper (avg_log_prob).
    """
    if not transcript or is_non_relevant(transcript):
        # Jika transkrip kosong atau tidak relevan, confidence score rendah
        return 0.1 
    
    # Skala ulang log_prob (-1.0 hingga 0.0) ke (0.0 hingga 1.0)
    # Clamp nilai agar tidak melebihi 100% atau di bawah 0.1%
    scaled_confidence = np.clip(1.0 + text_confidence, 0.001, 1.0)
    
    # Tambahkan penalti kecil jika transkrip sangat pendek
    if len(transcript.split()) < 5:
        scaled_confidence *= 0.8 # Penalti 20%
        
    return scaled_confidence

# --- FUNGSI SCORING SEMANTIK (DARI NOTEBOOK 2) ---

def score_with_rubric(question_id, question_text, answer, rubric_data, model_embedder):
    """
    Menghitung skor berdasarkan perbandingan semantik dengan rubrik.
    question_id: Key untuk RUBRIC_DATA (e.g., 'q1', 'q2', dst.)
    rubric_data[question_id]["ideal_points"] adalah dictionary {'4': [...], '3': [...], ...}
    """
    if model_embedder is None:
        return 0, "Error: Embedding model failed to load."

    # question_id sudah berupa 'q1', 'q2', dst. Langsung digunakan.
    # Ambil rubrik untuk pertanyaan ini
    rubric_entry = rubric_data.get(question_id, {})
    rubric = rubric_entry.get("ideal_points", {})
    a = answer.strip()

    if is_non_relevant(a) or len(a.split()) < MIN_LENGTH_FOR_SCORE:
        # Mengembalikan skor 0 dan alasan dari rubrik level 0
        # FIX: Akses key "0" sebagai string
        return 0, rubric.get("0", ["Unanswered"])[0] 

    embedding_a = model_embedder.encode(a.lower())

    # Fungsi untuk menghitung kecocokan
    def count_matches(indicators, threshold=0.40):
        hits = 0
        # Pre-encode semua indikator untuk efisiensi
        embeddings_indicators = model_embedder.encode([ind.lower() for ind in indicators])
        
        # Hitung cosine similarity
        similarities = util.cos_sim(embedding_a, embeddings_indicators).flatten()
        
        for sim in similarities:
            if sim.item() >= threshold:
                hits += 1
        return hits

    # Iterasi dari skor tertinggi ke terendah (4, 3, 2, 1)
    # Gunakan string keys karena dimuat dari JSON
    for point_str in ["4", "3", "2", "1"]:
        point = int(point_str)
        # FIX: Akses key rubrik sebagai string
        indicators = rubric.get(point_str)
        
        if not indicators: continue

        hits = count_matches(indicators)
        
        # Logika Min hits:
        if point == 4:
            min_hits = max(1, int(len(indicators) * 0.6))
        elif point == 3:
            min_hits = max(1, int(len(indicators) * 0.5))
        else: # Untuk point 2 dan 1
            min_hits = 1

        if hits >= min_hits:
            # Mengembalikan skor tertinggi yang memenuhi kriteria
            # FIX: Akses alasan rubrik sebagai string point_str
            return point, rubric.get(point_str, [f"Score {point} achieved"])[0] 

    # Jika tidak ada yang cocok sama sekali, berikan skor 1 (atau minimal)
    # FIX: Akses key "1" sebagai string
    return 1, rubric.get("1", ["Minimal or Vague Response"])[0]
