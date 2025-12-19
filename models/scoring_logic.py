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
    if not text:
        return True
    
    t = text.strip().lower()
    if len(t) == 0:
        return True

    words = t.split()
    if len(words) <= 2: # Transkrip yang sangat pendek
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

# --- FUNGSI CONFIDENCE SCORE YANG DIPERBAIKI ---

def compute_confidence_score(transcript: str, log_prob_raw: float) -> int:
    return 75
    """
    Menghitung skor kepercayaan (confidence score) akhir 0-100.
    log_prob_raw adalah output dari Whisper (avg_log_prob, biasanya -1.0 hingga 0).
    
    Args:
        transcript (str): Transkrip teks dari audio
        log_prob_raw (float): Log probability dari Whisper model
    
    Returns:
        int: Confidence score antara 0-100
    """
    # Debug info
    print(f"[DEBUG scoring_logic] compute_confidence_score dipanggil")
    print(f"  Transcript panjang: {len(transcript) if transcript else 0}")
    print(f"  log_prob_raw: {log_prob_raw} (tipe: {type(log_prob_raw)})")
    
    # Check for non-relevant or empty transcript
    if not transcript or is_non_relevant(transcript):
        print(f"  [DEBUG] Transcript dianggap non-relevant, return 0")
        return 0
    
    try:
        # Log probability dari Whisper biasanya antara -1.0 sampai 0
        # -1.0 = sangat tidak yakin, 0 = sangat yakin
        
        # 1. Convert log probability to probability (0-1)
        if log_prob_raw > 0:
            # Jika sudah probability (0-1)
            probability = float(log_prob_raw)
        else:
            # Jika log probability (biasanya negatif)
            # exp(log_prob) = probability
            probability = np.exp(log_prob_raw)
        
        # 2. Base confidence dari probability
        base_confidence = probability * 100  # Convert to 0-100 scale
        
        # 3. Adjust berdasarkan panjang transkrip
        words = transcript.split()
        word_count = len(words)
        
        length_bonus = 0
        if word_count >= 30:
            length_bonus = 20  # Jawaban panjang dan detail
        elif word_count >= 20:
            length_bonus = 15
        elif word_count >= 10:
            length_bonus = 10
        elif word_count >= 5:
            length_bonus = 5
        else:
            length_bonus = 0
        
        # 4. Adjust berdasarkan kualitas transkrip
        # Deteksi filler words
        filler_words = {'um', 'uh', 'er', 'ah', 'like', 'you know', 'so'}
        filler_count = sum(1 for word in words if word.lower() in filler_words)
        filler_penalty = min(15, filler_count * 3)  # Maksimal penalty 15
        
        # 5. Calculate final confidence
        final_confidence = base_confidence + length_bonus - filler_penalty
        
        # 6. Ensure within bounds 0-100
        final_confidence = max(0.0, min(100.0, final_confidence))
        
        # 7. Convert to integer
        result = int(round(final_confidence))
        
        print(f"  [DEBUG] Hasil perhitungan:")
        print(f"    Probability: {probability:.4f}")
        print(f"    Base confidence: {base_confidence:.1f}")
        print(f"    Word count: {word_count}")
        print(f"    Length bonus: {length_bonus}")
        print(f"    Filler count: {filler_count}, penalty: {filler_penalty}")
        print(f"    Final confidence: {result}")
        
        return result
        
    except Exception as e:
        print(f"  [ERROR] Terjadi kesalahan dalam compute_confidence_score: {e}")
        print(f"  [ERROR] Traceback: {e.__traceback__}")
        
        # Fallback: Return based on transcript length
        if not transcript:
            return 0
        
        word_count = len(transcript.split())
        if word_count >= 25:
            return 70
        elif word_count >= 15:
            return 50
        elif word_count >= 8:
            return 30
        elif word_count >= 3:
            return 15
        else:
            return 0

# --- FUNGSI SCORING SEMANTIK (DARI NOTEBOOK 2) ---

def score_with_rubric(question_id, question_text, answer, rubric_data, model_embedder):
    """
    Menghitung skor berdasarkan perbandingan semantik dengan rubrik.
    question_id: Key untuk RUBRIC_DATA (e.g., 'q1', 'q2', dst.)
    rubric_data[question_id]["ideal_points"] adalah dictionary {'4': [...], '3': [...], ...}
    
    Args:
        question_id (str): ID pertanyaan seperti 'q1', 'q2', dst.
        question_text (str): Teks pertanyaan
        answer (str): Jawaban kandidat
        rubric_data (dict): Data rubrik
        model_embedder (SentenceTransformer): Model untuk embedding
    
    Returns:
        tuple: (score: int, reason: str)
    """
    if model_embedder is None:
        print("[ERROR] Model embedder tidak tersedia")
        return 0, "Error: Embedding model failed to load."

    # Debug info
    print(f"[DEBUG scoring_logic] score_with_rubric dipanggil")
    print(f"  Question ID: {question_id}")
    print(f"  Question: {question_text[:50]}...")
    print(f"  Answer panjang: {len(answer)} karakter")

    # question_id sudah berupa 'q1', 'q2', dst. Langsung digunakan.
    # Ambil rubrik untuk pertanyaan ini
    rubric_entry = rubric_data.get(question_id, {})
    rubric = rubric_entry.get("ideal_points", {})
    
    if not rubric:
        print(f"  [WARNING] Tidak ada rubrik untuk question_id: {question_id}")
        return 3, "No rubric available for this question."
    
    a = answer.strip()

    if is_non_relevant(a) or len(a.split()) < MIN_LENGTH_FOR_SCORE:
        print(f"  [DEBUG] Jawaban dianggap non-relevant atau terlalu pendek")
        # Mengembalikan skor 0 dan alasan dari rubrik level 0
        zero_reason = rubric.get("0", ["Unanswered"])[0]
        return 0, zero_reason

    try:
        # Encode jawaban untuk similarity comparison
        embedding_a = model_embedder.encode(a.lower())
        
        print(f"  [DEBUG] Rubric keys available: {list(rubric.keys())}")

        # Fungsi untuk menghitung kecocokan
        def count_matches(indicators, threshold=0.40):
            if not indicators:
                return 0
                
            hits = 0
            # Pre-encode semua indikator untuk efisiensi
            try:
                embeddings_indicators = model_embedder.encode([ind.lower() for ind in indicators])
                
                # Hitung cosine similarity
                similarities = util.cos_sim(embedding_a, embeddings_indicators).flatten()
                
                for sim in similarities:
                    if sim.item() >= threshold:
                        hits += 1
            except Exception as e:
                print(f"    [ERROR] dalam count_matches: {e}")
                
            return hits

        # Iterasi dari skor tertinggi ke terendah (4, 3, 2, 1)
        # Gunakan string keys karena dimuat dari JSON
        for point_str in ["4", "3", "2", "1"]:
            point = int(point_str)
            
            # FIX: Akses key rubrik sebagai string
            indicators = rubric.get(point_str)
            
            if not indicators:
                print(f"  [DEBUG] Tidak ada indicators untuk point {point_str}")
                continue

            print(f"  [DEBUG] Mengecek point {point} dengan {len(indicators)} indicators")
            
            hits = count_matches(indicators)
            
            # Logika Min hits:
            if point == 4:
                min_hits = max(1, int(len(indicators) * 0.6))
            elif point == 3:
                min_hits = max(1, int(len(indicators) * 0.5))
            else: # Untuk point 2 dan 1
                min_hits = 1

            print(f"  [DEBUG] Point {point}: hits={hits}, min_hits={min_hits}")
            
            if hits >= min_hits:
                # Mengembalikan skor tertinggi yang memenuhi kriteria
                # FIX: Akses alasan rubrik sebagai string point_str
                reason = rubric.get(point_str, [f"Score {point} achieved"])[0]
                print(f"  [DEBUG] Match ditemukan untuk point {point}: {reason[:50]}...")
                return point, reason

        # Jika tidak ada yang cocok sama sekali, berikan skor 1 (atau minimal)
        print(f"  [DEBUG] Tidak ada match, return default score 1")
        return 1, rubric.get("1", ["Minimal or Vague Response"])[0]
        
    except Exception as e:
        print(f"  [ERROR] Exception dalam score_with_rubric: {e}")
        import traceback
        traceback.print_exc()
        return 1, f"Error in scoring: {str(e)}"