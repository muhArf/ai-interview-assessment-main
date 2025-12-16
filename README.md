#  SEI.AI (Sistem Evaluasi Interview berbasi Artficial Intelligence)

## Deskripsi
Proyek ini diberi nama SEI.AI (Sistem Evaluasi Interview berbasis Artificial Intelligence) yang bertujuan untuk mengotomatisasi proses interview dan evaluasi kandidat. Sistem ini menganalisis non-verbal secara terintegrasi untuk menghasilkan penilaian yang objektif, konsisten, dan efisien. Inovasi utama terletak pada evaluasi wawancara yang mampu mengurangi bias subjektif dan meningkatkan kualitas pengambilan keputusan rekrutmen tanpa adanya penilain objektif.

## Petunjuk Setup Environment
Untuk menjalankan aplikasi ini, pastikan environment telah memenuhi persyaratan berikut:

### 1. Prasyarat
- Python versi 3.9 atau lebih baru
- pip (Python package manager)
- Virtual environment (opsional)

### 2. Clone Repository
```bash
git clone https://github.com/muhArf/ai-interview-assessment-main
cd ai-interview-assessment-main 

## Membuat dan Mengaktifkan Virtual Environment (Opsional)
python -m venv venv

## Install Dependencies
pip install -r requirements.txt

##Tautan Model Machine Learning
Model Machine Learning yang digunakan dalam project ini dapat diunduh melalui tautan berikut:
https://drive.google.com/drive/folders/1i34s5pOT43a0q_6Yt8eq054fyMCFKAyq?usp=drive_link 

## Cara Menjalankan Aplikasi
- Pastikan seluruh dependencies telah terinstal
- Pastikan file model Machine Learning telah diletakkan pada folder yang sesuai
- Jalankan aplikasi menggunakan perintah berikut: 
streamlit run app.py
- Aplikasi akan berjalan pada browser secara otomatis
- Upload file interview kandidat dan sistem akan menampilkan hasil evaluasi
