# ElOne

# BDC2026 V1 - Local Training Pipeline

Versi ini dibuat untuk workflow lokal VS Code/PowerShell.
Fitur utama:

- ConvNeXt Base/Tiny via `timm`
- AMP mixed precision
- Class weights berdasarkan distribusi train
- Stratified train/validation split
- Macro F1 validation
- Cosine scheduler
- Early stopping
- Checkpoint `best.pth` dan `last.pth`
- Resume otomatis
- Time limit / max epoch per run agar training tidak kebablasan
- TensorBoard log
- TTA horizontal flip saat inference
- Auto generate submission CSV

## 1. Masuk folder project

```powershell
cd C:\Users\Republic Of Gamers\Downloads\BDC2026_v1
```

## 2. Aktifkan virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Kalau PowerShell menolak activate:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 3. Install PyTorch CUDA

Cek driver:

```powershell
nvidia-smi
```

Install PyTorch CUDA:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Install library lain:

```powershell
pip install -r requirements.txt
```

Cek GPU:

```powershell
python src/check_gpu.py
```

Kalo belum:
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

Harus muncul `CUDA available: True`.

## 4. Edit config

Buka:

```text
configs/convnext_base.yaml
```

Pastikan:

```yaml
data_dir: "D:/Dataset BDC26"
```

Struktur dataset harus:

```text
D:/Dataset BDC26/
├── train/
│   ├── 0_Recyclable/
│   ├── 1_Electronic/
│   └── 2_Organic/
├── test/
└── submission.csv
```

## 5. Run EDA

```powershell
python src/eda.py --config configs/convnext_base.yaml
```

Output ada di:

```text
results/
plots/
```

## 6. Training

```powershell
python src/train.py --config configs/convnext_base.yaml
```

Default aman:

```yaml
max_epochs_per_run: 2
time_limit_hours: 2
resume: true
```

Jadi training akan berhenti aman setelah 2 epoch atau 2 jam. Jalankan command yang sama untuk lanjut dari `last.pth`.

## 7. Kalau CUDA out of memory

Edit config:

```yaml
batch_size: 1
```

Kalau masih OOM, ganti:

```yaml
model_name: "convnext_tiny"
batch_size: 8
```

## 8. TensorBoard

```powershell
tensorboard --logdir logs
```

Buka URL yang muncul, biasanya:

```text
http://localhost:6006
```

## 9. Inference / buat submission

Setelah `weights/<run_name>/best.pth` ada:

```powershell
python src/inference.py --config configs/convnext_base.yaml
```

Output:

```text
submissions/submission_convnext_base_v1.csv
```

## 10. Catatan penting aturan BDC

Pipeline ini hanya memakai data train untuk training/validasi/EDA. Data test hanya digunakan saat inference untuk menghasilkan submission, sehingga sesuai aturan kompetisi.
