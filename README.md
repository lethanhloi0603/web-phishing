# GIPRL: phát hiện phishing do LLM tạo

Framework nghiên cứu phòng thủ cho đề tài:

> **Generator-Invariant Representation Learning for Detecting LLM-Generated Phishing Websites under Distribution Shift**

Mục tiêu là phát hiện `benign`/`phishing` khi website kiểm thử đến từ generator, model family, brand hoặc template chưa xuất hiện lúc huấn luyện. Generator chỉ là metadata hỗ trợ huấn luyện và đánh giá; đầu ra cuối cùng luôn là `0=benign`, `1=phishing`.

Repository chỉ đọc source HTML lưu trữ. Không phục vụ website, không chạy JavaScript và không gửi form trong mẫu phishing.

## Framework hiện tại

```text
HTML text ──> MobileBERT ─────────┐
                                  │
DOM tags ──> DOM Transformer ─────┼──> Fusion ──> benign/phishing
                                  │        │
Static URL/HTML features ──> MLP ─┘        ├──> contrastive projection
                                           └──> generator discriminator + GRL

Loss = BCE + SupCon + generator-adversarial + CORAL domain alignment
```

Các cơ chế chính:

- Chống exact-duplicate leakage bằng SHA-256 của HTML chuẩn hóa.
- Group split theo `template_id` giữa train và validation.
- Tùy chọn cô lập template nghiêm ngặt giữa train/validation và test.
- Cross-generator supervised contrastive learning.
- Gradient Reversal Layer để giảm thông tin generator trong embedding.
- CORAL để căn chỉnh conventional phishing và LLM phishing.
- Artifact tests: xóa comment, chuẩn hóa class/ID và xóa visible text.
- Hai baseline nhẹ: TF-IDF + Logistic Regression và static features + HistGradientBoosting.

DOM encoder hiện là **Transformer trên chuỗi tag preorder**, chưa phải DOM Graph Transformer.

## 1. Cài môi trường

Windows PowerShell:

```powershell
cd C:\Users\admin\Downloads\phishgen_detector
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu PowerShell chặn activation:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.venv\Scripts\Activate.ps1
```

Kiểm tra code mà không tải MobileBERT:

```powershell
python scripts\check_project.py
```

## 2. Chạy thử hoàn toàn offline

### Bước 1: tạo dữ liệu demo

```powershell
python scripts\make_demo_data.py
```

Ba file được tạo trong `data\demo`.

### Bước 2: tạo kịch bản Mixed chống leakage

```powershell
python scripts\build_scenarios.py `
  --phresh-train data\demo\phresh_train.csv `
  --phresh-test data\demo\phresh_test.csv `
  --llm data\demo\llm_phish.csv `
  --scenario mixed `
  --out-dir data\demo\scenario_mixed
```

Kiểm tra `data\demo\scenario_mixed\manifest.json` để xem số mẫu, generator và leakage report.

### Bước 3: chạy baseline

```powershell
python scripts\train_baseline.py `
  --train data\demo\scenario_mixed\train.csv `
  --val data\demo\scenario_mixed\val.csv `
  --test data\demo\scenario_mixed\test.csv `
  --model tfidf
```

Baseline static nhẹ:

```powershell
python scripts\train_baseline.py `
  --train data\demo\scenario_mixed\train.csv `
  --test data\demo\scenario_mixed\test.csv `
  --model static
```

### Bước 4: smoke-test deep model không tải MobileBERT

Lệnh này chỉ dùng DOM + static branch, phù hợp để xác minh pipeline:

```powershell
python scripts\train_deep.py `
  --train data\demo\scenario_mixed\train.csv `
  --val data\demo\scenario_mixed\val.csv `
  --disable-html `
  --epochs 2 `
  --batch-size 8 `
  --output outputs\demo_light.pt
```

Đánh giá và chạy artifact tests:

```powershell
python scripts\evaluate_deep.py `
  --checkpoint outputs\demo_light.pt `
  --val data\demo\scenario_mixed\val.csv `
  --test data\demo\scenario_mixed\test.csv `
  --target-fpr 0.01 `
  --artifact-tests `
  --output-json outputs\demo_light_metrics.json `
  --predictions outputs\demo_light_predictions.csv
```

## 3. Chuẩn bị dữ liệu thật

### PhreshPhish subset

Không tải toàn bộ dataset. Với máy còn khoảng 19 GB, bắt đầu bằng 2.000 mẫu mỗi lớp cho train và 1.000 mẫu mỗi lớp cho test:

```powershell
python scripts\download_phreshphish.py `
  --split train --n-per-class 2000 `
  --seed 42 `
  --shuffle-buffer 10000 `
  --out data\phresh_train.csv

python scripts\download_phreshphish.py `
  --split test --n-per-class 1000 `
  --seed 42 `
  --shuffle-buffer 10000 `
  --out data\phresh_test.csv
```

### Dataset website do LLM coding agent tạo

Clone shallow để giảm dung lượng, không chạy các website trong repository:

```powershell
git clone --depth 1 `
  https://github.com/TailiaReganMalloy/assessingSpearPhishing.git `
  external\assessingSpearPhishing
```

Quét source HTML:

```powershell
python scripts\prepare_llm_repo.py `
  --repo external\assessingSpearPhishing `
  --out data\llm_phish.csv
```

Script tạo thêm `data\llm_phish_audit.csv`. Cần kiểm tra thủ công các cột:

- `manual_validated`: đặt `1` khi mẫu phù hợp nghiên cứu.
- `generator` và `generator_family`.
- `brand`, `template_id`, `quality_level`.

Không xem một codebase được sinh ra là phishing hợp lệ chỉ vì nó nằm trong dataset.

### Triage nhanh trước khi audit

Chấm điểm và sắp xếp các mẫu theo mức `high`, `medium`, `low`:

```powershell
python scripts\triage_llm_audit.py `
  --llm data\llm_phish.csv `
  --audit data\llm_phish_audit.csv `
  --automated-screening `
  --out data\llm_phish_audit_ranked.csv
```

Mở `data\llm_phish_audit_ranked.csv`. Bộ luật xác định ghi riêng
`validation_status`, `validated_by`, `review_method` và lý do; nó không thay đổi
`manual_validated`. Cột `ai_validated` được giữ để tương thích với dữ liệu pilot
cũ, không được dùng thay cho đánh giá thủ công trong kết quả chính.

Review các mẫu được sàng lọc bằng giao diện cục bộ an toàn:

```powershell
python scripts\review_llm_audit.py
```

Trình duyệt sẽ mở `http://127.0.0.1:8765`. Bản xem trước loại script, chặn form và tài nguyên mạng. Ba nút `Phishing`, `Không phishing`, `Chưa chắc` lưu ngay vào CSV; script tạo một file backup trước lần ghi đầu tiên. Nhấn `Ctrl+C` trong terminal sau khi review xong.

## 4. Tạo các scenario nghiên cứu

### Human → LLM

```powershell
python scripts\build_scenarios.py `
  --phresh-train data\phresh_train.csv `
  --phresh-test data\phresh_test.csv `
  --llm data\llm_phish.csv `
  --llm-audit data\llm_phish_audit.csv `
  --require-validated `
  --scenario human_to_llm `
  --out-dir data\scenario_human_to_llm
```

Scenario này đo distribution shift nhưng training không có LLM sample, nên adversarial và CORAL loss tự động bằng 0.

### Mixed

```powershell
python scripts\build_scenarios.py `
  --phresh-train data\phresh_train.csv `
  --phresh-test data\phresh_test.csv `
  --llm data\llm_phish.csv `
  --llm-audit data\llm_phish_audit.csv `
  --require-validated `
  --scenario mixed `
  --out-dir data\scenario_mixed
```

### Leave-One-Generator-Out

```powershell
python scripts\build_scenarios.py `
  --phresh-train data\phresh_train.csv `
  --phresh-test data\phresh_test.csv `
  --llm data\llm_phish.csv `
  --llm-audit data\llm_phish_audit.csv `
  --require-validated `
  --scenario logo `
  --holdout-generator gpt-4o `
  --out-dir data\scenario_logo_gpt4o
```

### Leave-One-Family-Out

```powershell
python scripts\build_scenarios.py `
  --phresh-train data\phresh_train.csv `
  --phresh-test data\phresh_test.csv `
  --llm data\llm_phish.csv `
  --llm-audit data\llm_phish_audit.csv `
  --require-validated `
  --scenario lofo `
  --holdout-family gemini `
  --out-dir data\scenario_lofo_gemini
```

Thêm `--strict-template-isolation` khi metadata `template_id` đã được audit tốt. Với template tự suy ra từ DOM fingerprint, tùy chọn này có thể loại quá nhiều trang có cấu trúc đơn giản giống nhau.

Với ổ đĩa hạn chế, thêm `--compression gzip` khi build scenario. Các script train/evaluate đọc trực tiếp `train.csv.gz`, `val.csv.gz`, `test.csv.gz` mà không cần giải nén thủ công.

## 5. Train model đầy đủ

Lần đầu chạy, Transformers sẽ tải MobileBERT. Repo chỉ giữ checkpoint tốt nhất:

```powershell
python scripts\train_deep.py `
  --train data\scenario_lofo_gemini\train.csv `
  --val data\scenario_lofo_gemini\val.csv `
  --epochs 5 `
  --batch-size 8 `
  --output outputs\giprl_lofo_gemini.pt
```

Nếu thiếu VRAM/RAM:

```powershell
python scripts\train_deep.py `
  --train data\scenario_lofo_gemini\train.csv `
  --val data\scenario_lofo_gemini\val.csv `
  --epochs 3 `
  --batch-size 4 `
  --max-html-length 128 `
  --max-dom-length 128 `
  --freeze-html `
  --output outputs\giprl_small.pt
```

Đánh giá:

```powershell
python scripts\evaluate_deep.py `
  --checkpoint outputs\giprl_lofo_gemini.pt `
  --val data\scenario_lofo_gemini\val.csv `
  --test data\scenario_lofo_gemini\test.csv `
  --target-fpr 0.01 `
  --artifact-tests `
  --output-json outputs\giprl_lofo_gemini_metrics.json `
  --predictions outputs\giprl_lofo_gemini_predictions.csv
```

## 6. Ablation

Chạy các ablation chính với năm seed bằng DOM + static (không tải MobileBERT):

```powershell
python scripts\run_experiment_matrix.py `
  --scenario-dir data\scenario_lofo_qwen_page_matched_calibrated `
  --variants bce,supcon,full,artifact_consistency,robust_full `
  --seeds 42,43,44,45,46 `
  --epochs 5 `
  --batch-size 8 `
  --artifact-tests `
  --min-free-gb 5
```

Runner tự động:

- resume các checkpoint/metrics đã có;
- calibration bằng validation tại FPR mục tiêu 1%;
- chỉ chạy artifact tests cho các biến thể khai báo trong `--artifact-test-variants`;
- ghi predictions không kèm cột HTML;
- dừng trước khi train nếu ổ đĩa còn dưới 5 GB;
- tạo `summary.csv` và `summary.json` chứa mean, standard deviation và hierarchical bootstrap CI 95%.

Mặc định runner không dùng HTML branch. Chỉ thêm `--with-html` khi chủ động muốn tải/chạy MobileBERT. Có thể dùng `--dry-run` để xem toàn bộ lệnh trước khi chạy.

Không contrastive learning:

```powershell
python scripts\train_deep.py ... --lambda-supcon 0
```

Không generator-adversarial learning:

```powershell
python scripts\train_deep.py ... --lambda-adversarial 0
```

Không domain alignment:

```powershell
python scripts\train_deep.py ... --lambda-alignment 0
```

HTML-only, DOM-only hoặc static-only:

```powershell
# HTML-only
python scripts\train_deep.py ... --disable-dom --disable-static

# DOM-only
python scripts\train_deep.py ... --disable-html --disable-static

# Static-only
python scripts\train_deep.py ... --disable-html --disable-dom
```

## 7. Chỉ số báo cáo

Framework báo cáo Precision, Recall, F1, AUPRC, AUROC, FPR, FNR, MCC, Brier score và Expected Calibration Error. Thí nghiệm chính nên chạy ít nhất 5 seed và báo cáo `mean ± standard deviation`; repo hiện chạy một seed mỗi lệnh để tránh lưu nhiều checkpoint ngoài ý muốn.

## 8. Lưu ý dung lượng

- Không tăng subset cho đến khi demo và baseline chạy ổn.
- Không lưu checkpoint từng epoch; script chỉ giữ model tốt nhất và file history JSON nhỏ.
- Không thêm screenshot/runtime data trong giai đoạn đầu.
- `data`, Hugging Face cache, `.venv` và checkpoint là các phần chiếm dung lượng chính.
- Không xóa cache hoặc dữ liệu bằng script này; hãy kiểm tra và sao lưu trước khi dọn thủ công.

## Cấu trúc

```text
phishgen_detector/
├── config.yaml
├── requirements.txt
├── README.md
├── data/
├── outputs/
├── scripts/
│   ├── make_demo_data.py
│   ├── download_phreshphish.py
│   ├── prepare_llm_repo.py
│   ├── triage_llm_audit.py
│   ├── build_scenarios.py
│   ├── train_baseline.py
│   ├── train_deep.py
│   ├── evaluate_deep.py
│   ├── run_experiment_matrix.py
│   ├── summarize_experiments.py
│   └── check_project.py
└── src/
    ├── preprocessing.py
    ├── data.py
    ├── metrics.py
    ├── calibration.py
    ├── losses.py
    └── model.py
```

## 9. Publication gate và tái lập

Trước khi dùng một scenario trong bảng kết quả chính, chạy:

```powershell
python scripts\audit_dataset.py `
  --scenario-dir data\scenario_lofo_qwen_strict `
  --min-test-positives 100 `
  --require-template-isolation
```

Gate thất bại (exit code `2`) nếu còn exact/template leakage hoặc test có quá ít mẫu phishing. Mỗi scenario mới lưu SHA-256 của các split. Experiment runner lưu thêm hash dữ liệu, Git state, phiên bản Python/PyTorch/scikit-learn, CUDA và GPU.

Quy trình đầy đủ cho bảng khóa luận/Q1, paired statistical comparison và error analysis nằm trong `RESEARCH_CHECKLIST.md`. Chạy test hồi quy bằng:

```powershell
python -m unittest discover -s tests -v
```

Thiết kế chính mới dùng multi-family LOFO trên Claude, Gemini, GPT, Grok,
Mistral và Qwen. Quy trình tạo fold, chạy pilot, chạy năm seed, tổng hợp
macro/micro/worst-family và external validation nằm trong
`MULTIFAMILY_LOFO.md`.

## 10. Dữ liệu từ các nghiên cứu trước

Pipeline hỗ trợ ba domain tách biệt: `real_world`, `llm_generated` và `adversarial_external`. Nhãn do paper công bố được giữ bằng `published_label`/`label_provenance`; code không giả chúng thành kiểm duyệt thủ công.

Dataset LLM có đồng thời `page_label` và `site_label`. Mô hình page-level chỉ xem credential-entry page là positive; báo cáo site-level gom các trang theo `codebase_id` bằng xác suất lớn nhất. Vì vậy trang 404/dashboard không còn bị gán nhầm thành phishing page.

Dataset LLM gốc trong `external\assessingSpearPhishing` có thể được chuẩn hóa thành dữ liệu page-level và codebase-level. Các dataset PhishOracle, Multimodal LLM Phishing Detection và Zenodo có thể nhập bằng adapter CSV/thư mục rồi dùng trong scenario `external`.

Xem nguồn, vai trò và lệnh đầy đủ tại `DATASET_SOURCES.md`.
