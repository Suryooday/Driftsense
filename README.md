# DriftSense — Wafer Pattern Matching & Stage Navigation Drift Recovery

DriftSense is an automated semiconductor wafer inspection system built for the SEMICON India Hackathon. It localizes micro-scale chip reference patterns inside wide-field wafer inspection images and calculates sensorless stage coordinate corrections ($\text{MOVE X}$, $\text{MOVE Y}$) to recover intended inspection sites when navigation drift occurs.

---

## ⚡ Reviewer Quickstart (Evaluate in 30 Seconds)

To test the system on custom or synthetic datasets immediately, follow these simple steps:

### 1. Set Up Python Environment
Ensure you have Python 3.10+ installed. Activate the virtual environment and install dependencies:
```bash
# Activate virtual environment
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 2. Generate Synthetic Image Pairs (DRAM or FinFET)
Use the standalone procedurally controlled dataset generator:
```bash
# Generate 5 DRAM pattern pairs
python3 generate_dataset.py --style DRAM --num_pairs 5 --output_dir data/generated_dram

# Generate 5 FinFET pattern pairs
python3 generate_dataset.py --style FinFET --num_pairs 5 --output_dir data/generated_finfet
```
This generates reference patterns, drifted search fields, and writes the exact ground truth coordinates to `data/generated_dram/ground_truth.csv` and `data/generated_finfet/ground_truth.csv`.

### 3. Run Pattern Localization Inference
Execute the localization runner on any reference and search image pair. It runs in **~0.3 seconds** with sub-pixel and sub-degree accuracy:
```bash
# Run on a generated DRAM sample pair (using positional arguments)
python3 localize.py data/generated_dram/sample_000/reference_image.png data/generated_dram/sample_000/search_image.png
```
**Output Example:**
```
(x, y) = (257.6521, 255.4312)
{
  "predicted_center": {
    "x": 257.6521,
    "y": 255.4312
  },
  "rotation_degrees": -1.0421,
  "scale": 4.9812,
  "confidence_score": 0.8942,
  "inference_time_seconds": 0.3541
}
```

---

## 1. Project Overview (What is DriftSense?)

### The Semiconductor Wafer Problem
During semiconductor fabrication and defect inspection, automated tools move a high-precision stage to inspect specific chip coordinates across a wafer. Due to mechanical vibrations, thermal expansion, and mechanical backlash, the stage often drifts slightly off target.

Because semiconductor wafers feature repeating grid-like Manhattan patterns, even a minor drift of a few pixels can cause the tool to inspect the wrong die or miss critical defects.

### The DriftSense Solution
DriftSense solves this problem without requiring extra physical hardware sensors:
1. **High-Resolution Reference Pattern** ($100\times$ magnification, $256 \times 256$ px): The target pattern structure to be localized.
2. **Search Area** ($10\times$ magnification FOV, $512 \times 512$ px): The wider inspection image captured by the wafer camera.
3. **Pattern Localization**: Driftsense localizes the pattern center $(x_{detected}, y_{detected})$, rotation, and scale with sub-pixel precision.
4. **Drift Calculation**: Compares actual detected coordinates against expected inspection coordinates $(x_{expected}, y_{expected})$.
5. **Stage Coordinate Recovery**: Computes the exact correction vector ($\Delta x, \Delta y$) needed to move the stage back to the center of the target site.

---

## 2. System Architecture

```
                       ┌─────────────────────────┐
                       │    Reference Pattern    │
                       │     (256 x 256 px)      │
                       └────────────┬────────────┘
                                    │
                                    v
                       ┌─────────────────────────┐
                       │       Search Area       │
                       │     (512 x 512 px)      │
                       └────────────┬────────────┘
                                    │
                                    v
                       ┌─────────────────────────┐
                       │  Classical NCC Matcher  │
                       │ + Candidate Generator   │
                       └────────────┬────────────┘
                                    │
                                    v
                       ┌─────────────────────────┐
                       │ High-Resolution Pose    │
                       │ Refinement (Rot & Scale)│
                       └────────────┬────────────┘
                                    │
                                    v
                       ┌─────────────────────────┐
                       │  Target Localization    │
                       │  (X, Y, Rotation, Scale)│
                       └────────────┬────────────┘
                                    │
                                    v
                       ┌─────────────────────────┐
                       │  Drift & Stage Recovery │
                       │   (MOVE X, MOVE Y)      │
                       └────────────┬────────────┘
                                    │
                   ┌────────────────┴────────────────┐
                   │                                 │
                   v                                 v
        ┌────────────────────┐            ┌────────────────────┐
        │   FastAPI Backend  │            │  Next.js Frontend  │
        │   (localhost:8000) │<──────────>│  (localhost:3000)  │
        └────────────────────┘            └────────────────────┘
```

---

## 3. Directory & File Breakdown

Below is a detailed list of all project folders and files, explaining what each contains:

```
.
├── backend/                        # FastAPI Backend Application Layer
│   ├── main.py                     # Main FastAPI server entry point & API routes
│   ├── schemas.py                  # Pydantic data schemas for request/response validation
│   ├── requirements.txt            # Python dependencies for backend API
│   └── services/
│       └── driftsense_service.py   # Service wrapping frozen pattern matcher & drift math
│
├── frontend/                       # Next.js 15 Web Application (User Interface)
│   ├── package.json                # Frontend Node.js dependencies & scripts
│   ├── tailwind.config.ts          # Tailwind CSS styling configuration
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx            # Main single-page web dashboard
│   │   │   ├── layout.tsx          # Root HTML layout with custom typography
│   │   │   └── globals.css         # Global CSS styles & custom animations
│   │   ├── components/
│   │   │   ├── header.tsx          # Header bar with system status & "Run Demo" button
│   │   │   ├── hero-section.tsx    # Title banner & horizontal pipeline flowchart
│   │   │   ├── reference-viewer.tsx# 256x256 reference image viewer with 0-256px ruler scales
│   │   │   ├── search-viewer.tsx   # 512x512 search area viewer with 0-500px ruler scales & crosshairs
│   │   │   ├── analysis-controls.tsx # Upload buttons & expected X/Y coordinate input controls
│   │   │   ├── alignment-results.tsx # 5-metric display card (Position, Drift, Rotation, Scale, NCC)
│   │   │   ├── drift-status.tsx    # Status badge (ALIGNED, MINOR DRIFT, SIGNIFICANT DRIFT)
│   │   │   ├── stage-recovery.tsx  # Animated stage vector graph showing CURRENT vs TARGET position
│   │   │   └── process-flow.tsx    # 4-step technical explanation of the pipeline
│   │   ├── lib/
│   │   │   └── api.ts              # HTTP API client for calling FastAPI backend
│   │   └── types/
│   │       └── analysis.ts         # TypeScript interfaces matching API schemas
│
├── src/                            # Core Python Algorithmic Pipeline & CLI Utilities
│   ├── final_system.py             # Main production entry point for pattern matching
│   ├── drift_recovery.py           # Stage drift recovery & classification logic
│   ├── driftsense.py               # CLI tool for running end-to-end drift detection
│   ├── driftsense_demo.py          # Terminal demonstration runner
│   ├── run_final.py                # Command-line pattern localization runner
│   ├── verify_final_system.py      # Reproducibility verification test suite
│   ├── matching/
│   │   └── classical_matcher.py    # Sub-pixel classical NCC template matcher
│   ├── hybrid/
│   │   ├── candidate_generator.py  # Multi-scale & rotation candidate generator
│   │   └── patch_extractor.py      # Patch extraction utility
│   ├── audit/
│   │   ├── validate_drift_recovery.py # Math validation suite for stage recovery
│   │   └── verify_drift_audit.py   # Final audit log & hash integrity checker
│   └── visualization/
│       └── visualize_drift.py      # Generates synthetic drift plots
│
├── configs/
│   └── final_system_config.json    # Immutable production algorithm parameters & thresholds
│
├── data/                           # Wafer Inspection Image Benchmark & Datasets
│   ├── sample_000/ to sample_039/  # Frozen 40-sample benchmark dataset
│   └── robustness_samples/         # 200 independent evaluation samples
│
├── models/                         # Experimental Deep Learning Model Checkpoints (DL V1 & V2)
│
├── reports/                        # Documentation, Audit Logs & Plots
│   ├── DRIFT_RECOVERY_TECHNICAL_NOTE.md # Complete mathematical technical note
│   ├── FINAL_TECHNICAL_REPORT.md   # System engineering report
│   ├── PROJECT_PRESENTATION.md     # Hackathon presentation slides
│   └── drift_recovery/             # Validation JSONs, audit markdown logs, & figures
│
├── generate_dataset.py             # Standalone DRAM/FinFET synthetic wafer pattern generator
├── localize.py                     # Standalone pattern localization inference script
├── CITATIONS.md                    # Literature references for SEM noise and matching algorithms
├── requirements.txt                # Main Python environment dependencies
└── README.md                       # Main project documentation (this file)
```

---

## 4. Backend & Frontend Architecture

The web application is split into two clean layers:

### Backend Layer (`backend/`)
- **Technology**: FastAPI (Python 3.10+) running on **`http://localhost:8000`**
- **Role**: Wraps the frozen Python matching engine (`src/final_system.py`) and drift recovery module (`src/drift_recovery.py`) into typed REST endpoints.
- **Available Endpoints**:
  - `GET /api/health`: Health status check (`{ "status": "online", "system": "DriftSense" }`).
  - `GET /api/demo`: Runs analysis on benchmark `sample_010` and returns structured JSON along with base64 encoded images.
  - `POST /api/analyze`: Accepts multipart uploads (`reference` image, `search` image, `expected_x`, `expected_y`) and returns matching & drift results.

### Frontend Layer (`frontend/`)
- **Technology**: Next.js 15, TypeScript, Tailwind CSS, shadcn/ui, Lucide icons running on **`http://localhost:3000`**
- **Role**: Provides a semiconductor inspection UI for judges to interact with the system visually.
- **Key Features**:
  - **Pixel Coordinate Scale Rulers**: Displays $0\text{--}256$ px scales on the Reference Pattern and $0\text{--}500$ px scales on the Search Area with active detected coordinate markers (`X: 254`, `Y: 411`).
  - **Scanning & Target Overlays**: Animated scanning line during analysis and precise target crosshairs upon detection.
  - **Stage Correction Vector Graph**: Visualizes `CURRENT` position vs `TARGET` position with an animated arrow and exact `MOVE X` / `MOVE Y` values.
  - **Live Demo Mode**: Single-click interactive demonstration.

### How They Connect
The frontend sends HTTP requests to `http://localhost:8000` via `frontend/src/lib/api.ts`. Cross-Origin Resource Sharing (CORS) is enabled on the FastAPI backend to allow seamless local communication.

---

## 5. Step-by-Step Setup & Quickstart

### Prerequisites
- **Python**: 3.10 or higher
- **Node.js**: 18.0 or higher (with `npm`)

### Step 1: Clone Repository & Setup Python Environment
```bash
# Clone repository
git clone https://github.com/Suryooday/Driftsense.git
cd Driftsense

# Create virtual environment
python3 -m venv venv

# Activate virtual environment (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

### Step 2: Setup Frontend Dependencies
```bash
# Navigate to frontend folder and install Node packages
cd frontend
npm install
cd ..
```

---

## 6. How to Run the Web Application

To run the complete web application, start both the backend server and frontend web interface in separate terminal windows:

### Terminal 1: Backend Server (FastAPI)
```bash
# From project root directory
source venv/bin/activate
PYTHONPATH=. ./venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
*The backend API will start at `http://localhost:8000`.*

### Terminal 2: Frontend Web Interface (Next.js)
```bash
# From project root directory
cd frontend
npm run dev -- -p 3000
```
*The web dashboard will start at `http://localhost:3000`.*

Open **`http://localhost:3000`** in your browser to view and interact with DriftSense!

---

## 7. How to Use the Web Application (User Guide)

### Using Demo Mode (Recommended)
1. Click the **Run Demo** button in the top-right header bar.
2. The web app automatically loads benchmark `sample_010` data.
3. The search area animates a scanning line during localization.
4. Upon completion, the UI displays:
   - Target crosshair overlay with pixel coordinate rulers.
   - 5-metric alignment results (Position, Drift, Rotation, Scale, NCC Confidence).
   - Dynamic status badge (`MINOR DRIFT`).
   - Stage correction vector graph drawing the arrow from `CURRENT` to `TARGET` position, showing `MOVE X` (`+3.76 px`) and `MOVE Y` (`-1.60 px`).

### Manual Image Analysis Mode
1. Under **Reference Image**, click **Upload** and select a 256×256 px reference image.
2. Under **Search Image**, click **Upload** and select a 512×512 px search area image.
3. Enter your **Expected Target Coordinates** (X and Y).
4. Click **ANALYZE ALIGNMENT**.
5. View the detected coordinates, navigation drift, and recommended stage movements.

---

## 8. How to Run Model Inference & Evaluation (Judges Evaluation)

Judges can test the model on any custom test dataset CSV file using the universal `predict.py` script:

### Batch CSV Evaluation Mode (Judges Test Dataset)
```bash
python3 predict.py --input test_dataset.csv --output predictions.csv
```
*Processes all image pairs in the CSV (`search_image_path, reference_image_path`), outputs localized sub-pixel coordinates $(x, y)$, rotation, and scale to `predictions.csv`, and displays the 1px–5px Confusion Matrix accuracy table.*

### Single Image Pair Mode (Evaluation via `localize.py`)
Run the standalone localization inference script with either positional or keyword arguments:
```bash
# Positional Arguments
python3 localize.py data/sample_000/reference_image.png data/sample_000/search_image.png

# Keyword Arguments (with optional output path to save JSON results)
python3 localize.py --reference data/sample_000/reference_image.png --search data/sample_000/search_image.png --output results/prediction.json
```

### 5. Run Official Hackathon CSV Scoring Utility
```bash
python3 -m src.scoring.eval_dataset_csv \
    --csv data/benchmark_ground_truth.csv \
    --output-csv results/eval_results.csv
```
*Evaluates the pattern matcher against any ground truth CSV (`search_image_path, reference_image_path, GTx, GTy`) and outputs the 1px–5px Confusion Matrix and accuracy statistics.*

---

## 9. Performance & Validation Results

The frozen system **"Classical NCC-Based Wafer Pattern Matching with High-Resolution Pose Refinement"** has been rigorously validated across multiple benchmark datasets:

| Evaluation Dataset | Sample Count | Success Rate (%) | Median Loc Error (px) | Mean Rot Error (°) | Mean Scale Error | Avg Time (s) |
|---|---|---|---|---|---|---|
| **Frozen Benchmark** | 40 | **97.5% (39/40)** | 0.5280 px | 0.0910° | 0.00367 | 0.3666 s |
| **Independent Robustness Set** | 200 | **97.5% (195/200)** | 0.5648 px | 0.0847° | 0.00489 | 0.3666 s |

---

## 10. Navigation Drift Mathematics

Navigation drift and stage correction vectors are calculated as follows:
- **X-axis Drift**: $\Delta x = x_{expected} - x_{detected}$
- **Y-axis Drift**: $\Delta y = y_{expected} - y_{detected}$
- **Drift Magnitude**: $D = \sqrt{\Delta x^2 + \Delta y^2}$

### Status Classification:
- **`ALIGNED`**: $D \le 1.0$ px
- **`MINOR_DRIFT`**: $1.0 < D \le 5.0$ px
- **`SIGNIFICANT_DRIFT`**: $D > 5.0$ px

### Stage Correction Vector:
To realign the wafer inspection tool over the intended target:
$$\text{MOVE X} = \Delta x$$
$$\text{MOVE Y} = \Delta y$$

---

## 11. Stage Control Disclaimer

The current implementation provides pixel-space coordinate recovery. Conversion to physical stage displacement units such as microns requires a calibrated pixel-to-stage transformation matrix supplied by the inspection tool's hardware interface. The software does not directly actuate physical motors.
