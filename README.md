# VISTA: Triplet-Supervised Video Style Transfer with Diffusion Transformers

**Yiren Song, Wangzi Yao, Haofan Wang, Mike Zheng Shou**

[![Paper](https://img.shields.io/badge/Paper-NeurIPS%202026-blue)](https://arxiv.org/abs/2605.17312) [![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-orange)](https://huggingface.co/your-org/vista-dataset) [![Models](https://img.shields.io/badge/Models-HuggingFace-green)](https://huggingface.co/your-org/vista-models)

---

## Project Structure
```
Wan22Video/
├── src/
│   └── inference.py          # Main inference script
├── models/
│   └── wan2/                 # Modified Wan2 transformer with IPA support
├── datasets/
│   └── custom_dataset.py     # Data loading utilities
├── configs/
│   └── inference_example.yaml
├── example_data/
│   ├── input_videos/         # 3 example content videos (768x768, 81 frames)
│   ├── reference_images/     # 3 example style reference images (768x768)
│   └── output_videos/        # Example output videos
├── inference.sh              # Example inference command
├── requirements.txt
└── README.md
```
## Setup
### 1. Create conda environment
```bash
conda create -n wan22video python=3.9 -y
conda activate wan22video
```
### 2. Install PyTorch (CUDA 11.8)
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
> **Note:** `decord` may require `ffmpeg` system libraries. Install via `conda install -c conda-forge ffmpeg` if needed.
## Model Weights
Download the following:
1. **Base model**: [Wan-AI/Wan2.1-T2V-14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers) (or the 5B variant)
2. **Fine-tuned checkpoint**: `20000.ckpt` (contains LoRA weights + IPA adapter)
3. **SigLIP encoder**: [google/siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384)
## Inference
### Prepare input data
Organize your data as follows:
```
data/
├── input_videos/       # Content videos (.mp4)
├── reference_images/   # Style reference images (.png/.jpg), one per video
└── captions/           # (Optional) Text prompts (.txt), matching video filenames
```
- Videos and reference images are matched by natural sort order.
- Caption files should share the same stem name as the corresponding video (e.g., `video_001.mp4` -> `video_001.txt`).
### Edit config
Update `configs/inference_example.yaml` with your paths:
```yaml
model_id: "/path/to/Wan2.2-TI2V-5B-Diffusers"
output_root: "./output"
dataset:
  video_root:  "/path/to/input_videos"
  first_root:  "/path/to/reference_images"
  caption_root: "/path/to/captions"
  is_one2three: True
  height: 768
  width: 768
  sample_n_frames: 81
```
### Run inference
```bash
export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=0 python src/inference.py \
  --config configs/inference_example.yaml \
  --ckpt_path /path/to/20000.ckpt \
  --siglip-model /path/to/siglip-so400m-patch14-384 \
  --ipa-checkpoint /path/to/20000.ckpt
```
Or use the provided script:
```bash
bash inference.sh
```
### Arguments
| Argument | Description | Default |
|---|---|---|
| `--config` | Path to YAML config file | (required) |
| `--ckpt_path` | Path to fine-tuned checkpoint (.ckpt) | "" |
| `--siglip-model` | Path to SigLIP vision encoder | "" |
| `--ipa-checkpoint` | Path to IPA adapter weights | "" |
| `--start-index` | Skip first N samples | 0 |
### Output
Generated videos are saved to `{output_root}/{experiment_name}/infer_samples/`.
## Hardware Requirements
- GPU: NVIDIA A100 (80GB) recommended
- The 5B model requires ~40GB VRAM for inference at 768x768 resolution with 81 frames.
## Acknowledgments
This project is built upon:
- [Wan 2.1/2.2](https://github.com/Wan-Video/Wan2.1) by Wan-AI
- [Diffusers](https://github.com/huggingface/diffusers) by Hugging Face
- [SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384) by Google
