# ACE-Step Auto-Download

Clean, refactored version of [ACE-Step 1.5](https://github.com/AudioCraft-Evolution-Step/ACE-Step-1.5) with **automatic model downloading** and a HeartMuLa-style API.

**Target**: NVIDIA Linux GPU (cloud)  
**Features**: All core generation features + **automatic model downloading from HuggingFace/ModelScope**  
**Removed**: Gradio UI, REST API, CLI wizard, LoRA training, Apple Silicon/MLX code

## Features

- **Automatic Model Download**: First-run auto-downloads models from HuggingFace or ModelScope with network detection
- **Text-to-Music**: Generate music from text prompts
- **Lyrics-Conditional**: Generate music with synchronized lyrics
- **Cover Generation**: Transform existing music into new styles
- **Audio Understanding**: Extract BPM, key, lyrics from audio
- **Simple Mode**: Natural language queries (requires LM)
- **Custom LoRA**: Load custom style adapters at inference time
- **Multi-Track**: Generate multi-instrument stems
- **Performance**: Flash attention, torch.compile, int8/fp8 quantization

## Installation

```bash
cd ace-step-auto-download
pip install -e .

# Optional: For quantization
pip install -e ".[quantization]"

# Optional: For flash attention (faster)
pip install -e ".[flash-attn]"

# For model downloading
pip install huggingface_hub  # For HuggingFace downloads
# OR
pip install modelscope       # For ModelScope downloads (China)
```

## Automatic Model Download

On first run, ACE-Step will automatically download models from:
- **HuggingFace Hub** (if Google is accessible)
- **ModelScope** (automatic fallback if HF fails, or primary if in China)

The downloader includes intelligent network detection and automatic fallback between sources.

### Manual Download (Optional)

```bash
# Download all models manually
python -m acestep.model_downloader

# Download specific model
python -m acestep.model_downloader --model acestep-v15-sft

# List available models
python -m acestep.model_downloader --list

# Force source preference
python -m acestep.model_downloader --prefer-source huggingface
```

Or use `huggingface-cli`:

```bash
huggingface-cli download ACE-Step/Ace-Step1.5 --local-dir ./checkpoints
```

## Quick Start

### Python API

```python
from acestep import AceStepPipeline

# Load pipeline (auto-downloads models on first run)
pipe = AceStepPipeline.from_pretrained("./checkpoints")

# Generate music
result = pipe(
    prompt="upbeat jazz piano solo",
    duration=30,
    seed=42,
)

# Save audio
import soundfile as sf
sf.write("output.wav", result["audio"].cpu().numpy().T, result["sample_rate"])
```

### Command Line

```bash
# Basic generation (auto-downloads on first run)
python examples/run_music_generation.py \
    --prompt "upbeat jazz piano" \
    --duration 30 \
    --output jazz.wav

# With lyrics
python examples/run_music_generation.py \
    --prompt "pop song" \
    --lyrics "Hello world, this is my song" \
    --duration 60

# Cover generation
python examples/run_music_generation.py \
    --cover input.mp3 \
    --prompt "orchestral version" \
    --cover-strength 0.7

# Custom LoRA
python examples/run_music_generation.py \
    --prompt "jazz in my custom style" \
    --lora /path/to/my-lora \
    --lora-scale 0.8
```

## Model Download Details

### What Gets Downloaded

**Main Model** (~12 GB total):
- `vae/` - Audio codec (~400MB)
- `Qwen3-Embedding-0.6B/` - Text encoder (~1.2GB)
- `acestep-v15-turbo/` - DiT model (~1.2GB)
- `acestep-5Hz-lm-1.7B/` - LM for simple mode (~3.4GB)

**Optional Models**:
- `acestep-v15-small/` - Better quality DiT
- `acestep-v15-base/` - Best quality DiT
- `acestep-5Hz-lm-0.6B/` - Smaller LM
- `acestep-5Hz-lm-4B/` - Larger LM

### Network Detection

The downloader automatically detects your network environment:
- **Outside China**: Uses HuggingFace Hub → Falls back to ModelScope if HF fails
- **Inside China (Google blocked)**: Uses ModelScope → Falls back to HuggingFace if MS fails

You can override with `--prefer-source`:
```bash
# Force HuggingFace
python -m acestep.model_downloader --prefer-source huggingface

# Force ModelScope
python -m acestep.model_downloader --prefer-source modelscope
```

### Storage Requirements

- **Minimum** (turbo model only): ~4GB
- **Recommended** (with LM): ~10GB
- **Full** (all models): ~25GB

## Advanced Usage

### Custom LoRA

```python
# Load custom LoRA adapter
pipe.load_lora("/path/to/my-lora")
pipe.set_lora_scale(0.8)  # 0.0 = base model, 1.0 = full LoRA

# Generate with custom style
result = pipe(prompt="jazz in my custom style")

# Remove LoRA
pipe.unload_lora()
```

### Quantization (Save VRAM)

```python
pipe = AceStepPipeline.from_pretrained(
    "./checkpoints",
    compile_model=True,  # Required for quantization
    quantization="int8_weight_only",  # or fp8_weight_only, w8a8_dynamic
)
```

### Audio Analysis

```python
metadata = pipe.understand("input.mp3")
print(metadata)  # {"bpm": 120, "key": "C", "lyrics": "...", "caption": "..."}
```

## Project Structure

```
ace-step-auto-download/
├── checkpoints/            # Auto-downloaded models
├── src/acestep/            # Main package
│   ├── __init__.py
│   ├── pipeline.py         # HeartMuLa-style API
│   ├── model_downloader.py # Auto-download with fallback
│   ├── handler.py          # DiT handler
│   ├── llm_inference.py    # LM handler
│   ├── inference.py        # Generation orchestration
│   └── core/               # Core models
├── examples/
│   └── run_music_generation.py
└── pyproject.toml
```

## Differences from ace-step-simple

| Feature | ace-step-simple | ace-step-auto-download |
|---------|----------------|------------------------|
| Model Download | Manual only | **Automatic + Manual** |
| First Run | Requires pre-download | **Auto-downloads** |
| Network Support | N/A | **HF ↔ MS fallback** |
| Dependencies | Minimal | +huggingface_hub/modelscope |

## Credits

- Original ACE-Step 1.5: https://github.com/AudioCraft-Evolution-Step/ACE-Step-1.5
- Refactored structure inspired by HeartMuLa

## License

Apache-2.0 (same as original ACE-Step)
