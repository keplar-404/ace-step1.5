# ACE-Step Checkpoints (Auto-Download)

Models will be **automatically downloaded** on first run to this directory.

The downloader intelligently detects your network and chooses between HuggingFace Hub and ModelScope with automatic fallback.

## Automatic Download

When you first run ACE-Step, it will automatically download:

```
checkpoints/
├── vae/                          # VAE codec (Oobleck) - ~400MB
├── Qwen3-Embedding-0.6B/         # Text encoder - ~1.2GB
├── acestep-v15-turbo/            # DiT model (default) - ~1.2GB  
└── acestep-5Hz-lm-1.7B/          # LM model (for simple mode) - ~3.4GB
```

**Total**: ~6GB for core functionality, ~10GB with LM

## Network Detection

The downloader automatically detects your network:

- **Google accessible** → HuggingFace Hub (with ModelScope fallback)
- **Google blocked (China)** → ModelScope (with HuggingFace fallback)

## Manual Download (Optional)

If you prefer to download manually or want additional models:

### Using the Built-in Downloader

```bash
# Download main model (default)
python -m acestep.model_downloader

# Download specific optional model
python -m acestep.model_downloader --model acestep-v15-base

# List all available models
python -m acestep.model_downloader --list

# Download everything
python -m acestep.model_downloader --all

# Force HuggingFace or ModelScope
python -m acestep.model_downloader --prefer-source huggingface
python -m acestep.model_downloader --prefer-source modelscope
```

### Using huggingface-cli

```bash
# Main model
huggingface-cli download ACE-Step/Ace-Step1.5 --local-dir checkpoints/

# Optional models
huggingface-cli download ACE-Step/acestep-v15-base --local-dir checkpoints/acestep-v15-base
huggingface-cli download ACE-Step/acestep-5Hz-lm-4B --local-dir checkpoints/acestep-5Hz-lm-4B
```

## Available Models

### Included in Main Download

| Model | Size | VRAM (bf16) | Purpose |
|-------|------|-------------|---------|
| vae | ~400MB | 0.4GB | Audio codec (required) |
| Qwen3-Embedding-0.6B | ~1.2GB | 1.2GB | Text encoder (required) |
| acestep-v15-turbo | ~1.2GB | 1.2GB | Fast DiT (default) |
| acestep-5Hz-lm-1.7B | ~3.4GB | 3.4GB | LM for simple mode |

### Optional DiT Models

| Model | Size | VRAM | Quality | Speed |
|-------|------|------|---------|-------|
| acestep-v15-turbo | ~1.2GB | 1.2GB | Good | **Fast** |
| acestep-v15-small | ~1.8GB | 1.8GB | Better | Medium |
| acestep-v15-base | ~2.6GB | 2.6GB | **Best** | Slow |

### Optional LM Models

| Model | Size | VRAM | Purpose |
|-------|------|------|---------|
| acestep-5Hz-lm-0.6B | ~1.2GB | 1.2GB | Lightweight simple mode |
| acestep-5Hz-lm-1.7B | ~3.4GB | 3.4GB | Default (included in main) |
| acestep-5Hz-lm-4B | ~8GB | 8GB | Best LM quality |

## Minimum Requirements

- **Without LM**: ~4GB VRAM (VAE + text encoder + turbo DiT)
- **With LM**: ~8GB VRAM (add 1.7B LM)
- **Recommended**: 12GB+ VRAM for comfortable generation

## Download Sources

- **HuggingFace**: https://huggingface.co/ACE-Step
- **ModelScope**: https://www.modelscope.cn/organization/ACE-Step
