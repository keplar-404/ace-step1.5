# ACE-Step Checkpoints

Place your ACE-Step model weights here:

```
ckpt/
├── vae/                          # VAE codec (Oobleck)
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
├── Qwen3-Embedding-0.6B/         # Text encoder
│   ├── config.json
│   ├── model.safetensors
│   └── ...
├── acestep-v15-turbo/            # DiT model (required)
│   ├── config.json
│   ├── model.safetensors
│   └── silence_latent.pt
├── acestep-v15-small/            # DiT model (optional)
├── acestep-v15-base/             # DiT model (optional)
└── acestep-5Hz-lm-1.7B/          # LM model (optional, for simple mode)
    ├── config.json
    ├── model.safetensors
    └── ...
```

## Download Instructions

Download models from HuggingFace:
- **VAE**: https://huggingface.co/ACE-Step/acestep-v15-base/tree/main/vae
- **Text Encoder**: https://huggingface.co/ACE-Step/acestep-v15-base/tree/main/Qwen3-Embedding-0.6B
- **DiT Models**: https://huggingface.co/ACE-Step/acestep-v15-turbo
- **LM Models**: https://huggingface.co/ACE-Step/acestep-5Hz-lm-1.7B

Or use `huggingface-cli`:

```bash
# Download main components
huggingface-cli download ACE-Step/acestep-v15-base vae --local-dir ckpt/
huggingface-cli download ACE-Step/acestep-v15-base Qwen3-Embedding-0.6B --local-dir ckpt/

# Download DiT model
huggingface-cli download ACE-Step/acestep-v15-turbo --local-dir ckpt/acestep-v15-turbo

# (Optional) Download LM for simple mode
huggingface-cli download ACE-Step/acestep-5Hz-lm-1.7B --local-dir ckpt/acestep-5Hz-lm-1.7B
```

## Model Sizes

| Model | Parameters | VRAM (bf16) | Notes |
|-------|------------|-------------|-------|
| VAE | ~100M | 0.4GB | Required |
| Text Encoder | ~0.6B | 1.2GB | Required |
| acestep-v15-turbo | ~0.6B | 1.2GB | Fast, good quality |
| acestep-v15-small | ~0.9B | 1.8GB | Better quality |
| acestep-v15-base | ~1.3B | 2.6GB | Best quality |
| acestep-5Hz-lm-1.7B | ~1.7B | 3.4GB | For simple mode |

**Minimum VRAM**: ~4GB for turbo model (VAE + text encoder + DiT)
**Recommended**: 8GB+ for comfortable generation with LM
