#!/usr/bin/env python3
"""
Simple entry point for ACE-Step music generation.

Example usage:
    # Basic text-to-music
    python run_music_generation.py --prompt "upbeat jazz piano" --duration 30
    
    # With lyrics
    python run_music_generation.py --prompt "pop song" --lyrics "Hello world" --duration 60
    
    # Cover generation
    python run_music_generation.py --cover input.mp3 --prompt "orchestral version"
    
    # Custom LoRA
    python run_music_generation.py --prompt "jazz" --lora /path/to/lora --lora-scale 0.8
"""
import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from acestep import AceStepPipeline
import torch
import soundfile as sf


def main():
    parser = argparse.ArgumentParser(description="ACE-Step Music Generation")
    
    # Model args
    parser.add_argument("--ckpt", default="./ckpt", help="Checkpoint directory")
    parser.add_argument("--config", default="acestep-v15-turbo", help="DiT model config name")
    parser.add_argument("--device", default="auto", help="Device (auto/cuda/cpu)")
    parser.add_argument("--flash-attn", action="store_true", help="Use flash attention")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--quantization", choices=["int8_weight_only", "fp8_weight_only", "w8a8_dynamic"],
                        help="Quantization mode")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM (no simple mode)")
    
    # Generation args
    parser.add_argument("--prompt", help="Music style description")
    parser.add_argument("--lyrics", help="Lyrics text")
    parser.add_argument("--duration", type=float, default=30.0, help="Duration in seconds")
    parser.add_argument("--bpm", type=float, help="Beats per minute")
    parser.add_argument("--key", help="Musical key (e.g., C, Am)")
    parser.add_argument("--genre", help="Genre tags")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--cfg-scale", type=float, default=4.0, help="Guidance scale")
    parser.add_argument("--steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    
    # Cover/reference args
    parser.add_argument("--cover", help="Input audio path for cover generation")
    parser.add_argument("--cover-strength", type=float, default=0.7, help="Cover transformation strength (0-1)")
    
    # LoRA args
    parser.add_argument("--lora", help="Path to LoRA adapter")
    parser.add_argument("--lora-scale", type=float, default=1.0, help="LoRA scale (0-1)")
    
    # Simple mode
    parser.add_argument("--simple", help="Simple mode natural language query")
    
    # Output args
    parser.add_argument("--output", "-o", default="output.wav", help="Output file path")
    
    # Analyze mode
    parser.add_argument("--analyze", help="Analyze audio file and print metadata")
    
    args = parser.parse_args()
    
    # Load pipeline
    print(f"Loading ACE-Step pipeline from {args.ckpt}...")
    pipe = AceStepPipeline.from_pretrained(
        checkpoint_dir=args.ckpt,
        config_name=args.config,
        device=args.device,
        use_flash_attention=args.flash_attn,
        compile_model=args.compile,
        quantization=args.quantization,
        use_llm=not args.no_llm,
    )
    
    # Load LoRA if specified
    if args.lora:
        print(f"Loading LoRA from {args.lora}...")
        pipe.load_lora(args.lora)
        pipe.set_lora_scale(args.lora_scale)
    
    # Analyze mode
    if args.analyze:
        print(f"Analyzing {args.analyze}...")
        metadata = pipe.understand(args.analyze)
        print("\nMetadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        return
    
    # Generate music
    if args.simple:
        # Simple mode
        print(f"Simple mode: {args.simple}")
        result = pipe.simple(query=args.simple, duration=args.duration, seed=args.seed)
    elif args.cover:
        # Cover generation
        print(f"Generating cover of {args.cover}...")
        result = pipe.cover(
            audio_path=args.cover,
            prompt=args.prompt,
            strength=args.cover_strength,
            seed=args.seed,
        )
    else:
        # Standard generation
        if not args.prompt:
            print("Error: --prompt is required (or use --simple / --cover)")
            return
        
        print(f"Generating music: {args.prompt}")
        if args.lyrics:
            print(f"  Lyrics: {args.lyrics[:50]}...")
        
        result = pipe(
            prompt=args.prompt,
            lyrics=args.lyrics,
            duration=args.duration,
            bpm=args.bpm,
            key=args.key,
            genre=args.genre,
            seed=args.seed,
            guidance_scale=args.cfg_scale,
            num_inference_steps=args.steps,
            batch_size=args.batch_size,
        )
    
    # Save output
    audio = result["audio"]
    sample_rate = result["sample_rate"]
    
    # Handle batch output
    if audio.ndim == 3:  # [B, C, T]
        for i in range(audio.shape[0]):
            output_path = args.output.replace(".wav", f"_{i}.wav") if audio.shape[0] > 1 else args.output
            sf.write(output_path, audio[i].cpu().numpy().T, sample_rate)
            print(f"Saved: {output_path}")
    else:  # [C, T]
        sf.write(args.output, audio.cpu().numpy().T, sample_rate)
        print(f"Saved: {args.output}")
    
    print("Done!")


if __name__ == "__main__":
    main()
