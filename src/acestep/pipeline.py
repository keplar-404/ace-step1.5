"""
AceStepPipeline - Clean HeartMuLa-style API for ACE-Step 1.5
"""
import os
from typing import Optional, List, Union
from pathlib import Path

from .handler import AceStepHandler
from .llm_inference import LLMHandler
from .inference import generate_music, understand_music, create_sample, GenerationParams


class AceStepPipeline:
    """
    HeartMuLa-style pipeline for ACE-Step music generation.
    
    Example:
        ```python
        pipe = AceStepPipeline.from_pretrained("./ckpt")
        
        # Text-to-music
        result = pipe(prompt="upbeat jazz piano", duration=30)
        
        # With lyrics
        result = pipe(prompt="pop song", lyrics="Hello world", duration=60)
        
        # Cover generation  
        result = pipe.cover("input.mp3", prompt="orchestral version")
        
        # Custom LoRA
        pipe.load_lora("/path/to/lora")
        pipe.set_lora_scale(0.8)
        result = pipe(prompt="jazz in my custom style")
        ```
    """
    
    def __init__(
        self,
        handler: AceStepHandler,
        llm_handler: Optional[LLMHandler] = None,
    ):
        """Initialize pipeline with handlers"""
        self.handler = handler
        self.llm_handler = llm_handler
    
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str = "./ckpt",
        config_name: str = "acestep-v15-turbo",
        device: str = "auto",
        use_flash_attention: bool = False,
        compile_model: bool = False,
        offload_to_cpu: bool = False,
        offload_dit_to_cpu: bool = False,
        quantization: Optional[str] = None,
        use_llm: bool = True,
        llm_model_name: Optional[str] = None,
        llm_device: Optional[str] = None,
    ):
        """
        Load ACE-Step pipeline from checkpoint directory.
        
        Args:
            checkpoint_dir: Path to checkpoints (should contain vae/, DiT models, text encoder)
            config_name: DiT model config name (e.g., "acestep-v15-turbo")
            device: Device for DiT ("auto", "cuda", "cpu")
            use_flash_attention: Use flash attention for DiT (requires flash_attn)
            compile_model: Use torch.compile for DiT
            offload_to_cpu: Offload models to CPU when idle
            offload_dit_to_cpu: Offload DiT to CPU when idle
            quantization: Quantization mode ("int8_weight_only", "fp8_weight_only", "w8a8_dynamic")
            use_llm: Load LLM for "simple mode" and planning
            llm_model_name: LLM model path (default: acestep-5Hz-lm-1.7B)
            llm_device: LLM device (default: same as DiT)
            
        Returns:
            AceStepPipeline instance
        """
        # Initialize DiT handler
        handler = AceStepHandler()
        status, success = handler.initialize_service(
            project_root=checkpoint_dir,
            config_path=config_name,
            device=device,
            use_flash_attention=use_flash_attention,
            compile_model=compile_model,
            offload_to_cpu=offload_to_cpu,
            offload_dit_to_cpu=offload_dit_to_cpu,
            quantization=quantization,
        )
        
        if not success:
            raise RuntimeError(f"Failed to initialize DiT handler: {status}")
        
        # Initialize LLM handler (optional)
        llm_handler = None
        if use_llm:
            llm_device = llm_device or device
            llm_model_name = llm_model_name or "acestep-5Hz-lm-1.7B"
            llm_path = os.path.join(checkpoint_dir, llm_model_name)
            
            if os.path.exists(llm_path):
                llm_handler = LLMHandler()
                llm_status, llm_success = llm_handler.initialize_llm_service(
                    project_root=checkpoint_dir,
                    llm_model_path=llm_model_name,
                    device=llm_device,
                )
                if not llm_success:
                    print(f"Warning: LLM failed to initialize: {llm_status}")
                    llm_handler = None
            else:
                print(f"Warning: LLM model not found at {llm_path}, proceeding without LLM")
        
        return cls(handler=handler, llm_handler=llm_handler)
    
    def __call__(
        self,
        prompt: Optional[str] = None,
        lyrics: Optional[str] = None,
        duration: float = 30.0,
        bpm: Optional[float] = None,
        key: Optional[str] = None,
        genre: Optional[str] = None,
        reference_audio: Optional[str] = None,
        reference_strength: float = 0.5,
        seed: Optional[int] = None,
        guidance_scale: float = 4.0,
        num_inference_steps: int = 50,
        batch_size: int = 1,
        **kwargs,
    ):
        """
        Generate music.
        
        Args:
           prompt: Text description of music style
            lyrics: Lyrics text (optional)
            duration: Duration in seconds
            bpm: Beats per minute (optional)
            key: Musical key (e.g., "C", "Am")
            genre: Genre tags
            reference_audio: Path to reference audio for style transfer
            reference_strength: How much to follow reference (0-1)
            seed: Random seed for reproducibility
            guidance_scale: CFG scale (higher = more prompt adherence)
            num_inference_steps: Diffusion steps (more = higher quality)
            batch_size: Number of outputs to generate
            
        Returns:
            dict with "audio" (tensor), "sample_rate", "metadata"
        """
        params = GenerationParams(
            prompt=prompt or "",
            lyrics=lyrics,
            duration_sec=duration,
            bpm=bpm,
            key=key,
            genre=genre,
            reference_audio_path=reference_audio,
            audio_cover_strength=1.0 - reference_strength if reference_audio else None,
            seed=seed,
            cfg_scale=guidance_scale,
            infer_steps=num_inference_steps,
            batch_size=batch_size,
            **kwargs,
        )
        
        return generate_music(
            handler=self.handler,
            llm_handler=self.llm_handler,
            params=params,
        )
    
    def cover(
        self,
        audio_path: str,
        prompt: Optional[str] = None,
        strength: float = 0.7,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """
        Generate cover version of existing audio.
        
        Args:
            audio_path: Path to input audio
            prompt: Style description (e.g., "orchestral version")
            strength: Cover strength (0-1, higher = more transformation)
            seed: Random seed
            
        Returns:
            dict with generated audio
        """
        return self(
            reference_audio=audio_path,
            reference_strength=1.0 - strength,
            prompt=prompt or "cover version",
            seed=seed,
            **kwargs,
        )
    
    def understand(self, audio_path: str):
        """
        Analyze audio and extract metadata (BPM, key, lyrics, caption).
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            dict with metadata
        """
        return understand_music(
            handler=self.handler,
            audio_path=audio_path,
        )
    
    def simple(self, query: str, **kwargs):
        """
        Simple mode - natural language query generates everything.
        
        Args:
            query: Natural language description (e.g., "happy jazz song about cats")
            
        Returns:
            dict with generated audio
        """
        if self.llm_handler is None:
            raise ValueError("Simple mode requires LLM. Load pipeline with use_llm=True")
        
        sample = create_sample(
            handler=self.handler,
            llm_handler=self.llm_handler,
            query=query,
        )
        
        return self(**sample, **kwargs)
    
    # LoRA methods
    def load_lora(self, lora_path: str):
        """Load custom LoRA adapter for style control"""
        return self.handler.load_lora(lora_path)
    
    def unload_lora(self):
        """Remove LoRA adapter"""
        return self.handler.unload_lora()
    
    def set_lora_scale(self, scale: float):
        """Set LoRA influence scale (0-1)"""
        return self.handler.set_lora_scale(scale)
    
    
    def get_lora_status(self):
        """Get current LoRA status"""
        return self.handler.get_lora_status()
