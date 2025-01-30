# Repflix Studios

Generate and download videos with the same prompts across a bunch of different fine-tuned HunyuanVideo models on Replicate. 

This toolkit lets you:
- Generate videos with different style models
- Sweep across multiple parameters to explore variations
- Download results before they expire
- Create pre-generated parameter grids for web exploration

## Setup

1. Install `uv`:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Set your Replicate API token:
```bash
export REPLICATE_API_TOKEN=your_token_here
```

That's it! Each script has its dependencies defined inline, so `uv` will handle everything else automatically.

## Usage

### 1. Generate Individual Videos

For quick tests and single videos, use `generate_video.py`:

```bash
uv run generate_video.py "A TOK style video of a cat playing with yarn"
```

Sweep across parameters to explore variations:
```bash
# Sweep lora_strength and guidance_scale
uv run generate_video.py "A TOK style video of a cat" \
  --param1 lora_strength --param2 guidance_scale

# Custom parameter ranges
uv run generate_video.py "A TOK style video of a cat" \
  --param1 steps --param1-start 25 --param1-end 50
```

### 2. Generate Parameter Grid

To create a complete exploration space across models and parameters, use `generate_grid.py`. This script:
- Takes a set of prompts
- Generates videos for all combinations of:
  - 9 different style models
  - 3 key parameters (lora_strength, guidance_scale, steps)
  - 3 values per parameter
- Records prediction IDs and metadata for later retrieval
- Perfect for creating pre-generated content for web exploration

```bash
uv run generate_grid.py
```

### 3. Download Grid Results

After generating the parameter grid, use `download_grid.py` to:
- Download all generated videos before they expire
- Organize them in a CDN-friendly directory structure
- Create a complete exploration space for web interfaces

```bash
uv run download_grid.py
```

Use `--dry-run` to preview download paths:
```bash
uv run download_grid.py --dry-run
```

Videos are saved to `public/videos/` organized by model and parameters, ready for web serving.

### Key Parameters

These parameters have the most impact on video style and quality:
- `lora_strength`: Controls style adaptation (0.5-1.5 recommended)
- `guidance_scale`: Controls prompt adherence (5-8 recommended)
- `steps`: Number of denoising steps (25-50 recommended)

Less critical parameters:
- `num_frames`: Frames to generate (16-64 recommended)
- `frame_rate`: Frames per second (8-24 recommended)

## Notes

- The scripts use "TOK" in prompts as a placeholder - it's automatically replaced with the appropriate trigger word for each model
- Generated videos are temporarily stored on Replicate and should be downloaded promptly
- The grid generation workflow creates a complete exploration space for web interfaces
- The download script creates a CDN-friendly directory structure based on model and parameters

## Models

The script includes several Hunyuan models with different styles:
- Dune
- Pixar
- Arcane
- La La Land
- Twin Peaks
- Pulp Fiction
- Cowboy Bebop
- The Grand Budapest Hotel
- Spider-Man: Into the Spider-Verse

More models are available but commented out in the code.

## Requirements

- Python 3.11 or higher (installed automatically by `uv` if needed)
- Replicate API token

Each script has its own dependencies defined at the top of the file. The main dependencies are:
- replicate
- tqdm
- numpy

## License

MIT
