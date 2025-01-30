# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "replicate",       # Replicate.com API client
#   "tqdm",           # Progress bars
#   "numpy",          # Numerical operations
# ]
# ///

import replicate
import argparse
import re
from tqdm import tqdm
import numpy as np
from itertools import product

# Model configurations
MODELS = [
    # "deepfates/hunyuan-her:a4289ea2d95ec71b94fcbf243d47e406b473b93e5bb700c5ec8966730d8ca63d",
    # "deepfates/hunyuan-rrr:4f58a5abad34e6c52c5388f5bd866f6955176db0be73fd7d99164af46f4963fb",
    "deepfates/hunyuan-dune:4fbe2f9a8c5f5912fa4bba528d5b2e27494557ab922356e3f6374e3353e5c36e",
    # "deepfates/hunyuan-joker:7a1025aea09dce5abeeca5bf3555e3031eff118154b1e2e59f71b644bda98757",
    "deepfates/hunyuan-pixar:2f2d7b64fc02bee25ca1a1c48a2adaad69c410f34ded59aaf5b9d9f898fc554c",
    "deepfates/hunyuan-arcane:d294d8b37fd60ff1499e631d054250ae51709fe87e8e32d563dd98c610a40bad",
    # "deepfates/hunyuan-avatar:e02ecac7d322c9bfaa4d72b5f5877e1c2a70769a247a2be0d426d4a1669e9e47",
    # "deepfates/hunyuan-inception:a471cf828b9f03ea639c745bbd27dc931e6a575dfbbffb2fa7cad6d71a0dab9e",
    "deepfates/hunyuan-la-la-land:4c5d76fd0aba10043c23501c2d62142db7cd9d42b6445d0efe35950563a64d78",
    # "deepfates/hunyuan-westworld:e010979271e08c7c51cb50eba7efe208c175a7f0a6deeab736c341546eab1013",
    "deepfates/hunyuan-twin-peaks:9a578855228e41283cf23df1af710f71490e3bf9503bb0baaa485fa8c08bf2a6",
    # "deepfates/hunyuan-spiderverse:0149fb2019d637f6b3a610c68b04c84b629f71b8d1b838960147f8e8cca705c5",
    # "deepfates/hunyuan-interstellar:0c05e2390d57ba0c3e5b2222cfa2830fd4de55e3b593ad8df7cf7101a7996bec",
    # "deepfates/hunyuan-blade-runner:43195cc4a870109a20805229ba9e5801a875005f019e0ee7ae3a69c4b654228c",
    "deepfates/hunyuan-pulp-fiction:9e541e836c23a1ffad90bee5edca367968bb2816b13a593f85aceb35ee46a527",
    "deepfates/hunyuan-cowboy-bebop:aacf6ff798e14d44505352c4ccb14b3833094a16b395b847657636f232571975",
    # "deepfates/hunyuan-indiana-jones:f4a9ac08b2f70053e70e8b7777cf0b47e365c995b5865d22a04432aebc8ad526",
    # "deepfates/hunyuan-atomic-blonde:eae534f331ec515fee16b5533e0211a40f83bcb11b147e0b5cf6b4d2d590f851",
    # "deepfates/hunyuan-asteroid-city:e1c83b454fa96446b887e946b358279ed43a52324ffadae6c415bd0ed1effb48",
    # "deepfates/hunyuan-game-of-thrones:462def949f4f0301fe2f6a1a7364a162e6577170c22d6b13a0714c55633c4b2e",
    # "deepfates/hunyuan-blade-runner-2049:5c4ea69d26b3a4611f4929e9267ff916d132e55afb08ae6dd84fb8c98511ac69",
    # "deepfates/hunyuan-the-matrix-trilogy:e84c3dd23de21d8696fc4961b3960862a7efaf868393382119dab5a11acb0ad9",
    # "deepfates/hunyuan-mad-max-fury-road:0754458f856830986e26b6b975c626e4c0173b44fb661f62dd66f59bab9412c0",
    # "deepfates/hunyuan-the-lord-of-the-rings:0afc10af3117ce053b0430a8465bf0e500a1b7432305b813a4b5ea01393f4612",
    # "deepfates/hunyuan-the-neverending-story:af916146bfbf939240b5ddeac06c64d072142ce1579fa036e3111951ca6dc3ee",
    # "deepfates/hunyuan-neon-genesis-evangelion:94d5c1dec276a4d7d660064bde3cf1a03ff9918e48b0a2bf3594d430b885397e",
    "deepfates/hunyuan-the-grand-budapest-hotel:92da6ced97eac105bc66dcf75c64ec5e11e11dced920d7708c396c3b429929b6",
    # "deepfates/hunyuan-pirates-of-the-caribbean:763796a6d83dc3b1f4c8406bc29bf63ef46295d9c55108f83e2759c201a7436c",
    # "deepfates/hunyuan-once-upon-a-time-in-hollywood:47e048cf3de47305fccb9410f2d9c9f89e626a38e1d6dc08561698c5d8ffa2b3",
    "deepfates/hunyuan-spider-man-into-the-spider-verse:4bc14ccdf43518bf36a9b35b9414b1b21fc34742c51c458133a0860af532f320",
]

# Default model parameters
DEFAULT_PARAMS = {
    "frame_rate": 16,
    "guidance_scale": 6,
    "width": 640,
    "height": 360,
    "lora_strength": 1,
    "num_frames": 66,
    "seed": 53,
    "steps": 50
}

# Parameter sweep configurations
SWEEP_DEFAULTS = {
    'lora_strength': {
        'start': 0.7,
        'end': 1.3,
        'steps': 3,
        'description': 'Controls strength of style adaptation (0.5-1.5 recommended)',
        'type': float
    },
    'guidance_scale': {
        'start': 5.0,
        'end': 9.0,
        'steps': 3,
        'description': 'Controls adherence to prompt (5-8 recommended)',
        'type': float
    },
    'steps': {
        'start': 25,
        'end': 50,
        'steps': 3,
        'description': 'Number of denoising steps (25-50 recommended)',
        'type': int
    },
    'num_frames': {
        'start': 16,
        'end': 64,
        'steps': 3,
        'description': 'Number of frames to generate (16-64 recommended)',
        'type': int
    },
    'frame_rate': {
        'start': 8,
        'end': 24,
        'steps': 3,
        'description': 'Frames per second (8-24 recommended)',
        'type': int
    }
}

def get_trigger_word(model_name):
    """Extract first 5 consonants from model name."""
    match = re.search(r'hunyuan-(.+?)(?:/|$)', model_name.split(':')[0])
    if match:
        base = match.group(1).replace('-', '')
        consonants = ''.join(c for c in base if c.lower() not in 'aeiou')
        return consonants[:5].upper()
    return None

def generate_video(prompt, model, params=None):
    """Generate a video using the specified model and parameters."""
    # Get model hash and trigger word
    model_hash = model.split(':')[1]
    trigger_word = get_trigger_word(model)
    
    if not trigger_word:
        raise ValueError(f"Could not extract trigger word from model: {model}")
        
    # Replace TOK with model-specific trigger word
    modified_prompt = prompt.replace('TOK', trigger_word)
    
    # Merge default and custom parameters
    final_params = DEFAULT_PARAMS.copy()
    if params:
        final_params.update(params)
    
    # Add prompt to parameters
    final_params["prompt"] = modified_prompt
    
    # Create prediction
    prediction = replicate.predictions.create(
        version=model_hash,
        input=final_params
    )
    
    return prediction

def linear_sequence(start, end, steps, param_type=float):
    """Generate a linear sequence between start and end values."""
    sequence = np.linspace(start, end, steps)
    return [param_type(x) for x in sequence]

def fibonacci_sequence(start, end, steps, param_type=float):
    """Generate a Fibonacci-like sequence between start and end values."""
    if steps < 2:
        return [param_type(start)]
    
    # Generate standard Fibonacci sequence
    fib = [1, 1]
    while len(fib) < steps:
        fib.append(fib[-1] + fib[-2])
    
    # Scale to desired range
    fib = np.array(fib)
    scaled = start + (end - start) * (fib - min(fib)) / (max(fib) - min(fib))
    
    # Sort in descending order if start > end
    if start > end:
        scaled = sorted(scaled, reverse=True)
    
    # Generate Fibonacci sequence
    sequence = scaled[:steps]
    
    return [param_type(x) for x in sequence]

def main():
    parser = argparse.ArgumentParser(
        description='Generate videos using Replicate models with parameter sweeps',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example:\n' +
               '  python generate_video.py "A TOK style video of a cat" --guidance-scale 7 --steps 25\n' +
               '  python generate_video.py "A TOK style video of a cat" --param1 lora_strength --param2 guidance_scale\n' +
               '  python generate_video.py "A TOK style video of a cat" --param1 steps --param1-start 25 --param1-end 50\n\n' +
               'Available parameters for sweeping:\n' +
               '\n'.join(f'  {param}: {config["description"]}' for param, config in SWEEP_DEFAULTS.items())
    )
    
    # Basic arguments
    parser.add_argument('prompt', help='The prompt to use for video generation')
    parser.add_argument('--model', help='Specific model to use (if not specified, uses all models)')
    
    # Direct parameter settings
    parser.add_argument('--guidance-scale', type=float, help='Guidance scale value')
    parser.add_argument('--steps', type=int, help='Number of steps')
    parser.add_argument('--lora-strength', type=float, help='LoRA strength value')
    parser.add_argument('--num-frames', type=int, help='Number of frames')
    parser.add_argument('--frame-rate', type=int, help='Frame rate')
    
    # Parameter sweep arguments
    parser.add_argument('--param1', choices=list(SWEEP_DEFAULTS.keys()),
                      help='First parameter to sweep (see available parameters below)')
    parser.add_argument('--param1-start', type=float,
                      help='Starting value for param1 (uses default if not specified)')
    parser.add_argument('--param1-end', type=float,
                      help='Ending value for param1 (uses default if not specified)')
    parser.add_argument('--param1-steps', type=int,
                      help='Number of steps for param1 (uses default if not specified)')
    
    parser.add_argument('--param2', choices=list(SWEEP_DEFAULTS.keys()),
                      help='Second parameter to sweep (see available parameters below)')
    parser.add_argument('--param2-start', type=float,
                      help='Starting value for param2 (uses default if not specified)')
    parser.add_argument('--param2-end', type=float,
                      help='Ending value for param2 (uses default if not specified)')
    parser.add_argument('--param2-steps', type=int,
                      help='Number of steps for param2 (uses default if not specified)')
    
    parser.add_argument('--sequence-type', choices=['linear', 'fibonacci'], default='linear',
                      help='Type of sequence for parameter sweeps (default: linear)')
    
    # Basic video parameters
    parser.add_argument('--width', type=int, default=DEFAULT_PARAMS['width'],
                      help=f'Video width (default: {DEFAULT_PARAMS["width"]})')
    parser.add_argument('--height', type=int, default=DEFAULT_PARAMS['height'],
                      help=f'Video height (default: {DEFAULT_PARAMS["height"]})')
    parser.add_argument('--seed', type=int, default=DEFAULT_PARAMS['seed'],
                      help=f'Random seed (default: {DEFAULT_PARAMS["seed"]})')
    
    args = parser.parse_args()
    
    # Build base custom parameters
    custom_params = {}
    if args.width:
        custom_params['width'] = args.width
    if args.height:
        custom_params['height'] = args.height
    if args.seed:
        custom_params['seed'] = args.seed
    if args.guidance_scale:
        custom_params['guidance_scale'] = args.guidance_scale
    if args.steps:
        custom_params['steps'] = args.steps
    if args.lora_strength:
        custom_params['lora_strength'] = args.lora_strength
    if args.num_frames:
        custom_params['num_frames'] = args.num_frames
    if args.frame_rate:
        custom_params['frame_rate'] = args.frame_rate

    # Apply defaults for parameter sweeps if needed
    param_sequences = {}
    sequence_func = fibonacci_sequence if args.sequence_type == 'fibonacci' else linear_sequence
    
    for param_num in range(1, 3):
        param_name = getattr(args, f'param{param_num}')
        if param_name:
            defaults = SWEEP_DEFAULTS[param_name]
            start = getattr(args, f'param{param_num}_start') or defaults['start']
            end = getattr(args, f'param{param_num}_end') or defaults['end']
            steps = getattr(args, f'param{param_num}_steps') or defaults['steps']
            param_sequences[param_name] = sequence_func(start, end, steps, defaults['type'])

    # Determine which models to use
    models_to_use = [args.model] if args.model else MODELS
    
    # Calculate total runs
    if param_sequences:
        param_combinations = list(product(*param_sequences.values()))
        total_runs = len(models_to_use) * len(param_combinations)
    else:
        param_combinations = [{}]
        total_runs = len(models_to_use)
    
    print(f"\nGenerating videos for prompt: {args.prompt}")
    print(f"Using {len(models_to_use)} models")
    if param_sequences:
        print("\nParameter sweeps:")
        for param, sequence in param_sequences.items():
            print(f"{param}: {sequence}")
    print(f"\nTotal generations to run: {total_runs}\n")
    
    with tqdm(total=total_runs, desc="Overall progress") as pbar:
        for i, model in enumerate(models_to_use, 1):
            model_name = model.split('/')[1].split(':')[0]  # Extract readable model name
            tqdm.write(f"\nProcessing model {i}/{len(models_to_use)}: {model_name}")
            
            for params in param_combinations:
                try:
                    # Create parameter dict for this run
                    run_params = custom_params.copy()
                    if param_sequences:
                        for (param_name, param_value) in zip(param_sequences.keys(), params):
                            run_params[param_name] = param_value
                    
                    param_str = ', '.join(f"{k}={v:.2f}" for k, v in run_params.items())
                    tqdm.write(f"\nGenerating with parameters: {param_str}")
                    
                    prediction = generate_video(args.prompt, model, run_params)
                    tqdm.write(f"Prediction ID: {prediction.id}")
                    tqdm.write(f"Status URL: https://replicate.com/p/{prediction.id}")
                    
                except Exception as e:
                    tqdm.write(f"\nError with model {model_name}: {str(e)}")
                    continue
                
                pbar.update(1)

if __name__ == "__main__":
    main() 
