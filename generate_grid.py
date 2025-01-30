# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "replicate",       # Replicate.com API client
#   "tqdm",           # Progress bars
#   "numpy",          # Numerical operations
# ]
# ///

import replicate
import json
from datetime import datetime
from pathlib import Path
from itertools import product
from tqdm import tqdm
import re
import hashlib
from time import sleep

# Model configurations
MODELS = [
    "deepfates/hunyuan-dune:4fbe2f9a8c5f5912fa4bba528d5b2e27494557ab922356e3f6374e3353e5c36e",
    "deepfates/hunyuan-pixar:2f2d7b64fc02bee25ca1a1c48a2adaad69c410f34ded59aaf5b9d9f898fc554c",
    "deepfates/hunyuan-arcane:d294d8b37fd60ff1499e631d054250ae51709fe87e8e32d563dd98c610a40bad",
    "deepfates/hunyuan-la-la-land:4c5d76fd0aba10043c23501c2d62142db7cd9d42b6445d0efe35950563a64d78",
    "deepfates/hunyuan-twin-peaks:9a578855228e41283cf23df1af710f71490e3bf9503bb0baaa485fa8c08bf2a6",
    "deepfates/hunyuan-pulp-fiction:9e541e836c23a1ffad90bee5edca367968bb2816b13a593f85aceb35ee46a527",
    "deepfates/hunyuan-cowboy-bebop:aacf6ff798e14d44505352c4ccb14b3833094a16b395b847657636f232571975",
    "deepfates/hunyuan-the-grand-budapest-hotel:92da6ced97eac105bc66dcf75c64ec5e11e11dced920d7708c396c3b429929b6",
    "deepfates/hunyuan-spider-man-into-the-spider-verse:4bc14ccdf43518bf36a9b35b9414b1b21fc34742c51c458133a0860af532f320",
]

# Parameter grid for exploration
PARAMETER_GRID = {
    'lora_strength': [0.7, 1.15, 1.35],
    'guidance_scale': [6.0, 7.0, 9.0],
    'steps': [25, 35, 50]
}

# Test prompts
PROMPTS = [
    # Tests how models handle sci-fi lighting effects and subtle character reactions
    "A video in the style of TOK, TOK The video clip features a scientist in a brightly lit laboratory. They are wearing a white lab coat and protective goggles, standing in front of a large glass containment chamber. The chamber contains a softly pulsing light that casts shifting patterns across the scientist's face. As they observe the phenomenon, their expression changes from curiosity to amazement. The laboratory equipment around them creates interesting reflections and shadows, and the overall atmosphere is one of discovery and wonder.",
    
    # Tests how models handle architectural space and purposeful character movement
    "A video in the style of TOK, TOK The video clip shows a person walking through an elegant hotel lobby. The space is grand and open, with marble floors that reflect the light from crystal chandeliers above. The person is wearing formal attire and moving with purpose toward the center of the frame. Ornate columns line the sides of the lobby, and the lighting creates a warm, luxurious atmosphere. Other guests can be seen in the background, but they are slightly out of focus.",
    
    # Tests how models handle dynamic lighting interaction with environment and texture
    "A video in the style of TOK, TOK The video clip depicts someone exploring an ancient temple chamber. They are wearing explorer's gear and carrying a light source that illuminates the space around them. The walls are covered in carved symbols that catch the light as the person moves past them. Dust particles float in the beam of light, and the stone surfaces have a weathered, textured appearance. The chamber feels vast but enclosed, with architectural details that suggest great age and significance.",
    
    # Tests how models frame two-person scenes and handle intimate dramatic tension
    "A video in the style of TOK, TOK The video clip shows two people having an intense conversation in a diner booth. The booth has red vinyl seats and a window beside it showing nighttime city lights. One person is leaning forward, wearing a leather jacket, while the other sits back against the booth, wearing a white dress shirt with rolled-up sleeves. The fluorescent lights above cast stark shadows, and a coffee cup sits steaming on the formica table between them. The scene has a late-night atmosphere of urban tension.",
    
    # Tests how models handle controlled physical movement and environmental atmosphere
    "A video in the style of TOK, TOK The video clip features a martial arts master demonstrating a move in a traditional training hall. They are wearing loose black training clothes, moving with controlled precision on the polished wooden floor. Sunlight streams through high windows in diagonal beams, catching the dust motes in the air. The walls are lined with wooden practice weapons and scrolls, and the space has an atmosphere of disciplined energy. The movement is graceful but powerful.",
    
    # Tests how models handle environmental effects (wind, weather) on character and scene
    "A video in the style of TOK, TOK The video clip depicts a person standing on a windy rooftop helipad. They are wearing a long coat that moves dramatically in the wind, and their hair is being blown back from their face. The sky above shows threatening storm clouds, and the city lights below create a grid of illumination through the gathering darkness. Lightning occasionally illuminates the clouds, creating dramatic moments of bright contrast.",
    
    # Tests how models handle futuristic UI interaction and subtle hand movements
    "A video in the style of TOK, TOK The video clip shows someone working at a complex control panel. They are wearing a sleek headset and touching holographic displays that float in the air before them. The room is dark except for the blue glow of multiple screens and status lights, which reflect off the metallic surfaces of the equipment. Their fingers move with precise gestures, and each touch causes ripples of light across the interfaces.",
    
    # Tests how models handle reflections and emotional transitions in close-up
    "A video in the style of TOK, TOK The video clip features a performer backstage looking into a mirror. They are wearing stage makeup and a elaborate costume that catches the light. The mirror is surrounded by warm bulbs, creating a soft glow around their reflection. Their expression shifts as they study their reflection, and various emotions play across their face. The background shows glimpses of other performers and stage equipment, slightly out of focus.",
    
    # Tests how models handle multiple moving figures and dynamic environment
    "A video in the style of TOK, TOK The video clip shows a busy restaurant kitchen during service. A chef in white works at the center station, while other kitchen staff move purposefully around them. Steam rises from pots, and flames occasionally flare from the stovetops. The stainless steel surfaces are bright and reflective under the overhead lights, and the overall scene has an energy of coordinated chaos. The movement is quick but professional, with everyone seeming to know exactly where they need to be."
]

# Default parameters that won't change
DEFAULT_PARAMS = {
    'width': 640,
    'height': 360,
    'frame_rate': 16,
    'num_frames': 66,
    'seed': 53
}

def get_trigger_word(model_name):
    """Extract first 5 consonants from model name."""
    match = re.search(r'hunyuan-(.+?)(?:/|$)', model_name.split(':')[0])
    if match:
        base = match.group(1).replace('-', '')
        consonants = ''.join(c for c in base if c.lower() not in 'aeiou')
        return consonants[:5].upper()
    return None

def generate_video(prompt, model, params):
    """Generate a video using the specified model and parameters."""
    # Get model hash and trigger word
    model_hash = model.split(':')[1]
    trigger_word = get_trigger_word(model)
    
    if not trigger_word:
        raise ValueError(f"Could not extract trigger word from model: {model}")
        
    # Replace TOK with model-specific trigger word
    modified_prompt = prompt.replace('TOK', trigger_word)
    
    # Add prompt to parameters
    params["prompt"] = modified_prompt
    
    # Create prediction
    prediction = replicate.predictions.create(
        version=model_hash,
        input=params
    )
    
    return prediction

def generate_video_dry_run(prompt, model, params):
    """Generate fake prediction data for dry run."""
    # Create deterministic fake prediction ID from inputs
    input_string = f"{model}{prompt}{str(params)}"
    fake_id = hashlib.md5(input_string.encode()).hexdigest()[:12]
    
    class FakePrediction:
        def __init__(self, id):
            self.id = id
    
    return FakePrediction(fake_id)

def save_prediction_metadata(prediction, model, prompt, params):
    """Save prediction metadata to JSON file."""
    metadata = {
        'prediction_id': prediction.id,
        'model': model,
        'prompt': prompt,
        'parameters': params,
        'status_url': f"https://replicate.com/p/{prediction.id}",
        'timestamp': datetime.now().isoformat()
    }
    
    # Create outputs directory if it doesn't exist
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    # Save to JSON file named with prediction ID
    output_file = output_dir / f"{prediction.id}.json"
    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata

def main():
    # Add argument parsing
    import sys
    dry_run = "--dry-run" in sys.argv
    
    # Generate all combinations
    all_combinations = list(product(
        MODELS,
        PROMPTS,
        PARAMETER_GRID['lora_strength'],
        PARAMETER_GRID['guidance_scale'],
        PARAMETER_GRID['steps']
    ))
    
    total_runs = len(all_combinations)
    print(f"\nTotal generations to run: {total_runs}")
    print(f"Estimated cost: ${total_runs * 0.15:.2f} - ${total_runs * 0.25:.2f}")
    
    # Ask for confirmation unless dry run
    if not dry_run and input("\nContinue? (y/n): ").lower() != 'y':
        return
    
    all_metadata = []
    
    with tqdm(total=total_runs, desc="Overall progress") as pbar:
        for model, prompt, lora_strength, guidance_scale, steps in all_combinations:
            model_name = model.split('/')[1].split(':')[0]
            
            params = {
                'lora_strength': lora_strength,
                'guidance_scale': guidance_scale,
                'steps': steps,
                **DEFAULT_PARAMS
            }
            
            try:
                tqdm.write(f"\nGenerating with model {model_name}")
                tqdm.write(f"Prompt: {prompt}")
                tqdm.write(f"Parameters: {params}")
                
                # Use dry run function if specified
                prediction = (generate_video_dry_run(prompt, model, params) 
                            if dry_run else 
                            generate_video(prompt, model, params))
                
                metadata = save_prediction_metadata(prediction, model, prompt, params)
                all_metadata.append(metadata)
                
                tqdm.write(f"Prediction ID: {prediction.id}")
                tqdm.write(f"Status URL: {metadata['status_url']}")
                
                # Add delay between calls (skip during dry run)
                if not dry_run:
                    sleep(0.5)  # 2 second delay between API calls
                
            except Exception as e:
                tqdm.write(f"\nError with model {model_name}: {str(e)}")
                continue
            
            pbar.update(1)
    
    # Save all metadata to a single file
    output_dir = Path('outputs_dry_run' if dry_run else 'outputs')
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / 'all_predictions.json', 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    if dry_run:
        print(f"\nDry run complete! Check {output_dir} for generated metadata files")

if __name__ == "__main__":
    main()
