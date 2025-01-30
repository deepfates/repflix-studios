# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "replicate",       # Replicate.com API client
#   "tqdm",           # Progress bars
# ]
# ///

import json
import replicate
from pathlib import Path
from urllib.request import urlretrieve
from tqdm import tqdm
import time
import sys

def get_cdn_path(metadata):
    """Generate CDN-friendly path from metadata."""
    model_name = metadata['model'].split('/')[1].split(':')[0].replace('hunyuan-', '')
    params = metadata['parameters']
    scene_hash = hash(metadata['prompt']) % 1000  # Simple scene identifier
    
    return (f"{model_name}/"
            f"scene_{scene_hash:03d}/"
            f"lora_{params['lora_strength']:.2f}/"
            f"cfg_{params['guidance_scale']:.1f}/"
            f"steps_{params['steps']:02d}.mp4")

def download_prediction(metadata, output_dir, dry_run=False):
    """Download a single prediction's output."""
    try:
        cdn_path = get_cdn_path(metadata)
        output_path = output_dir / cdn_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if dry_run:
            print(f"Would download to: {output_path}")
            return True
            
        prediction = replicate.predictions.get(metadata['prediction_id'])
        
        # Wait for completion if needed
        while prediction.status == "processing":
            time.sleep(5)
            prediction = replicate.predictions.get(metadata['prediction_id'])
        
        if prediction.status != "succeeded":
            print(f"Prediction failed: {metadata['prediction_id']}")
            return False
            
        # Get video URL and download
        url = prediction.output[0] if isinstance(prediction.output, list) else prediction.output
        urlretrieve(url, output_path)
        return True
        
    except Exception as e:
        print(f"Error downloading {metadata['prediction_id']}: {e}")
        return False

def main():
    dry_run = "--dry-run" in sys.argv
    output_dir = Path("public/videos")
    
    # Load prediction metadata
    with open("outputs/all_predictions.json") as f:
        predictions = json.load(f)
    
    total_predictions = len(predictions)
    print(f"\nFound {total_predictions} predictions to download")
    
    if not dry_run and input("\nContinue? (y/n): ").lower() != 'y':
        return
    
    # Download all predictions
    success = 0
    with tqdm(total=total_predictions, desc="Downloading grid") as pbar:
        for metadata in predictions:
            if download_prediction(metadata, output_dir, dry_run):
                success += 1
            pbar.update(1)
    
    print(f"\nProcessed {success}/{total_predictions} videos successfully")
    print(f"Videos saved to: {output_dir}")

if __name__ == "__main__":
    main() 
