# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "yt-dlp",          # Video downloading
#   "scenedetect",     # Scene change detection
#   "python-slugify",  # Clean text for URLs/filenames
#   "moviepy==1.0.3",  # Video manipulation
#   "replicate",       # Replicate.com API client
#   "opencv-python",   # Required by scenedetect
#   "tqdm",           # Progress bars
# ]
# ///

"""
Hunyuan Video Fine-tuning Processor
==================================

This script automates the process of creating fine-tuning datasets for the Hunyuan video model.
It handles the complete pipeline from video download to model training:

1. Downloads videos (e.g., from YouTube)
2. Processes them into appropriate clip lengths
3. Generates consistent naming and trigger words
4. Creates training datasets
5. Initiates training on Replicate
6. Pushes models to HuggingFace

Usage:
------
1. Create a video_urls.txt file with one URL per line
2. Set environment variables:
   export REPLICATE_API_TOKEN='your-token'
   export HUGGINGFACE_TOKEN='your-token'
3. Run: uv run process_videos.py [options]

Optional arguments:
  --input-file FILE  Text file containing video URLs (one per line)
  --epochs N        Number of training epochs (default: 2)
  --batch-size N    Training batch size (default: 8)
  --rank N         LoRA rank for training (default: 32)
  --learning-rate N Learning rate (default: 0.001)
  --frame-stride N  Frame stride for extraction (default: 10)

The script handles videos in the following way:
- Removes intros (first 13s) and outros (last 20s)
- Splits into scenes using content detection
- Keeps only scenes between 1-5 seconds
- Generates consistent model names and trigger words
- Creates training datasets with auto-captioning
"""

import argparse
import yt_dlp
from scenedetect import detect, ContentDetector, split_video_ffmpeg
from slugify import slugify
from moviepy.editor import VideoFileClip
import os
import re
import shutil
import json
from pathlib import Path
import replicate
import time
from typing import Optional
import logging
from tqdm import tqdm

# Set up logging with timestamps and levels
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Hunyuan training version
TRAINING_VERSION = "zsxkib/hunyuan-video-lora:04279caf015c30a635cabc4077b5bd82c5c706262eb61797a48db139444bcca9"

def extract_title_and_trigger(title: str, username: str = "deepfates", video_prefix: str = "The Beauty of") -> tuple[str, str]:
    """
    Extract model name and trigger word from a video title.
    
    This function:
    1. Removes 'The Beauty of' prefix if present
    2. Creates a model name in format: deepfates/hunyuan-{slugified-title}
    3. Generates a trigger word from the first 5 consonants
    
    The trigger word is used in training to associate the style. It should be:
    - Uppercase
    - Not a real word
    - Related to the content/style being trained
    
    Args:
        title: Original video title
        username: Replicate username for model naming (default: "deepfates")
        video_prefix: Prefix to remove from video titles (default: "The Beauty of")
        
    Returns:
        tuple: (model_name, trigger_word)
        
    Example:
        "The Beauty of Interstellar" -> 
        ("deepfates/hunyuan-interstellar", "NTRST")
    """
    # Remove prefix and clean title
    title = re.sub(f'^{re.escape(video_prefix)}\s+', '', title, flags=re.IGNORECASE).strip()
    logging.info(f"Cleaned title: {title}")
    
    # Create consistent model name format
    model_name = f"{username}/hunyuan-{slugify(title)}"
    
    # Generate trigger word from consonants
    consonants = ''.join(re.findall(r'[BCDFGHJKLMNPQRSTVWXYZ]', title.upper()))
    if not consonants:
        # Fallback if no consonants found
        trigger_word = slugify(title).upper()[:5]
    else:
        trigger_word = consonants[:5]
    
    logging.info(f"Generated model_name: {model_name}, trigger_word: {trigger_word}")
    return model_name, trigger_word

def process_clips_batch(video_path: Path, scenes: list, clips_dir: Path, 
                       trigger_word: str, end_time: float) -> list:
    """
    Process multiple video clips in parallel using a single VideoFileClip instance.
    
    This function:
    1. Takes detected scenes from the video
    2. Filters for scenes between 1-5 seconds
    3. Extracts each valid scene as a clip
    4. Generates metadata for training
    
    Args:
        video_path: Path to source video
        scenes: List of detected scene changes
        clips_dir: Output directory for clips
        trigger_word: Generated trigger word for captions
        end_time: Video end time (excluding outro)
        
    Returns:
        list: Metadata for all processed clips
    """
    clips_metadata = []
    valid_clips = 0
    
    with VideoFileClip(str(video_path)) as video:
        for i, scene in enumerate(scenes):
            start_time = scene[0].get_seconds()
            if start_time >= end_time:
                continue
                
            end_time_scene = min(scene[1].get_seconds(), end_time)
            duration = end_time_scene - start_time
            
            # Only process clips within our target duration
            if 1 <= duration <= 5:
                clip_path = clips_dir / f"clip_{i:04d}.mp4"
                try:
                    # End slightly before scene boundary for clean cuts
                    adjusted_end_time = end_time_scene - 0.04
                    clip = video.subclip(start_time, adjusted_end_time)
                    clip.write_videofile(
                        str(clip_path), 
                        codec='libx264',
                        audio_codec='aac',
                        logger=None,
                        threads=4  # Parallel processing
                    )
                    
                    # Generate metadata for this clip
                    clips_metadata.append({
                        "file_name": clip_path.name,
                        "text": f"A video in the style of {trigger_word}",
                        "id": f"{trigger_word}_{i:04d}"
                    })
                    valid_clips += 1
                except Exception as e:
                    logging.error(f"Error processing clip {i}: {e}")
    
    logging.info(f"Successfully processed {valid_clips} clips")
    return clips_metadata

def process_video(url: str, output_dir: Path) -> Optional[dict]:
    """
    Process a single video through the complete pipeline.
    
    Steps:
    1. Download video using yt-dlp
    2. Extract title and generate model names
    3. Detect scene changes
    4. Create clips from valid scenes
    5. Generate training dataset
    
    Args:
        url: Video URL (YouTube or direct)
        output_dir: Output directory for processed files
        
    Returns:
        dict: Metadata about the processed video and created model
              or None if processing failed
    """
    # Extract video ID for consistent file naming
    video_id = url.split('v=')[-1] if 'youtube.com' in url else url.split('/')[-1]
    logging.info(f"Processing video ID: {video_id}")
    
    # Check for existing processed dataset
    existing_zips = list(output_dir.glob(f"*{video_id}*_dataset.zip"))
    if existing_zips:
        logging.info(f"Found existing dataset: {existing_zips[0]}")
        metadata_path = existing_zips[0].with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                return json.load(f)
        return None

    # Set up temporary processing directory
    temp_dir = output_dir / "temp"
    clips_dir = temp_dir / "clips"
    os.makedirs(clips_dir, exist_ok=True)
    
    # Configure video downloader
    ydl_opts = {
        'format': 'bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4][height<=1080]',
        'quiet': True,
        'outtmpl': str(temp_dir / '%(title)s.%(ext)s'),
        'socket_timeout': 30,
        'retries': 3,
    }
    
    try:
        # Download and process video
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logging.info(f"Downloading: {url}")
            info = ydl.extract_info(url, download=True)
            title = info.get('title', '')
            logging.info(f"Downloaded: {title}")
            
            # Find downloaded video file
            video_files = list(temp_dir.glob("*.mp4"))
            if not video_files:
                raise FileNotFoundError("No MP4 file found after download")
            video_path = video_files[0]
            
            # Generate model information
            model_name, trigger_word = extract_title_and_trigger(
                title, 
                "deepfates",
                "The Beauty of"
            )
            if not trigger_word:
                raise ValueError(f"Failed to generate trigger word from: {title}")
            
            # Get video duration and detect scenes
            with VideoFileClip(str(video_path)) as video:
                end_time = video.duration - 20  # Exclude outro
            
            logging.info(f"Detecting scenes (duration: {end_time}s)")
            scenes = detect(str(video_path), ContentDetector())
            clips_metadata = process_clips_batch(video_path, scenes, clips_dir, 
                                              trigger_word, end_time)
            
            # Create and save dataset
            dataset_name = f"{slugify(title)}_{video_id}_dataset"
            zip_path = output_dir / f"{dataset_name}.zip"
            shutil.make_archive(str(zip_path.with_suffix('')), 'zip', clips_dir)
            
            # Store processing metadata
            metadata = {
                "model_name": model_name,
                "trigger_word": trigger_word,
                "clips": clips_metadata,
                "zip_path": str(zip_path),
                "dataset_name": dataset_name
            }
            
            metadata_path = zip_path.with_suffix('.json')
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            # Cleanup temporary files
            shutil.rmtree(temp_dir)
            
            return metadata
            
    except Exception as e:
        logging.error(f"Error processing {url}: {e}")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        return None

def check_training_status(training, model_name: str) -> bool:
    """
    Check and log the status of a Replicate training job.
    
    Args:
        training: Replicate training object
        model_name: Name of model being trained
        
    Returns:
        bool: True if training is complete (success or failure)
    """
    try:
        training.reload()
        if training.status == "succeeded":
            logging.info(f"Training completed successfully: {model_name}")
            return True
        elif training.status in ["failed", "canceled"]:
            logging.error(f"Training failed for {model_name}: {training.error}")
            return True
        else:
            logging.info(f"Training status for {model_name}: {training.status}")
            return False
    except Exception as e:
        logging.error(f"Error checking training status for {model_name}: {e}")
        return False

def create_or_get_model(model_name: str) -> Optional[dict]:
    """
    Check if a model exists on Replicate and create it if it doesn't.
    """
    owner, name = model_name.split('/')
    
    try:
        # Try to get existing model
        model = replicate.models.get(model_name)
        logging.info(f"Found existing model: {model_name}")
        return model
    except replicate.exceptions.ReplicateError as e:
        if "Model not found" in str(e):
            logging.info(f"Creating new model: {model_name}")
            try:
                return replicate.models.create(
                    owner=owner,
                    name=name,
                    visibility="public",
                    hardware="gpu-t4"
                )
            except Exception as e:
                logging.error(f"Failed to create model {model_name}: {e}")
                return None
        else:
            logging.error(f"Error getting model {model_name}: {e}")
            return None

def upload_and_start_training(zip_path, metadata, args, max_retries=3):
    """Helper function to handle upload and training start with retries"""
    base_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            # Ensure model exists
            model = create_or_get_model(metadata['model_name'])
            if not model:
                raise Exception(f"Could not create/get model {metadata['model_name']}")
            
            # Open file for upload
            with open(zip_path, "rb") as zip_file:
                training = replicate.trainings.create(
                    version=TRAINING_VERSION,
                    input={
                        "input_videos": zip_file,
                        "trigger_word": metadata["trigger_word"],
                        "autocaption": True,
                        "autocaption_prefix": f"A video in the style of {metadata['trigger_word']},",
                        "epochs": args.epochs,
                        "batch_size": args.batch_size,
                        "rank": args.rank,
                        "learning_rate": args.learning_rate,
                        "frame_stride": args.frame_stride,
                        "hf_repo_id": metadata["model_name"],
                        "hf_token": os.environ["HUGGINGFACE_TOKEN"],
                    },
                    destination=metadata["model_name"]
                )
                return training
                
        except Exception as e:
            delay = base_delay * (2 ** attempt)  # Exponential backoff
            if attempt < max_retries - 1:
                tqdm.write(f"Attempt {attempt + 1} failed for {metadata['model_name']}, retrying in {delay}s: {str(e)}")
                time.sleep(delay)
            else:
                tqdm.write(f"Failed to start training for {metadata['model_name']} after {max_retries} attempts: {str(e)}")
                return None

def process_videos():
    """
    Main execution function that:
    1. Parses command line arguments
    2. Validates environment
    3. Processes all videos
    4. Manages training jobs
    """
    # Set up command line arguments
    parser = argparse.ArgumentParser(
        description='Process videos and train Hunyuan models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('input_file', type=str,
                      help='Text file containing video URLs (one per line)')
    parser.add_argument('--username', type=str, default="deepfates",
                      help='Replicate username for model naming')
    parser.add_argument('--video-prefix', type=str, default="The Beauty of",
                      help='Video title prefix to remove')
    parser.add_argument('--epochs', type=int, default=2,
                      help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                      help='Training batch size')
    parser.add_argument('--rank', type=int, default=32,
                      help='LoRA rank for training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--frame-stride', type=int, default=10,
                      help='Frame stride for extraction')
    args = parser.parse_args()

    # Validate environment and input file
    required_vars = ["REPLICATE_API_TOKEN", "HUGGINGFACE_TOKEN"]
    missing_vars = [var for var in required_vars if var not in os.environ]
    if missing_vars:
        print(f"Missing environment variables: {', '.join(missing_vars)}")
        return

    if not os.path.exists(args.input_file):
        print(f"Input file not found: {args.input_file}")
        return
        
    # Create output directory
    output_dir = Path("processed_videos")
    output_dir.mkdir(exist_ok=True)
    
    # Read input URLs
    with open(args.input_file, "r") as f:
        urls = [line.strip() for line in f if line.strip()]
    
    # Track active training jobs
    active_trainings = []  # List of (training, metadata) tuples
    
    # Process each video
    for url in tqdm(urls, desc="Processing videos"):
        # Check status of existing trainings
        active_trainings = [(t, m) for t, m in active_trainings 
                           if not check_training_status(t, m['model_name'])]
        
        tqdm.write(f"Processing {url}...")
        metadata = process_video(url, output_dir)
        
        if metadata:
            tqdm.write(
                f"Created dataset for {metadata['model_name']} "
                f"with trigger word {metadata['trigger_word']}\n"
                f"Generated {len(metadata['clips'])} clips"
            )
            
            # Start training if dataset exists
            zip_path = Path(metadata['zip_path'])
            if zip_path.exists():
                tqdm.write(f"Starting Replicate training: {metadata['model_name']}")
                training = upload_and_start_training(zip_path, metadata, args)
                
                if training:
                    active_trainings.append((training, metadata))
                    tqdm.write(f"Training started for {metadata['model_name']}")
                else:
                    tqdm.write(f"Skipping {metadata['model_name']} due to upload failure")
            else:
                tqdm.write(f"Error: Could not find ZIP file at {zip_path}")
                
        tqdm.write("-" * 50)
    
    # Wait for remaining trainings to complete
    while active_trainings:
        time.sleep(30)  # Check every 30 seconds
        active_trainings = [(t, m) for t, m in active_trainings 
                           if not check_training_status(t, m['model_name'])]
        if active_trainings:
            tqdm.write(f"Waiting for {len(active_trainings)} trainings to complete...")

if __name__ == "__main__":
    process_videos()
