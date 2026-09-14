"""
Altair Visualization Review System - Data preparation script
Function: Batch translate the vis_question field in samples.json to Chinese
"""

import json
import time
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('output/translation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# GLM-4 API configuration
API_KEY = "530ea2ea25014472bbb659dd07bdd32d.TB9Ssh7C1SnAt5HO"
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
MODEL = "glm-4-flash"

# Concurrent configuration
MAX_WORKERS = 50  # Concurrent number

# File paths
INPUT_FILE = Path("output/samples.json")
OUTPUT_FILE = Path("output/samples_with_translation.json")
PROGRESS_FILE = Path("output/translation_progress.json")

# Thread lock, for protecting shared resources
progress_lock = threading.Lock()
samples_lock = threading.Lock()


def init_openai_client():
    """Initialize OpenAI client"""
    return OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )


def translate_text(client, text, max_retries=3):
    """
    Use GLM-4 to translate text
    
    Args:
        client: OpenAI client
        text: English text to be translated
        max_retries: Maximum number of retries
    
    Returns:
        Translated Chinese text
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个专业的翻译助手。请将英文准确翻译成中文，保持原文的专业性和准确性。"
                    },
                    {
                        "role": "user",
                        "content": f"请将以下文本翻译成中文：\n\n{text}"
                    }
                ],
                temperature=0.3
            )
            
            translation = response.choices[0].message.content.strip()
            return translation
            
        except Exception as e:
            logger.warning(f"Translation failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"Translation finally failed: {text[:100]}...")
                return f"[Translation failed] {text}"


def load_progress():
    """Load translation progress"""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"translated_ids": [], "last_index": 0}


def save_progress(progress):
    """Save translation progress"""
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def save_intermediate_results(samples):
    """Save intermediate results"""
    temp_file = OUTPUT_FILE.parent / f"{OUTPUT_FILE.stem}_temp.json"
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    logger.info(f"Intermediate results saved to: {temp_file}")


def translate_sample(sample, client, progress, translated_ids):
    """
    Translate a single sample (for concurrent processing)
    
    Args:
        sample: Sample data (contains index information)
        client: OpenAI client
        progress: Progress dictionary
        translated_ids: Translated ID set
    
    Returns:
        (sample_id, vis_question_translation, vis_question_clear_translation, success, sample_index)
    """
    sample_data = sample['data']
    sample_index = sample['index']
    sample_id = sample_data["sample_id"]
    
    # Check if already translated
    with progress_lock:
        if sample_id in translated_ids:
            return (sample_id, None, None, True, sample_index)
    
    vis_question = sample_data.get("vis_question", "")
    vis_question_clear = sample_data.get("vis_question_clear", "")
    
    # Translate vis_question
    translation_vq = ""
    if vis_question:
        translation_vq = translate_text(client, vis_question)
    
    # Translate vis_question_clear
    translation_vqc = ""
    if vis_question_clear:
        translation_vqc = translate_text(client, vis_question_clear)
    
    success = not (translation_vq.startswith("[Translation failed]") or translation_vqc.startswith("[Translation failed]"))
    
    return (sample_id, translation_vq, translation_vqc, success, sample_index)


def main():
    """Main function"""
    logger.info("=" * 60)
    logger.info("Start preparing review data (concurrent mode)")
    logger.info("=" * 60)
    
    # Check input file
    if not INPUT_FILE.exists():
        logger.error(f"Input file does not exist: {INPUT_FILE}")
        return
    
    # Load sample data
    logger.info(f"Loading sample data: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    logger.info(f"Total number of samples: {len(samples)}")
    logger.info(f"Concurrent number: {MAX_WORKERS}")
    
    # Initialize OpenAI client (one per thread)
    logger.info("Initializing GLM-4 API client...")
    
    # Load progress
    progress = load_progress()
    translated_ids = set(progress.get("translated_ids", []))
    
    if translated_ids:
        logger.info(f"Continue from last progress (translated: {len(translated_ids)} samples)")
    
    # Prepare list of samples to translate (contains index)
    samples_to_translate = []
    for i, sample in enumerate(samples):
        if sample["sample_id"] not in translated_ids:
            samples_to_translate.append({
                'index': i,
                'data': sample
            })
    
    total_to_translate = len(samples_to_translate)
    logger.info(f"Number of samples to translate: {total_to_translate}")
    
    if total_to_translate == 0:
        logger.info("All samples have been translated!")
        return
    
    # Concurrent translation
    logger.info(f"Start concurrent translation ({MAX_WORKERS} worker threads)...")
    translation_count = 0
    success_count = 0
    fail_count = 0
    save_interval = 100  # Save every 100 samples
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create independent client for each thread
        def create_client():
            return init_openai_client()
        
        # Submit all tasks
        future_to_sample = {}
        for sample_item in samples_to_translate:
            client = create_client()  # Each task uses an independent client
            future = executor.submit(
                translate_sample,
                sample_item,
                client,
                progress,
                translated_ids
            )
            future_to_sample[future] = sample_item
        
        # Use tqdm to display progress
        with tqdm(total=total_to_translate, desc="Translation progress") as pbar:
            for future in as_completed(future_to_sample):
                try:
                    sample_id, translation_vq, translation_vqc, success, sample_index = future.result()
                    
                    if translation_vq is not None:  # Not translated samples
                        # Update sample data
                        with samples_lock:
                            samples[sample_index]["vis_question_zh"] = translation_vq
                            samples[sample_index]["vis_question_clear_zh"] = translation_vqc
                        
                        # Update progress
                        with progress_lock:
                            translated_ids.add(sample_id)
                            translation_count += 1
                            
                            if success:
                                success_count += 1
                            else:
                                fail_count += 1
                            
                            # Save progress periodically
                            if translation_count % save_interval == 0:
                                progress["translated_ids"] = list(translated_ids)
                                progress["last_index"] = sample_index + 1
                                save_progress(progress)
                                save_intermediate_results(samples)
                                logger.info(f"Translated {translation_count}/{total_to_translate} samples (success: {success_count}, fail: {fail_count})")
                    
                    pbar.update(1)
                    
                except Exception as e:
                    logger.error(f"Error processing sample: {str(e)}")
                    pbar.update(1)
    
    # Save final results
    logger.info(f"Save final results to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    # Save final progress
    progress["translated_ids"] = list(translated_ids)
    progress["last_index"] = len(samples)
    save_progress(progress)
    
    # Clean up temporary files
    temp_file = OUTPUT_FILE.parent / f"{OUTPUT_FILE.stem}_temp.json"
    if temp_file.exists():
        temp_file.unlink()
        logger.info("Temporary files cleaned up")
    
    logger.info("=" * 60)
    logger.info(f"Translation completed!")
    logger.info(f"- Total number of samples: {len(samples)}")
    logger.info(f"- Number of samples translated: {translation_count}")
    logger.info(f"- Translation successful: {success_count}")
    logger.info(f"- Translation failed: {fail_count}")
    logger.info(f"- Output file: {OUTPUT_FILE}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
