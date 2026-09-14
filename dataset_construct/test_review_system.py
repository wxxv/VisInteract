"""
Test the components of the review system
"""

import json
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_files_exist():
    """Test if the necessary files exist"""
    logger.info("=" * 60)
    logger.info("Test 1: Check if the files exist")
    logger.info("=" * 60)
    
    files_to_check = {
        "Original data": Path("output/samples.json"),
        "Translation script": Path("prepare_review_data.py"),
        "Extract script": Path("extract_vega_specs.py"),
        "Generate script": Path("generate_review_page.py"),
        "Run script": Path("run_all.py"),
    }
    
    all_exist = True
    for name, path in files_to_check.items():
        if path.exists():
            logger.info(f"✅ {name}: {path}")
        else:
            logger.error(f"❌ {name}: {path} does not exist")
            all_exist = False
    
    return all_exist


def test_samples_json_structure():
    """Test the structure of samples.json"""
    logger.info("\n" + "=" * 60)
    logger.info("Test 2: Check the structure of samples.json")
    logger.info("=" * 60)
    
    samples_file = Path("output/samples.json")
    if not samples_file.exists():
        logger.error("❌ samples.json does not exist, skip test")
        return False
    
    try:
        with open(samples_file, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        
        logger.info(f"✅ Successfully loaded JSON file")
        logger.info(f"✅ Total number of samples: {len(samples)}")
        
        # Check the structure of the first sample in the samples.json file
        if len(samples) > 0:
            first_sample = samples[0]
            required_fields = ['sample_id', 'vis_question', 'altair_code']
            
            for field in required_fields:
                if field in first_sample:
                    logger.info(f"✅ Field '{field}' exists")
                else:
                    logger.error(f"❌ Field '{field}' does not exist")
                    return False
            
            logger.info(f"\nExample sample_id: {first_sample['sample_id']}")
            logger.info(f"vis_question length: {len(first_sample.get('vis_question', ''))} characters")
            logger.info(f"altair_code length: {len(first_sample.get('altair_code', ''))} characters")
        
        return True
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON parse error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return False


def test_translation_data():
    """Test if the translation data exists"""
    logger.info("\n" + "=" * 60)
    logger.info("Test 3: Check the translation data")
    logger.info("=" * 60)
    
    translation_file = Path("output/samples_with_translation.json")
    
    if not translation_file.exists():
        logger.warning("⚠️  Translation data does not exist (need to run prepare_review_data.py)")
        return False
    
    try:
        with open(translation_file, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        
        logger.info(f"✅ Translation data exists")
        logger.info(f"✅ Number of samples: {len(samples)}")
        
        # Check the translation fields
        translated_count = sum(1 for s in samples if s.get('vis_question_zh'))
        logger.info(f"✅ Translated samples: {translated_count}/{len(samples)}")
        
        if translated_count > 0:
            # Display a translation example
            for sample in samples:
                if sample.get('vis_question_zh'):
                    logger.info(f"\nTranslation example:")
                    logger.info(f"English: {sample['vis_question'][:100]}...")
                    logger.info(f"Chinese: {sample['vis_question_zh'][:100]}...")
                    break
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return False


def test_vega_specs():
    """Test if the Vega specs exist"""
    logger.info("\n" + "=" * 60)
    logger.info("Test 4: Check the Vega-Lite specs")
    logger.info("=" * 60)
    
    vega_file = Path("output/samples_with_vega_specs.json")
    
    if not vega_file.exists():
        logger.warning("⚠️  Vega specs do not exist (need to run extract_vega_specs.py)")
        return False
    
    try:
        with open(vega_file, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        
        logger.info(f"✅ Vega specs data exists")
        logger.info(f"✅ Number of samples: {len(samples)}")
        
        # Count the number of successful and failed extractions
        success_count = sum(1 for s in samples if s.get('vega_spec'))
        fail_count = sum(1 for s in samples if s.get('vega_error'))
        
        logger.info(f"✅ Successfully extracted: {success_count}/{len(samples)}")
        logger.info(f"⚠️  Extraction failed: {fail_count}/{len(samples)}")
        
        if success_count > 0:
            logger.info(f"\nExtraction success rate: {success_count/len(samples)*100:.1f}%")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return False


def test_html_page():
    """Test if the HTML page is generated"""
    logger.info("\n" + "=" * 60)
    logger.info("Test 5: Check the HTML review page")
    logger.info("=" * 60)
    
    html_file = Path("output/pages/review.html")
    
    if not html_file.exists():
        logger.warning("⚠️  HTML page does not exist (need to run generate_review_page.py)")
        return False
    
    try:
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        logger.info(f"✅ HTML page exists")
        logger.info(f"✅ File size: {len(html_content)/1024/1024:.2f} MB")
        
        # Check critical elements
        checks = {
            "Vega-Embed library": "vega-embed" in html_content,
            "Sample data embedding": "allSamples" in html_content,
            "Navigation buttons": "prevSample" in html_content and "nextSample" in html_content,
            "Mark button": "markSample" in html_content,
            "Filter function": "setFilter" in html_content,
        }
        
        for check_name, passed in checks.items():
            if passed:
                logger.info(f"✅ {check_name}")
            else:
                logger.error(f"❌ {check_name}")
        
        return all(checks.values())
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return False


def test_review_status():
    """Test the review status file"""
    logger.info("\n" + "=" * 60)
    logger.info("Test 6: Check the review status file")
    logger.info("=" * 60)
    
    status_file = Path("output/review_status.json")
    
    if not status_file.exists():
        logger.warning("⚠️  Review status file does not exist (will be created automatically on first run)")
        return True  # This is not an error
    
    try:
        with open(status_file, 'r', encoding='utf-8') as f:
            status = json.load(f)
        
        logger.info(f"✅ Review status file exists")
        logger.info(f"✅ Number of reviewed samples: {len(status)}")
        
        if len(status) > 0:
            pass_count = sum(1 for s in status.values() if s.get('status') == 'pass')
            fail_count = sum(1 for s in status.values() if s.get('status') == 'fail')
            
            logger.info(f"  - Pass: {pass_count}")
            logger.info(f"  - Fail: {fail_count}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║          Altair Visualization Review System - Test Suite  ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    tests = [
        ("File check", test_files_exist),
        ("Data structure", test_samples_json_structure),
        ("Translation data", test_translation_data),
        ("Vega specs", test_vega_specs),
        ("HTML page", test_html_page),
        ("Review status", test_review_status),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test '{test_name}' failed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test summary")
    logger.info("=" * 60)
    
    for test_name, result in results:
        status = "✅ Pass" if result else "❌ Fail/Warning"
        logger.info(f"{status} - {test_name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    logger.info(f"\nPass rate: {passed}/{total} ({passed/total*100:.1f}%)")
    
    # Give suggestions
    logger.info("\n" + "=" * 60)
    logger.info("Next steps")
    logger.info("=" * 60)
    
    if not Path("output/samples_with_translation.json").exists():
        logger.info("1️⃣ Run: python prepare_review_data.py (Translation data)")
    
    if not Path("output/samples_with_vega_specs.json").exists():
        logger.info("2️⃣ Run: python extract_vega_specs.py (Extract charts)")
    
    if not Path("output/pages/review.html").exists():
        logger.info("3️⃣ Run: python generate_review_page.py (Generate page)")
    
    if all([
        Path("output/samples_with_translation.json").exists(),
        Path("output/samples_with_vega_specs.json").exists(),
        Path("output/pages/review.html").exists()
    ]):
        logger.info("✅ System is ready! Please open output/pages/review.html in your browser")
    else:
        logger.info("\n💡 Or run: python run_all.py (One-click to complete all steps)")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
