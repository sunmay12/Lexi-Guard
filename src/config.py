import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)

GENERATOR_MODEL = "models/gemini-3.1-flash-lite"
JUDGE_MODEL = "models/gemini-3.1-flash-lite"

TEST_MODE = False
TEST_SIZE = 10

API_DELAY = 5
ARTICLE_DELAY = 10
CHECKPOINT_INTERVAL = 1

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
LOG_DIR = PROJECT_ROOT / "logs"

def get_gemini_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
    return api_key