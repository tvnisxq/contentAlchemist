import json
import asyncio
# from sklearn.feature_extraction.text import TfidfVectorizer # Optional: for advanced similarity
# from sklearn.metrics.pairwise import cosine_similarity # Optional: for advanced similarity
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load Env for API Key
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# Basic configuration
if API_KEY:
    genai.configure(api_key=API_KEY)

class EvalHarness:
    def __init__(self, dataset_path="test_cases.jsonl"):
        self.dataset_path = dataset_path
        self.results = []
        # Configure Gemini for "LLM-as-a-Judge"
        # We use flash for speed/cost in evaluation
        self.judge_model = genai.GenerativeModel('gemini-1.5-flash')

    def load_test_cases(self):
        # FIX: Ensure UTF-8 encoding when reading test cases
        with open(self.dataset_path, 'r', encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    async def evaluate_consistency(self, generated_text, original_style_sample):
        """
        Uses LLM to judge if the generated voice matches the creator's style.
        """
        prompt = f"""
        You are a literary critic. 
        Sample A (Creator's Style): "{original_style_sample}"
        Sample B (AI Generated): "{generated_text}"
        
        Rate Sample B on how well it mimics the tone/voice of Sample A on a scale of 1-10.
        Return ONLY the number.
        """
        try:
            response = await self.judge_model.generate_content_async(prompt)
            return int(response.text.strip())
        except:
            return 5 # Fallback score

    def run_evals(self):
        print("🧪 Starting Automated Evaluation Harness...")
        try:
            test_cases = self.load_test_cases()
        except FileNotFoundError:
            print(f"❌ Error: {self.dataset_path} not found. Run create_dummy_data() first.")
            return

        for case in test_cases:
            print(f"Running Case: {case['id']}...")
            
            # --- MOCK AGENT EXECUTION ---
            # In a real scenario, you would import 'ContentAlchemist' here and run it.
            # For this harness demo, we simulate output based on the test case goal.
            if "blog" in case['id']:
                actual_output = "This is a comprehensive guide about Python Asyncio. It uses async and await."
            else:
                actual_output = "Hustle culture is about growth and scale. Let's go!" 
            
            # --- METRIC 1: Keyword Presence (Deterministic) ---
            # Check if expected keywords appear in the output
            has_keywords = all(k in actual_output.lower() for k in case['expected_keywords'])
            
            # --- METRIC 2: Length Check (Deterministic) ---
            # Check if output meets minimum length
            valid_length = len(actual_output.split()) >= case['min_words']
            
            result = {
                "id": case['id'],
                "pass": has_keywords and valid_length,
                "metrics": {
                    "keywords_found": has_keywords,
                    "valid_length": valid_length
                }
            }
            self.results.append(result)

    def generate_report(self):
        passed = sum(1 for r in self.results if r['pass'])
        total = len(self.results)
        
        report = f"""
        # Evaluation Report
        **Pass Rate:** {passed}/{total} ({(passed/total)*100}%)
        
        ## Details
        """
        for r in self.results:
            # Emojis like ✅ caused the crash on Windows without UTF-8
            status_icon = '✅ PASS' if r['pass'] else '❌ FAIL'
            report += f"- Case {r['id']}: {status_icon} (Metrics: {r['metrics']})\n"
            
        # FIX: Force UTF-8 encoding to support emojis on Windows
        with open("EVAL_REPORT.md", "w", encoding="utf-8") as f:
            f.write(report)
        print("✅ Report generated: EVAL_REPORT.md")

# Create a dummy dataset for the harness to run if it doesn't exist
def create_dummy_data():
    data = [
        {
            "id": "test_001_tech_blog",
            "input": "Video about Python Asyncio",
            "expected_keywords": ["python", "async", "await"],
            "min_words": 5, 
            "style_sample": "I love writing code that runs fast."
        },
        {
            "id": "test_002_linkedin",
            "input": "Video about Startup Growth",
            "expected_keywords": ["growth", "scale"],
            "min_words": 5,
            "style_sample": "Hustle is the key to success."
        }
    ]
    # FIX: Use UTF-8 for writing data
    with open("test_cases.jsonl", "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    # 1. Setup Data
    create_dummy_data() 
    
    # 2. Run Harness
    harness = EvalHarness()
    harness.run_evals()
    harness.generate_report()