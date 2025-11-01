
import json
import datetime
from functools import lru_cache

import torch
import transformers
from transformers import BitsAndBytesConfig, pipeline
from tqdm import tqdm

import benchmark_set_up as benchmarks


# Configuration
# =============
MODEL_VARIANT = "4b-it"
MODEL_NAME = f"google/medgemma-{MODEL_VARIANT}"
USE_QUANTIZATION = False
PIPELINE_PARAMS = {"do_sample": False}

print("torch:", torch.__version__)
print("torch cuda runtime:", torch.version.cuda)
print("is_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))

PubMedQALSWE_SYSTEM_PROMPT = "Var vänlig och överväg varje aspekt av medicinska frågan nedan noggrant. Ta en stund, andas djupt, och när du känner dig redo, vänligen svara med endast ett av de fördefinierade svaren: 'ja', 'nej', eller 'kanske'. Det är viktigt att du begränsar ditt svar till dessa alternativ för att säkerställa tydlighet i kommunikationen."
GeneralPractioner_SYSTEM_PROMPT = "Du är en utmärkt läkare och skriver ett läkarprov. Var vänlig och överväg varje aspekt av medicinska frågan nedan noggrant. Ta en stund, andas djupt, och när du känner dig redo, vänligen svara med endast ett av alternativen."
SwedishDoctorsExam = "Du är en utmärkt läkare och skriver ett läkarprov. Var vänlig och överväg varje aspekt av medicinska frågan nedan noggrant. Ta en stund, andas djupt, och när du känner dig redo, vänligen svara med endast ett av alternativen. Svara med hela svarsalternativet. Utöver det är det viktigt att du inte inkluderar någon annan text i ditt svar."
# Make sure to uncomment the benchmarks you want to run
BENCHMARKS = [
    #benchmarks.PubMedQALSWE(
    #    prompt=PubMedQALSWE_SYSTEM_PROMPT
    #    + "\n\nFråga:\n{question} svara bara 'ja', 'nej' eller 'kanske'"
    #),
    # Uncomment to also run the GeneralPractioner benchmark
    benchmarks.EmergencyMedicine(
         prompt=GeneralPractioner_SYSTEM_PROMPT
         + "\n\nFråga:\n{question}\n\nSvara endast ett av alternativen. Svara endast med en bokstav. Tex. \"d\" som det ska vara alternativ d)"
    ),
    # benchmarks.SwedishDoctorsExam(prompt=SwedishDoctorsExam + "\n\nFråga:\n{question}\n\nSvara med endast ett av alternativen. Svara med hela svarsalternativet."),
]

# Functions
# =========
@lru_cache(maxsize=1)
def load_pipeline(model=MODEL_NAME):
    model_kwargs = dict(
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    if USE_QUANTIZATION:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

    # Use the correct task for medgemma-4b-it
    pipe = transformers.pipeline("image-text-to-text", model=model, model_kwargs=model_kwargs)
    pipe.model.generation_config.do_sample = False
    return pipe


def build_messages(system_text: str, prompt: str):
    return [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user",   "content": [{"type": "text", "text": prompt}]},
    ]

def get_response(pipe_output) -> str:
    # MedGemma pipeline returns: [{'generated_text': [ {...}, {'role':'assistant','content':'...'} ]}]
    last_turn = pipe_output[0]["generated_text"][-1]
    assert last_turn["role"] == "assistant", f"Unexpected output: {pipe_output}"
    return last_turn["content"].strip().lower()

def timestamp():
    return datetime.datetime.now().isoformat()


# Main
# ====
if __name__ == "__main__":
    pipeline = load_pipeline()
    result = {
        "llm_info": {
            "model": MODEL_NAME,
            "pipeline_params": PIPELINE_PARAMS,
            "use_quantization": USE_QUANTIZATION,
            "model_run": timestamp(),
        },
    }
    for benchmark in BENCHMARKS:
        llm_results = []
        ids = []
        ground_truths = benchmark.get_ground_truth()
        sample_items = list(benchmark.data.items())

        for k, v in tqdm(sample_items, desc=f"Processing {benchmark.name}"):
            if isinstance(benchmark,benchmarks.PubMedQALSWE):
                question =  v["QUESTION"]
            elif isinstance(benchmark,benchmarks.EmergencyMedicine):
                question=v["Question"]
            messages = build_messages(
                benchmark.prompt,
                benchmark.prompt.format(question=question),
            )
            out = pipeline(
                text=messages,                     
                max_new_tokens=max(benchmark.max_tokens, 32),
                do_sample=PIPELINE_PARAMS["do_sample"],
            )
            llm_results.append(get_response(out))
            predictions = benchmark.detect_answers(llm_results)
            ids.append(k)
            result[benchmark.name] = {
                "prompt": benchmark.prompt,
                "ground_truths": ground_truths.tolist(),
                "predictions": predictions.tolist(),
                "ids": ids,
            }
            with open("./results.json", "w") as f:
                json.dump(result, f)

        assert len(ground_truths) == len(predictions)

        print(f"Accuracy {(predictions == ground_truths).sum() / len(ground_truths)}")
        print(f"Malformed answers {(predictions == 'missformat').sum()}")

    print(
        "Done! You can now run the evaluate_results.py script to get detailed performance metrics."
    )
