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
USE_QUANTIZATION = True
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
    benchmarks.PubMedQALSWE(
        prompt=PubMedQALSWE_SYSTEM_PROMPT
        + "\n\nFråga:\n{question} svara bara 'ja', 'nej' eller 'kanske'"
    ),
    # Uncomment to also run the GeneralPractioner benchmark
    # benchmarks.GeneralPractioner(
    #     prompt=GeneralPractioner_SYSTEM_PROMPT
    #     + "\n\nFråga:\n{question}\nAlternativ:{options}\n\nSvara endast ett av alternativen."
    #),
    # benchmarks.SwedishDoctorsExam(prompt=SwedishDoctorsExam + "\n\nFråga:\n{question}\n\nSvara med endast ett av alternativen. Svara med hela svarsalternativet."),
]

# Functions
# =========
@lru_cache(maxsize=1)
def load_pipeline(
    model=MODEL_NAME,
) -> transformers.pipelines.text_generation.TextGenerationPipeline:
    model_kwargs = dict(
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if USE_QUANTIZATION:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

    return transformers.pipeline(
        "text-generation", model=model, model_kwargs=model_kwargs
    )


def get_response(messages: list[str]) -> str:
    response = messages[0]["generated_text"][-1]
    assert response["role"] == "assistant"
    return response["content"].lower()


def fmt_message(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


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

        for k, v in tqdm(benchmark.data.items(), desc=f"Processing {benchmark.name}"):
            messages = [
                fmt_message("user", benchmark.prompt.format(question=v["QUESTION"]))
            ]
            out = pipeline(
                messages,
                max_new_tokens=benchmark.max_tokens,
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
