import json
import datetime
from functools import lru_cache
import torch
from tqdm import tqdm

import transformers
from transformers import BitsAndBytesConfig
import benchmark_set_up as benchmarks

# guidance
import guidance
from guidance import models, system, user, assistant, select, gen

# ========== Config ==========
MODEL_VARIANT = "4b-it"
MODEL_NAME = f"google/medgemma-{MODEL_VARIANT}"
USE_QUANTIZATION = True
PIPELINE_PARAMS = {"do_sample": False}

PubMedQALSWE_SYSTEM_PROMPT = "Var vänlig och överväg varje aspekt av medicinska frågan nedan noggrant. Ta en stund, andas djupt, och när du känner dig redo, vänligen svara med endast ett av de fördefinierade svaren: 'ja', 'nej', eller 'kanske'. Det är viktigt att du begränsar ditt svar till dessa alternativ för att säkerställa tydlighet i kommunikationen."
BENCHMARKS = [
    benchmarks.PubMedQALSWE( 
        prompt=PubMedQALSWE_SYSTEM_PROMPT
        + "\n\nFråga:\n{question} svara bara 'ja', 'nej' eller 'kanske'"
    ),
    # Uncomment to also run the GeneralPractioner benchmark
    #benchmarks.EmergencyMedicine(
    #     prompt=
    #     + "\n\nFråga:\n{question}\nAlternativ:{options}\n\nSvara endast ett av alternativen."
    #),
    # benchmarks.SwedishDoctorsExam(prompt=SwedishDoctorsExam + "\n\nFråga:\n{question}\n\nSvara med endast ett av alternativen. Svara med hela svarsalternativet."),
]
# ========== Guidance model loader ==========
@lru_cache(maxsize=1)
def load_guidance_model():
    model_kwargs = dict(
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    if USE_QUANTIZATION:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

    lm = models.Transformers(MODEL_NAME, **model_kwargs)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return lm

# ========== Guidance program (fresh state per call) ==========
@guidance
def zero_shot_yesnomaybe(lm, question: str): 
    with user():
        lm += (
            f"Fråga:\n{question}\n"
            "Format exakt:\n<reasoning>...</reasoning><prediction>ja|nej|kanske</prediction>"
        )
    with assistant():
        lm += "<reasoning>" + gen("reasoning", max_tokens=160) + "</reasoning>"
        lm += "<prediction>" + select(["ja", "nej", "kanske"], name="pred") + "</prediction>"
    return lm


def timestamp():
    return datetime.datetime.now().isoformat()

# ========== Main ==========
if __name__ == "__main__":
    lm = load_guidance_model()
    result = {
        "llm_info": {
            "model": MODEL_NAME,
            "use_quantization": USE_QUANTIZATION,
            "pipeline_params": PIPELINE_PARAMS,
            "model_run": timestamp(),
        },
    }

    for benchmark in BENCHMARKS:
        llm_results, ids = [], []
        # ground truth and sample selection
        ground_truths = benchmark.get_ground_truth()[:10]
        sample_items = list(benchmark.data.items())[:10]

        with system():
            language_model += benchmark.prompt + "\nSvara i XML-format."

        for k, v in tqdm(sample_items, desc=f"Processing {benchmark.name}"):
            # system = Swedish instruction; user = RAW question only
            lm_temp = lm + zero_shot_yesnomaybe(question=v["QUESTION"])
            pred = lm_temp["string_choice"]
            print(pred)
            llm_results.append(pred)
            ids.append(k)

            predictions = benchmark.detect_answers(llm_results)
            result[benchmark.name] = {
                "prompt": benchmark.prompt,
                "ground_truths": ground_truths.tolist(),
                "predictions": predictions.tolist(),
                "ids": ids,
            }
            with open("./results.json", "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        assert len(ground_truths) == len(predictions)
        acc = (predictions == ground_truths).sum() / len(ground_truths)
        malformed = (predictions == "missformat").sum()
        print(f"Accuracy {acc:.3f}")
        print(f"Malformed answers {malformed}")
