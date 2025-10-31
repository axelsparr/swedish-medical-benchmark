
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
MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
USE_QUANTIZATION = True
PIPELINE_PARAMS = {"do_sample": False}
USE_FEW_SHOT = True  # if true we sample the first of each class ie 4 samples
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
         + " Svara endast med en bokstav."
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
    pipe = pipeline("text-generation",
                     model=model,
                     model_kwargs=model_kwargs,
                     trust_remote_code=False)
    pipe.model.generation_config.do_sample = False
    return pipe


def build_messages(system_text: str, prompt: str):
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": prompt}
    ]
def build_messages_few_shot(system_text: str, prompt: str, example_prompts: list[str], example_answers: list[str]):
    assert len(example_prompts) == len(example_answers)
    message_history = [{"role": "system", "content": system_text}] 
    for i in range(len(example_prompts)):
        message_history.append({"role": "user", "content": example_prompts[i]})
        message_history.append({"role": "assistant", "content": example_answers[i]})
    message_history.append({"role": "user", "content": prompt})
    return message_history


def get_response(pipe_output) -> str:
    text = pipe_output[0]["generated_text"]
    return text[-1]["content"].strip()
def timestamp():
    return datetime.datetime.now().isoformat()
def few_shot_eval_split(items_kv, answers):
    #extract the first item from each class and return the rest as sample_items
    answers = list(map(str, answers))
    example_prompts, example_answers = [], []
   
    taken = {c: False for c in ["a","b","c","d"]}
    chosen_keys = []
    for i,(k, v) in enumerate(items_kv):
        y = answers[i]
        if y in taken and not taken[y]:
            chosen_keys.append(k)
            taken[y] = True
            example_answers.append(answers[i])
            example_prompts.append(v["Question"])
            items_kv.pop(i)
            answers.pop(i)
    print("Few shot examples chosen with keys:", chosen_keys)
    return items_kv, answers, example_prompts, example_answers


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
        benchmark_prompts = list(benchmark.data.items())
        
        # split in to few shot examples and evaluation data, like "train/test"
        if USE_FEW_SHOT:
            benchmark_prompts,benchmark_answer, example_prompts, example_answers = few_shot_eval_split(
                benchmark_prompts, ground_truths)

        for k, v in tqdm(benchmark_prompts, desc=f"Processing {benchmark.name}"):
            if isinstance(benchmark,benchmarks.PubMedQALSWE):
                question =  v["QUESTION"]
            elif isinstance(benchmark,benchmarks.EmergencyMedicine):
                question=v["Question"]
            if USE_FEW_SHOT:
                
                messages = build_messages_few_shot(
                    benchmark.prompt,
                    question,
                    example_prompts,
                    example_answers
                )
            else:
                messages = build_messages(
                    benchmark.prompt,
                    question
                )
            #print("Messages:", messages)
            out = pipeline(
                text_inputs=messages,                     
                max_new_tokens=max(benchmark.max_tokens, 32),
                do_sample=PIPELINE_PARAMS["do_sample"],
            )
            #print("current k:",k)
            llm_results.append(get_response(out)[0].lower())  # only get the first character
            predictions = benchmark.detect_answers(llm_results)
            ids.append(k)
            result[benchmark.name] = {
                "prompt": benchmark.prompt,
                "ground_truths": benchmark_answer,
                "predictions": predictions.tolist(),
                "ids": ids,
            }
            with open("./results.json", "w") as f:
                json.dump(result, f)

        assert len(benchmark_answer) == len(predictions)

        print(f"Accuracy {(predictions == benchmark_answer).sum() / len(benchmark_answer)}")
        print(f"Malformed answers {(predictions == 'missformat').sum()}")

    print(
        "Done! You can now run the evaluate_results.py script to get detailed performance metrics."
    )
