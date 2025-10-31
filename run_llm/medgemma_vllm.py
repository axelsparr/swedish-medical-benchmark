from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
PubMedQALSWE_SYSTEM_PROMPT = "Var vänlig och överväg varje aspekt av medicinska frågan nedan noggrant. Ta en stund, andas djupt, och när du känner dig redo, vänligen svara med endast ett av de fördefinierade svaren: 'ja', 'nej', eller 'kanske'. Det är viktigt att du begränsar ditt svar till dessa alternativ för att säkerställa tydlighet i kommunikationen."

messages = [
    {"role": "system", "content": "Svara strikt i XML med reasoning och prediction."},
    {"role": "user", "content": "Fråga: ..."}
]
resp = client.chat.completions.create(
    model="google/medgemma-4b-it",
    messages=messages,
    extra_body={"guided_choice": ["ja", "nej", "kanske"]},  # llguidance constraint
)
print(resp.choices[0].message.content)
