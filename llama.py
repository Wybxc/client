import json
import requests


def get_response(
    prompt: str,
    *,
    endpoint: str,
    n_predict: int,
    temperature: float,
    stop: list[str],
    repeat_last_n: int,
    repeat_penalty: float,
    top_k: int,
    top_p: float,
    min_p: float,
    tfs_z: float,
    typical_p: float,
    presence_penalty: float,
    frequency_penalty: float,
    mirostat: int,
    mirostat_tau: float,
    mirostat_eta: float,
    grammar: str,
    n_probs: int,
    cache_prompt: bool,
    slot_id: int,
):
    body = {
        "stream": True,
        "n_predict": n_predict,
        "temperature": temperature,
        "stop": stop,
        "repeat_last_n": repeat_last_n,
        "repeat_penalty": repeat_penalty,
        "top_k": top_k,
        "top_p": top_p,
        "min_p": min_p,
        "tfs_z": tfs_z,
        "typical_p": typical_p,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "mirostat": mirostat,
        "mirostat_tau": mirostat_tau,
        "mirostat_eta": mirostat_eta,
        "grammar": grammar,
        "n_probs": n_probs,
        "cache_prompt": cache_prompt,
        "slot_id": slot_id,
        "prompt": prompt,
    }
    response = requests.post(endpoint, json=body, stream=True)

    for line in response.iter_lines(decode_unicode=False):
        line = line.decode("utf-8")
        if line.startswith("data: "):
            data = json.loads(line[6:])
            if data.get("stop", True) == False:
                content = data["content"]
                yield content
