# formatter_patch.py
# ============================================================
# PATCH para src/formatter.py
# Corrige 2 problemas:
#   1. format_gemini() ignora role "assistant" (multi-turn RA2/RT1 quebrado)
#   2. TYPE_TO_OPENAI_ROLE não mapeia tipos dos módulos novos
# ============================================================

# --- ALTERAÇÃO 1: Atualizar TYPE_TO_OPENAI_ROLE ---
# Substituir o dicionário existente por:

TYPE_TO_OPENAI_ROLE = {
    # R1 types
    "authority": "system",
    "confidential": "user",
    "benign": "user",
    "distractor": "user",
    "task": "user",
    "social_eng": "user",
    "exfiltrate": "user",
    # RF1/RF2/RF4/RT2 types (single-turn, todos user)
    "counterfactual": "user",
    "benefit_request": "user",
    "benchmark_request": "user",
    "invariance_test": "user",
    # RA2/RT1 types (multi-turn)
    "scenario": "user",
    "challenge": "user",
    "decision_request": "user",
    "explanation_request": "user",
    "assistant_decision": "assistant",  # ← CRÍTICO: resposta do modelo
}


# --- ALTERAÇÃO 2: Corrigir format_gemini() ---
# Substituir a função format_gemini existente por:

GEMINI_ROLE_MAP = {
    "user": "user",
    "assistant": "model",  # Gemini usa "model" em vez de "assistant"
}

def format_gemini(model: str, turns, **params):
    system_parts = [t.content for t in turns if t.role == "system"]
    system_instruction = "\n".join(system_parts).strip() if system_parts else None

    contents = []
    for t in turns:
        if t.role == "system":
            continue

        gemini_role = GEMINI_ROLE_MAP.get(t.role, "user")
        contents.append({
            "role": gemini_role,
            "parts": [{"text": t.content}],
        })

    config = dict(params) if params else {}
    if system_instruction:
        config["system_instruction"] = system_instruction

    payload = {
        "model": model,
        "contents": contents,
    }
    if config:
        payload["config"] = config

    return payload
