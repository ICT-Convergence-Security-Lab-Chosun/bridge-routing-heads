from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Per-language instruction strings (kept language-specific to prime the model)
# ---------------------------------------------------------------------------

INSTRUCTIONS: dict[str, str] = {
    "en": "Answer briefly with only the answer.",
    "ko": "정답만 짧게 답하시오.",
    "zh": "只简短回答答案。",
    "ja": "答えだけを短く答えなさい。",
    "es": "Responde brevemente solo con la respuesta.",
    "id": "Jawab singkat hanya dengan jawabannya.",
    "vi": "Chỉ trả lời ngắn gọn bằng đáp án.",
    "hi": "केवल उत्तर संक्षेप में दें।"
}

# ---------------------------------------------------------------------------
# Question-word lookup: relation → "who" (person answer) or "what" (thing/place)
# [Answer] token is intentionally kept identical across all languages as the
# readout anchor for mechanistic interpretability analysis.
# ---------------------------------------------------------------------------

_PERSON_RELATIONS: frozenset[str] = frozenset({
    "spouse",
    "father",
    "mother",
    "composer",
    "performer",
    "director",
    "producer",
    "screenwriter",
    "author",
    "publisher",
    "employer",
    "head of state",
    "founded by",
    "chief executive officer",
    "chief operating officer",
})

_Q_WORDS: dict[str, dict[str, str]] = {
    "en": {"who": "who",   "what": "what"},
    "ko": {"who": "누구",  "what": "무엇"},
    "zh": {"who": "谁",    "what": "什么"},
    "ja": {"who": "誰",    "what": "何"},
    "es": {"who": "quién", "what": "cuál"},
    "id": {"who": "Siapa", "what": "Apa"},
    "vi": {"who": "ai",    "what": "gì"},
    "hi": {"who": "कौन",   "what": "क्या"},
}


def _q(relation: str, lang: str) -> str:
    kind = "who" if relation in _PERSON_RELATIONS else "what"
    return _Q_WORDS[lang][kind]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _has_korean_final_consonant(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    char = stripped[-1]
    code = ord(char)
    if 0xAC00 <= code <= 0xD7A3:
        return (code - 0xAC00) % 28 != 0
    return False


def get_topic_particle(text: str) -> str:
    return "은" if _has_korean_final_consonant(text) else "는"


def _capitalize_first(text: str) -> str:
    return text[:1].upper() + text[1:]


def _contract_phrase(text: str, lang: str) -> str:
    if lang == "es":
        return (
            text.replace(" de el ", " del ")
            .replace(" De el ", " Del ")
        )
    return text


def _relation_template(templates: dict[str, Any], relation: str, lang: str) -> dict[str, str]:
    if relation not in templates:
        raise KeyError(f"Missing relation template for relation={relation!r}")
    if lang not in templates[relation]:
        raise KeyError(f"Missing {lang!r} template for relation={relation!r}")
    return templates[relation][lang]


def wrap_prompt(question: str, lang: str) -> str:
    """Wrap a raw question in the [Instruction]/[Question]/[Answer] format.

    Called at inference time (step2), not at data-build time (step1), so that
    the stored JSON contains only the question text and the format can be
    changed without re-running step1.

    [Answer] is intentionally an ASCII-only English token across all languages
    so that the readout position is language-invariant for head analysis.
    """
    # return f"Question:{question} Answer:"
    return f"Instruction:{INSTRUCTIONS[lang]} Question:{question} Answer:"

# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_prompts_for_lang(
    r1: str,
    r2: str,
    e1_label: str,
    e2_label: str,
    templates: dict[str, Any],
    lang: str,
    e1_prompt_label: str | None = None,
    e2_prompt_label: str | None = None,
) -> dict[str, str]:
    e1_for_prompt = e1_prompt_label or e1_label
    e2_for_prompt = e2_prompt_label or e2_label
    r1_t = _relation_template(templates, r1, lang)
    r2_t = _relation_template(templates, r2, lang)
    q1 = _q(r1, lang)
    q2 = _q(r2, lang)

    if lang == "en":
        r1_embed = r1_t["embed"]
        r2_embed = r2_t["embed"]
        q1w, q2w = q1.capitalize(), q2.capitalize()
        return {
            "two_hop":       f"{q2w} is {r2_embed} {r1_embed} {e1_for_prompt}?",
            "first_hop":     f"{q1w} is {r1_embed} {e1_for_prompt}?",
            "second_hop":    f"{q2w} is {r2_embed} {e2_for_prompt}?",
            "shortcut_no_e1": f"{q2w} is {r2_embed} {r1_embed}?",
            "shortcut_no_r1": f"{q2w} is {r2_embed} {e1_for_prompt}?",
        }

    if lang == "ko":
        r1_noun = r1_t["r_noun"]
        r1_particle = r1_t.get("particle", get_topic_particle(r1_noun))
        r2_noun = r2_t["r_noun"]
        r2_particle = r2_t.get("particle", get_topic_particle(r2_noun))
        return {
            "two_hop":       f"{e1_for_prompt}의 {r1_noun}의 {r2_noun}{r2_particle} {q2}입니까?",
            "first_hop":     f"{e1_for_prompt}의 {r1_noun}{r1_particle} {q1}입니까?",
            "second_hop":    f"{e2_for_prompt}의 {r2_noun}{r2_particle} {q2}입니까?",
            # shortcut_no_e1 intentionally drops e1 (per HoppingTooLate filter 1)
            "shortcut_no_e1": f"의 {r1_noun}의 {r2_noun}{r2_particle} {q2}입니까?",
            "shortcut_no_r1": f"{e1_for_prompt}의 {r2_noun}{r2_particle} {q2}입니까?",
        }

    if lang == "zh":
        r1_noun = r1_t["r_noun"]
        r2_noun = r2_t["r_noun"]
        return {
            "two_hop":       f"{e1_for_prompt}的{r1_noun}的{r2_noun}是{q2}？",
            "first_hop":     f"{e1_for_prompt}的{r1_noun}是{q1}？",
            "second_hop":    f"{e2_for_prompt}的{r2_noun}是{q2}？",
            "shortcut_no_e1": f"的{r1_noun}的{r2_noun}是{q2}？",
            "shortcut_no_r1": f"{e1_for_prompt}的{r2_noun}是{q2}？",
        }

    if lang == "ja":
        r1_noun = r1_t["r_noun"]
        r2_noun = r2_t["r_noun"]
        return {
            "two_hop":       f"{e1_for_prompt}の{r1_noun}の{r2_noun}は{q2}ですか？",
            "first_hop":     f"{e1_for_prompt}の{r1_noun}は{q1}ですか？",
            "second_hop":    f"{e2_for_prompt}の{r2_noun}は{q2}ですか？",
            "shortcut_no_e1": f"の{r1_noun}の{r2_noun}は{q2}ですか？",
            "shortcut_no_r1": f"{e1_for_prompt}の{r2_noun}は{q2}ですか？",
        }

    if lang == "vi":
        r1_noun = r1_t["r_noun"]
        r2_noun = r2_t["r_noun"]
        return {
            "two_hop":       f"{_capitalize_first(r2_noun)} của {r1_noun} của {e1_for_prompt} là {q2}?",
            "first_hop":     f"{_capitalize_first(r1_noun)} của {e1_for_prompt} là {q1}?",
            "second_hop":    f"{_capitalize_first(r2_noun)} của {e2_for_prompt} là {q2}?",
            "shortcut_no_e1": f"{_capitalize_first(r2_noun)} của {r1_noun} của là {q2}?",
            "shortcut_no_r1": f"{_capitalize_first(r2_noun)} của {e1_for_prompt} là {q2}?",
        }

    if lang == "es":
        r1_obj_e1    = _contract_phrase(r1_t["object"].format(e=e1_for_prompt), "es")
        r1_obj_empty = _contract_phrase(r1_t["object"].format(e=""), "es")
        q1w = "Quién" if q1 == "quién" else "Cuál"
        q2w = "Quién" if q2 == "quién" else "Cuál"
        return {
            "two_hop":       f"¿{q2w} es {_contract_phrase(r2_t['object'].format(e=r1_obj_e1), 'es')}?",
            "first_hop":     f"¿{q1w} es {r1_obj_e1}?",
            "second_hop":    f"¿{q2w} es {_contract_phrase(r2_t['object'].format(e=e2_for_prompt), 'es')}?",
            "shortcut_no_e1": f"¿{q2w} es {_contract_phrase(r2_t['object'].format(e=r1_obj_empty), 'es')}?",
            "shortcut_no_r1": f"¿{q2w} es {_contract_phrase(r2_t['object'].format(e=e1_for_prompt), 'es')}?",
        }

    if lang == "id":
        r1_obj_e1    = r1_t["object"].format(e=e1_for_prompt)
        r1_obj_empty = r1_t["object"].format(e="")
        return {
            "two_hop":       f"{q2} {r2_t['object'].format(e=r1_obj_e1)}?",
            "first_hop":     f"{q1} {r1_obj_e1}?",
            "second_hop":    f"{q2} {r2_t['object'].format(e=e2_for_prompt)}?",
            "shortcut_no_e1": f"{q2} {r2_t['object'].format(e=r1_obj_empty)}?",
            "shortcut_no_r1": f"{q2} {r2_t['object'].format(e=e1_for_prompt)}?",
        }

    if lang == "hi":
        r1_noun = r1_t["r_noun"]
        r2_noun = r2_t["r_noun"]
        return {
            "two_hop":       f"{e1_for_prompt} के {r1_noun} का {r2_noun} {q2} है?",
            "first_hop":     f"{e1_for_prompt} का {r1_noun} {q1} है?",
            "second_hop":    f"{e2_for_prompt} का {r2_noun} {q2} है?",
            "shortcut_no_e1": f"के {r1_noun} का {r2_noun} {q2} है?",
            "shortcut_no_r1": f"{e1_for_prompt} का {r2_noun} {q2} है?",
        }

    raise ValueError(f"Unsupported language: {lang}")


def _as_token_list(tokens: Any) -> list[int]:
    if hasattr(tokens, "detach"):
        tokens = tokens.detach().cpu().tolist()
    if tokens and isinstance(tokens[0], list):
        return list(tokens[0])
    return list(tokens)


def _rfind_subsequence(sequence: list[int], pattern: list[int]) -> int | None:
    if not pattern or len(pattern) > len(sequence):
        return None
    for start in range(len(sequence) - len(pattern), -1, -1):
        if sequence[start : start + len(pattern)] == pattern:
            return start
    return None


def find_t1_position(tokenizer: Any, tokens: Any, e1_label_en: str) -> tuple[int | None, bool]:
    token_ids = _as_token_list(tokens)
    candidates = [
        tokenizer.encode(e1_label_en, add_special_tokens=False),
        tokenizer.encode(" " + e1_label_en, add_special_tokens=False),
    ]
    for pattern in candidates:
        start = _rfind_subsequence(token_ids, pattern)
        if start is not None:
            return start + len(pattern) - 1, True
    return None, False


def find_t2_position(tokens: Any) -> int:
    token_ids = _as_token_list(tokens)
    return len(token_ids) - 1
