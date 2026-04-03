import re

NUM_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?")

def normalize_answer(s: str) -> str:
    s = s.strip()
    s = s.replace("$", "").replace(",", "")
    s = re.sub(r"(?i)^(?:the\s+)?(?:final\s+)?answer\s*(?:is|=|:)\s*", "", s)
    s = s.strip().rstrip(".")
    return s

# TODO: Correctly implement the answer extractor so that fair comparison is ensured.
# Maybe just follow the 
def extract_pred_answer(pred_text: str):
    zones = [pred_text.rsplit("</think>", 1)[-1], pred_text] if "</think>" in pred_text else [pred_text]

    for zone in zones:
        # 1. #### answer
        m = re.search(r"####\s*(.+?)\s*$", zone, flags=re.S)
        if m:
            return normalize_answer(m.group(1))

        # 2. boxed answer
        boxed = extract_last_boxed(zone)
        if boxed:
            return boxed

        # 3. explicit answer line
        for line in reversed(zone.splitlines()):
            if re.search(r"(?i)(final answer|answer is|therefore|thus)", line):
                nums = NUM_RE.findall(line)
                if nums:
                    return normalize_answer(nums[-1])

        # 4. fallback: last number
        nums = NUM_RE.findall(zone)
        if nums:
            return normalize_answer(nums[-1])

    return None