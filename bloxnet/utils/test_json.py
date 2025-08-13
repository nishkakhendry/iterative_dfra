import json

json_str = """[
    {
        "name": "base",
        "shape": "cuboid",
        "dimensions": {"x": 90, "y": 70, "z": 20},
        "color": [0.5, 0.3, 0.2, 1],
        "position": {"x": 0, "y": 0},
        "yaw": 0
    },
    {
        "name": "backrest",
        "shape": "cuboid",
        "dimensions": {"x": 90, "y": 25, "z": 25},
        "color": [0.7, 0.5, 0.4, 1],
        "position": {"x": 0, "y": 20},  // Adjusted to be above the base
        "yaw": 0
    },
    {
        "name": "armrest_left",
        "shape": "cuboid",
        "dimensions": {"x": 25, "y": 25, "z": 50},
        "color": [0.6, 0.4, 0.3, 1],
        "position": {"x": -47.5, "y": 0},
        "yaw": 0
    },
    {
        "name": "armrest_right",
        "shape": "cuboid",
        "dimensions": {"x": 25, "y": 25, "z": 50},
        "color": [0.6, 0.4, 0.3, 1],
        "position": {"x": 47.5, "y": 0},
        "yaw": 0
    }
]"""

import re

def extract_code_from_response(
    gpt_output: str,
    lang: str = "python",
    last_block_only: bool = False
) -> str:
    """
    Extract (and clean) code blocks of the given language from a GPT-style response.
    Strips out any // comments so that JSON or other strict formats will parse.
    """
    # 1) Grab all fenced code blocks of the form ```lang\n ... ```
    pattern = rf"```{lang}\s*\n(.*?)```"
    code_blocks = re.findall(pattern, gpt_output, re.DOTALL)

    if not code_blocks:
        raise ValueError(f"No `{lang}` code block found in response.")

    # 2) If requested, just take the last block
    if last_block_only:
        code = code_blocks[-1]
    else:
        # Join all blocks together
        code = "\n".join(block.strip() for block in code_blocks)

    # 3) Remove any // comments (from '//' to end of line)
    #    This will clean out lines like:   "position": {"x": 0, "y": 20},  // comment
    cleaned = re.sub(r"//.*?$", "", code, flags=re.MULTILINE)

    return cleaned

# raw = extract_code_from_response(json_str, lang="json", last_block_only=True)
cleaned = re.sub(r"//.*?$", "", json_str, flags=re.MULTILINE)
obj = json.loads(cleaned)
print(obj)