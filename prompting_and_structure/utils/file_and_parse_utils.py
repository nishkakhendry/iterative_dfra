import importlib.util
import os
import sys
import re
import base64
from PIL import Image
import io
import unicodedata
import re
import numpy as np
import json
import os
from io import BytesIO


def write_error(file_path, error_text):
    with open(file_path, "w") as file:
        file.write(error_text)


def switch_file_extension(file_path, new_extension):
    directory = os.path.dirname(file_path)
    filename, _ = os.path.splitext(os.path.basename(file_path))
    return os.path.join(directory, f"{filename}.{new_extension}")


def add_suffix_to_filename(path, suffix):
    directory = os.path.dirname(path)
    filename, extension = os.path.splitext(os.path.basename(path))
    new_filename = f"{filename}{suffix}{extension}"
    return os.path.join(directory, new_filename)


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def save_base64_image(base64_str, file_path):
    # Extract base64 image data
    base64_data = base64_str.split(",")[1]
    image_data = base64.b64decode(base64_data)
    # Save image
    image = Image.open(BytesIO(image_data))
    image.save(file_path)


# Function to convert NumPy arrays to lists
def convert_numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_to_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, default=convert_numpy_to_list, indent=4)


def markdown_json(data):
    pretty_data = json.dumps(data, default=convert_numpy_to_list)
    return f"```json\n{pretty_data}\n```"


def load_from_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_file(data, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(data)


def load_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def import_function_from_file(file_path, function_name):
    # Create a module spec from the file location
    spec = importlib.util.spec_from_file_location("module.name", file_path)

    # Create a module from the spec
    module = importlib.util.module_from_spec(spec)

    # Add the module to sys.modules
    sys.modules["module.name"] = module

    # Execute the module
    spec.loader.exec_module(module)

    # Get the function
    function = getattr(module, function_name)

    return function


def extract_code_from_response(
    gpt_output: str, lang: str = "python", last_block_only=False
) -> str:
    # Regular expression to match Python code enclosed in '''python ... '''
    code_blocks = re.findall(rf"```{lang}(.*?)```", gpt_output, re.DOTALL)


    if last_block_only:
        # return code_blocks[-1]
        cleaned = re.sub(r"//.*?$", "", code_blocks[-1], flags=re.MULTILINE)
        return cleaned
    # Concatenate all code blocks into a single string
    concatenated_code = "\n".join(code_block.strip() for code_block in code_blocks)
    return concatenated_code


def get_last_json_as_dict(gpt_output):
    try:
        extracted = extract_code_from_response(gpt_output, lang="json", last_block_only=True)
        json_code = json.loads(
            extracted
        )
        return json_code
    except:
        print(gpt_output)
        print("ERROR in code extraction")
        raise AssertionError("Terminating Process Early Because of bad JSON Response")
