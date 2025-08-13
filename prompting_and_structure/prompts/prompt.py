import os

from matplotlib import pyplot as plt
import numpy as np
from prompting_and_structure.utils.file_and_parse_utils import (
    save_to_json,
    load_from_json,
    save_file,
    load_file,
    save_base64_image,
)
from prompting_and_structure.utils.gpt_client import GPTClient

# Prompt GPT via OpenAI API and log responses
def prompt_with_caching(
    messages_and_images,
    context,
    save_dir,
    name,
    cache=True,
    i=0,
    system_message=None,
    temperature=0,
    img_path=None,
    second_view_path=None
):
    if type(messages_and_images) == str:
        messages_and_images = [messages_and_images]

    # Create necessary directories
    os.makedirs(os.path.join(save_dir, "responses"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "context"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "prompts"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "prompts", "images"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "context", "images"), exist_ok=True)

    new_name = f"{name}_{i}"

    # Paths for saving prompt, response, and context
    prompt_path = os.path.join(save_dir, "prompts", f"{new_name}_prompt.md")
    response_path = os.path.join(save_dir, "responses", f"{new_name}_response.md")
    context_path = os.path.join(save_dir, "context", f"{new_name}_context.json")
    context_markdown_path = os.path.join(save_dir, "context", f"{new_name}_context.md")

    # Bypass cache for specific prompt types
    do_not_use_cache = ['missing_suggestions','topple', 're_plan', 'order_replan', 'positions_replan']
    if cache and os.path.exists(response_path) and os.path.exists(context_path) and name not in do_not_use_cache:
        response = load_file(response_path)
        context = load_from_json(context_path)
    else:
        # Prompt GPT
        if system_message:
            response, context = GPTClient.prompt_gpt(
                messages_and_images,
                context=context,
                system_message=system_message,
                temperature=temperature,
                name=name,
                img_path=img_path,
                second_view_path=second_view_path
            )
        else:
            response, context = GPTClient.prompt_gpt(
                messages_and_images, context=context, temperature=temperature,name=name,img_path=img_path, second_view_path=second_view_path
            )

    save_file(response, response_path)
    save_to_json(context, context_path)

    # Save prompt as markdown with images
    prompt_md = ""
    img_counter = 0
    for j, message in enumerate(messages_and_images):
        if type(message) == str:
            prompt_md += message + "\n"
            continue
        img_path = os.path.join(
            save_dir, "prompts", "images", f"{new_name}_image_{img_counter}.png"
        )
        relative_img_path = os.path.join(
            "images", f"{new_name}_image_{img_counter}.png"
        )
        if type(message) == np.ndarray:
            plt.imsave(img_path, message)
        else:
            message.save(img_path)
        prompt_md += f"![image{j}]({relative_img_path})\n"
        img_counter += 1
    save_file(prompt_md, prompt_path)

    # Save context as markdown with images
    context_md = ""
    img_counter = 0
    for i, entry in enumerate(context):
        role = entry.get("role")
        content = entry.get("content")

        context_md += f"# {role.capitalize()}\n\n"

        if isinstance(content, list):
            for item in content:
                if item["type"] == "text":
                    context_md += f"{item['text']}\n\n"
                elif item["type"] == "image_url":
                    img_path = os.path.join(
                        save_dir,
                        "context",
                        "images",
                        f"{new_name}_image_{img_counter}.png",
                    )
                    relative_img_path = os.path.join(
                        "images", f"{new_name}_image_{img_counter}.png"
                    )
                    context_md += f"![Image {img_counter}]({relative_img_path})\n\n"
                    base64_str = item["image_url"]["url"]
                    save_base64_image(base64_str, img_path)
                    img_counter += 1
        else:
            context_md += f"{content}\n\n"

    save_file(context_md, context_markdown_path)

    return response, context
