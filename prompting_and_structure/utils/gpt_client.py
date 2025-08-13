from io import BytesIO
from openai import OpenAI
from PIL import Image
import numpy as np
import base64
from dotenv import load_dotenv
import requests

# Encode visual inputs for GPT API
def encode_image_from_pil(pil_img: Image.Image):
    pil_img = pil_img.convert("RGB")
    buffered = BytesIO()
    pil_img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def encode_image_from_array(image: np.ndarray):
    pil_img = Image.fromarray(image)
    return encode_image_from_pil(pil_img)


def encode_image_from_file(image_path):
    with Image.open(image_path) as img:
        return encode_image_from_pil(img)

# GPT API wrapper with image support
class GPTClient:
    _instance = None

    @classmethod
    def _get_instance(cls):
        if cls._instance is None:
            cls._instance = OpenAI()  # Initialize the client once
        return cls._instance

    # Convert mixed image/text input to GPT-compatible format
    @classmethod
    def encode_messages_and_images(cls, messages_and_images):
        def _encode_image_content(img):
            if type(img) == str:
                base64_image = encode_image_from_file(img)
            elif type(img) == np.ndarray:
                base64_image = encode_image_from_array(img)
            else:
                base64_image = encode_image_from_pil(img)

            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                },
            }

        def _encode_text_content(message):
            return {"type": "text", "text": message}

        content = []
        for item in messages_and_images:
            if (
                type(item) == str
            ):  # and not os.path.exists(item): if the img input is as a file path this won't work :(
                content.append(_encode_text_content(item))
            else:
                content.append(_encode_image_content(item))

        return content

    # Main prompt interface (handles single or dual image)
    @classmethod
    def prompt_gpt(
        cls,
        messages_and_images,
        context=[],
        max_tokens=3500,
        temperature=0,
        system_message=None,
        name=None,
        img_path=None,
        second_view_path=None
    ):
        """
        get full gpt response object
        """
        client = cls._get_instance()

        content = cls.encode_messages_and_images(messages_and_images)

        if system_message:
            messages = [
                {"role": "system", "content": system_message},
                *context,
                {"role": "user", "content": content},
            ]
        else:
            messages = [*context, {"role": "user", "content": content}]

        # call gpt
        if name not in ['missing_suggestions', 'topple']:
            response = client.chat.completions.create(
                model="gpt-4o", #"gpt-4o-mini",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        else:
            with open(img_path, "rb") as img:
                b64 = base64.b64encode(img.read()).decode()

            # re-craft messages with image path if required - TODO clean up
            if second_view_path:
                with open(second_view_path, "rb") as img:
                    b64_2nd = base64.b64encode(img.read()).decode()
                messages = [{"role": "system", "content": system_message},
                            *context, {"role": "user", "content": [
                        {"type": "text", "text": messages_and_images[0]},
                        {"type": "image_url", "image_url": {"url": f"data:image/gif;base64,{b64}", 'detail': "high"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/gif;base64,{b64_2nd}", 'detail': "high"}}
                    ]}]
            else:
                messages = [{"role": "system", "content": system_message},
                            *context, {"role": "user", "content": [
                                {"type": "text", "text":  messages_and_images[0]},
                                {"type": "image_url", "image_url": {"url": f"data:image/gif;base64,{b64}", 'detail':"high"} } ] } ]

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        # get message from chat completion
        response_message = response.choices[0].message.model_dump()

        # get text response
        response_txt = response_message["content"]
        # add response to context
        new_context = messages + [response_message]

        return response_txt, new_context

# Basic usage example
if __name__ == "__main__":
    load_dotenv()

    response, context = GPTClient.prompt_gpt("hello")
    print(response)
