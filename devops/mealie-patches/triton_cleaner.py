# # import os
# # import numpy as np
# # from mealie.core.root_logger import get_logger

# # logger = get_logger("triton-cleaner")


# # def clean_with_triton(recipe_text: str) -> str | None:
# #     """
# #     Sends recipe text to the Triton inference server for cleaning.
# #     Returns cleaned text, or None if Triton is unavailable.
# #     """
# #     triton_url = os.environ.get("TRITON_SERVER_URL")
# #     model_name = os.environ.get("TRITON_MODEL_NAME", "recipe_model")

# #     if not triton_url:
# #         logger.warning("TRITON_SERVER_URL not set, skipping Triton cleaning")
# #         return None

# #     # strip the http:// prefix — tritonclient expects host:port
# #     url = triton_url.replace("http://", "").replace("https://", "")

# #     try:
# #         import tritonclient.http as httpclient
# #         from tritonclient.utils import np_to_triton_dtype

# #         client = httpclient.InferenceServerClient(url=url)

# #         text_tensor = np.array([[recipe_text]], dtype=object)
# #         infer_input = httpclient.InferInput("INPUT_TEXT", text_tensor.shape, "BYTES")
# #         infer_input.set_data_from_numpy(text_tensor)

# #         requested_output = httpclient.InferRequestedOutput("OUTPUT_TEXT")

# #         response = client.infer(
# #             model_name=model_name,
# #             inputs=[infer_input],
# #             outputs=[requested_output],
# #         )

# #         output = response.as_numpy("OUTPUT_TEXT")
# #         return output[0][0].decode("utf-8") if output is not None else None

# #     except Exception:
# #         logger.exception("Triton cleaning failed, falling back to original recipe")
# #         return None
# import os
# import numpy as np
# import struct
# import requests
# from mealie.core.root_logger import get_logger

# logger = get_logger("triton-cleaner")


# def clean_with_triton(recipe_text: str) -> str | None:
#     triton_url = os.environ.get("TRITON_SERVER_URL", "").rstrip("/")
#     model_name = os.environ.get("TRITON_MODEL_NAME", "recipe_model")

#     if not triton_url:
#         logger.warning("TRITON_SERVER_URL not set, skipping Triton cleaning")
#         return None

#     try:
#         encoded = recipe_text.encode("utf-8")
#         # Triton binary format for BYTES: 4-byte little-endian length prefix + data
#         byte_data = struct.pack("<I", len(encoded)) + encoded

#         payload = {
#             "inputs": [{
#                 "name": "INPUT_TEXT",
#                 "shape": [1, 1],
#                 "datatype": "BYTES",
#                 "data": [recipe_text]
#             }],
#             "outputs": [{"name": "OUTPUT_TEXT"}]
#         }

#         response = requests.post(
#             f"{triton_url}/v2/models/{model_name}/infer",
#             json=payload,
#             timeout=30,
#         )
#         response.raise_for_status()
#         result = response.json()
#         return result["outputs"][0]["data"][0]

#     except Exception:
#         logger.exception("Triton cleaning failed")
#         return None

import os
import requests
from mealie.core.root_logger import get_logger

logger = get_logger("triton-cleaner")


def clean_with_triton(recipe_text: str) -> str | None:
    triton_url = os.environ.get("TRITON_SERVER_URL", "").rstrip("/")
    model_name = os.environ.get("TRITON_MODEL_NAME", "recipe_model")
    recipe_text = recipe_text[:800]
    if not triton_url:
        logger.warning("TRITON_SERVER_URL not set, skipping Triton cleaning")
        return None

    logger.info(f"Sending to Triton at {triton_url}, model={model_name}, input_len={len(recipe_text)}")

    # Truncate to ~800 chars to stay within 256 token limit
    payload = {
        "inputs": [{
            "name": "INPUT_TEXT",
            "shape": [1, 1],
            "datatype": "BYTES",
            "data": [recipe_text]
        }],
        "outputs": [{"name": "OUTPUT_TEXT"}]
    }

    try:
        response = requests.post(
            f"{triton_url}/v2/models/{model_name}/infer",
            json=payload,
            timeout=60,
        )
        logger.info(f"Triton response status: {response.status_code}")
        response.raise_for_status()
        result = response.json()
        output = result["outputs"][0]["data"][0]
        logger.info(f"Triton output: {output[:100]}")
        return output
    except Exception:
        logger.exception("Triton cleaning failed")
        return None