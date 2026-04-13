import os
import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, T5ForConditionalGeneration


class TritonPythonModel:

    def initialize(self, args):
        model_dir = os.path.dirname(__file__)
        # model_path = os.path.join(model_dir, "model.safetensors")

        instance_kind = args.get("model_instance_kind", "cpu").lower()
        if instance_kind == "gpu":
            device_id = int(args.get("model_instance_device_id", 0))
            torch.cuda.set_device(device_id)
            self.device = torch.device(
                f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device("cpu")

        # Load tokenizer and model from the local model directory
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = T5ForConditionalGeneration.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

        # Generation defaults - adjust as needed
        self.max_input_length = 256
        self.max_output_length = 64

    def _to_text_list(self, input_array):
        texts = []
        for item in input_array.reshape(-1):
            if isinstance(item, bytes):
                texts.append(item.decode("utf-8"))
            else:
                texts.append(str(item))
        return texts

    def execute(self, requests):
        responses = []

        for request in requests:
            in_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT")
            input_data = in_tensor.as_numpy()
            texts = self._to_text_list(input_data)

            enc = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_input_length,
                return_tensors="pt"
            )

            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.max_output_length
                )

            outputs = self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )

            out_np = np.array(outputs, dtype=object).reshape(-1, 1)
            out_tensor = pb_utils.Tensor("OUTPUT_TEXT", out_np)

            responses.append(
                pb_utils.InferenceResponse(output_tensors=[out_tensor])
            )

        return responses