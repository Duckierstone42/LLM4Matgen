# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import sys
import time
import json
import copy
from tqdm import tqdm
from datetime import datetime
from omegaconf import OmegaConf
from typing import Any, Dict, List, Optional, Union

import torch
from omegaconf import DictConfig
from torch import nn

from torchtune import config, training, utils
from torchtune.config._utils import _get_component_from_path
from torchtune.data import ChatFormat, InstructTemplate, Message

from llm4structgen.generation.unconditional_generation_prompts import *

logger = utils.get_logger("DEBUG")


class InferenceRecipe:
    """
    Recipe for generating tokens from a dense Transformer-based LLM.

    Currently this recipe supports single-GPU generation only. Speculative
    decoding is not supported.

    For more details on how to use this recipe for generation, please see our
    tutorial: https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#generation

    For using this recipe with a quantized model, please the following section of
    the above tutorial:
    https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#speeding-up-generation-using-quantization
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(dtype=cfg.dtype, device=self._device)
        self._quantizer = config.instantiate(cfg.quantizer)
        self._quantization_mode = training.get_quantizer_mode(self._quantizer)

        # utils.set_seed(seed=cfg.seed)

    def setup(self, cfg: DictConfig) -> None:
        checkpointer = config.instantiate(cfg.checkpointer)
        if self._quantization_mode is None:
            ckpt_dict = checkpointer.load_checkpoint()
        else:
            # weights_only needs to be False when loading a quantized model
            # currently loading a quantized model is only supported with the
            # FullModelTorchTuneCheckpointer
            ckpt_dict = checkpointer.load_checkpoint(weights_only=False)

        self._model = self._setup_model(
            model_cfg=cfg.model,
            model_state_dict=ckpt_dict[training.MODEL_KEY],
            enable_kv_cache=cfg.enable_kv_cache,
        )
        self._tokenizer = config.instantiate(cfg.tokenizer)

    def _setup_model(
        self,
        model_cfg: DictConfig,
        model_state_dict: Dict[str, Any],
        enable_kv_cache: bool = True,
    ) -> nn.Module:
        with training.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(model_cfg)

        if self._quantization_mode is not None:
            model = self._quantizer.quantize(model)
            model = model.to(device=self._device, dtype=self._dtype)

        model.load_state_dict(model_state_dict)

        # Validate model was loaded in with the expected dtype.
        training.validate_expected_param_dtype(
            model.named_parameters(), dtype=self._dtype
        )
        logger.info(f"Model is initialized with precision {self._dtype}.")

        # Ensure the cache is setup on the right device
        if enable_kv_cache:
            with self._device:
                model.setup_caches(batch_size=1, dtype=self._dtype)

        return model

    def convert_prompt_to_tokens(
        self,
        prompt: Union[DictConfig, str],
        chat_format: Optional[ChatFormat],
        instruct_template: Optional[InstructTemplate],
    ) -> List[Message]:
        """
        Either:
        (1) a raw string is passed as the prompt, in which case we call tokenizer.encode directly, or
        (2) a DictConfig is passed as the prompt. In this case there are three possibilities:
            (a) an InstructTemplate is provided. Since instruct templates output a string, we will
                call tokenizer.encode on the output of the instruct template.
            (b) a ChatFormat is provided. Since chat formats output a list of messages, we will
                call tokenizer.tokenize_messages on the output of the chat format.
            (c) neither an InstructTemplate nor a ChatFormat is provided. In this case we will
                convert the DictConfig to a list of messages and call tokenizer.tokenize_messages directly.
        """

        # Should only be chat-style prompt or instruct-style prompt
        if chat_format and instruct_template:
            raise ValueError(
                "Cannot pass both chat format and instruct template for generation"
            )

        # If instruct template is provided, assert that the prompt is a DictConfig
        # and apply it
        if instruct_template:
            if not isinstance(prompt, DictConfig):
                raise ValueError("Cannot apply instruct template to raw string")
            instruct_template = _get_component_from_path(instruct_template)
            prompt = instruct_template.format(prompt)

        # To hit this block, either the raw prompt is a string or an
        # instruct template has been provided to convert it to a string
        if isinstance(prompt, str):
            return self._tokenizer.encode(prompt, add_bos=True, add_eos=False)

        # dict.items() will respect order for Python >= 3.7
        else:
            messages = [Message(role=k, content=v) for k, v in prompt.items()]
            messages += [Message(role="assistant", content="")]
            if chat_format:
                chat_format = _get_component_from_path(chat_format)
                messages = chat_format.format(messages)
            return self._tokenizer.tokenize_messages(messages)[0]

    @torch.no_grad()
    def generate(self, cfg: DictConfig):
        tokens = self.convert_prompt_to_tokens(
            cfg.prompt, cfg.get("chat_format", None), cfg.get("instruct_template", None)
        )
        prompt = torch.tensor(tokens, dtype=torch.int, device=self._device)

        custom_generate_next_token = None

        # since quantized model uses torch.compile to get speedup, it needs a warm up / prefill run
        # to get the accurate performance measurement
        if self._quantization_mode is not None:
            logger.info("Starting compilation to improve generation performance ...")
            custom_generate_next_token = torch.compile(
                utils.generate_next_token, mode="max-autotune", fullgraph=True
            )
            t0 = time.perf_counter()
            _ = utils.generate(
                model=self._model,
                prompt=prompt,
                max_generated_tokens=2,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                stop_tokens=self._tokenizer.stop_tokens,
                custom_generate_next_token=custom_generate_next_token,
            )
            t = time.perf_counter() - t0
            logger.info(f"Warmup run for quantized model takes: {t:.02f} sec")

        t0 = time.perf_counter()
        generated_tokens = utils.generate(
            model=self._model,
            prompt=prompt,
            max_generated_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            stop_tokens=self._tokenizer.stop_tokens,
            custom_generate_next_token=custom_generate_next_token,
        )
        t = time.perf_counter() - t0

        logger.info(self._tokenizer.decode(generated_tokens[0]))

        model_size = sum(
            [
                p.numel() * p.dtype.itemsize
                for p in itertools.chain(
                    self._model.parameters(), self._model.buffers()
                )
            ]
        )

        tokens_generated = len(generated_tokens[0]) - prompt.size(0)
        tokens_sec = tokens_generated / t
        logger.info(
            f"Time for inference: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec"
        )
        logger.info(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")
        logger.info(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")
    
    @torch.no_grad()
    def _generate(self, cfg: DictConfig):
        # if seed is set in the config, deterministic generation is enabled
        # if seed is not set, a random seed is used
        # utils.set_seed(seed=cfg.seed)

        if cfg.prompt is not None:
            _prompt = cfg.prompt
        elif cfg.representation_type == "cartesian":
            _prompt = UNCONDITIONAL_CARTESIAN_GENERATION_PROMPT_HEADER
        elif cfg.representation_type == "distance":
            _prompt = UNCONDITIONAL_DISTANCE_MATRIX_GENERATION_PROMPT_HEADER
        elif cfg.representation_type == "slices":
            _prompt = UNCONDITIONAL_SLICES_GENERATION_PROMPT_HEADER
        elif cfg.representation_type == "zmatrix":
            _prompt = UNCONDITIONAL_Z_MATRIX_GENERATION_PROMPT_HEADER
        else:
            raise ValueError(f"Invalid representation type: {cfg.representation_type}")

        tokens = self.convert_prompt_to_tokens(
            _prompt, cfg.get("chat_format", None), cfg.get("instruct_template", None)
        )
        prompt = torch.tensor(tokens, dtype=torch.int, device=self._device)

        custom_generate_next_token = None

        generated_tokens = utils.generate(
            model=self._model,
            prompt=prompt,
            max_generated_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            stop_tokens=self._tokenizer.stop_tokens,
            custom_generate_next_token=custom_generate_next_token,
        )

        output_str = self._tokenizer.decode(generated_tokens[0])

        return output_str
    
    def load_encoder(self, representation_type):
        if representation_type == "cartesian":
            from llm4structgen.representations.cartesian import Cartesian
            encoder = Cartesian()
        elif representation_type == "zmatrix":
            from llm4structgen.representations.z_matrix import ZMatrix
            encoder = ZMatrix()
        elif representation_type == "distance":
            from llm4structgen.representations.distance_matrix import DistanceMatrix
            encoder = DistanceMatrix()
        elif representation_type == "slices":
            from llm4structgen.representations.slices import SLICES
            encoder = SLICES()
        else:
            raise ValueError(f"Invalid representation type: {representation_type}")
        
        return encoder

    def generate_and_save(self, cfg: DictConfig):
        n_structures = cfg.generation.n_structures
        output_dir = cfg.generation.output_dir
        require_valid = cfg.generation.require_valid
        model = cfg.model._component_.split('.')[-1]

        timestamp = datetime.now().strftime("%d%m%y_%H%M_%S")
        fname = f"{output_dir}/{model}_{cfg.representation_type}_{timestamp}.json"

        data = []

        if require_valid:
            self.encoder = self.load_encoder(cfg.representation_type)
            
            valid_count = 0
            with tqdm(total=n_structures, desc="Generating valid structures ...") as pbar:
                while valid_count < n_structures:
                    generated_str = self._generate(cfg=cfg)
                    data.append(generated_str)

                    decoded = self.safe_decode(generated_str)
                    if decoded is not None:
                        valid_count += 1
                        pbar.update(1)
        else:
            for _ in tqdm(range(n_structures)):
                data.append(
                    self._generate(cfg=cfg)
                )
        
        # save config and generated data as json
        _cfg = copy.deepcopy(cfg)
        _cfg.outputs = data
        _cfg_dict = OmegaConf.to_container(_cfg, resolve=True)
        with open(fname, 'w') as f:
            json.dump(_cfg_dict, f)

    def safe_decode(self, generated_str):
        _splits = generated_str.strip().split("\n", 1)
        assert len(_splits) == 2
        generated_str = _splits[1]
        
        try:
            decoded = self.encoder.decode(generated_str)
            return decoded
        except Exception as e:
            return None

@config.parse
def main(cfg: DictConfig) -> None:
    config.log_config(recipe_name="InferenceRecipe", cfg=cfg)
    recipe = InferenceRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.generate_and_save(cfg=cfg)

if __name__ == "__main__":
    sys.exit(main())