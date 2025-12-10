# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import logging
import os
import pickle
import socket
import threading
from contextlib import contextmanager
from copy import deepcopy
from types import MethodType
from typing import Any, Dict, List, Union
import random
import re
import numpy as np
import ray
import torch
import torch.distributed
import zmq
from filelock import FileLock
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from vllm.lora.request import LoRARequest
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.worker.worker_base import WorkerWrapperBase
import copy

from verl import DataProto
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout
from recipe.countdown.reward_function import compute_score

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        
        self.config = config

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(
                model_hf_config.llm_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(
                model_hf_config.text_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")

            assert max_position_embeddings >= config.prompt_length + config.response_length, (
                "model context length should be greater than total sequence length"
            )

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs
        # copy it to avoid secretly modifying the engine config
        engine_kwargs = (
            {}
            if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs
            else OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
        )
        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=config.free_cache_engine,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **lora_kwargs,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        if config.free_cache_engine:
            self.inference_engine.sleep(level=1)

        kwargs = dict(
                n=1,
                logprobs=0,  # can be set to 0 and let actor to recompute
                max_tokens=config.response_length,
                include_stop_str_in_output=True,
                stop=["</answer>"]
            )
        
        self.tokenizer = tokenizer
        self.allow_hint = config.allow_hint
        if self.allow_hint:
            kwargs["stop"].append("<hint>")

        self.n = config.n

        kwargs["detokenize"] = True

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences for a batch of prompts.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        """import debugpy
        debugpy.listen(("0.0.0.0", 5678))  # or ("127.0.0.1", 5678)
        print("⏸ Waiting for debugger attach...")
        debugpy.wait_for_client()
        import debugpy; debugpy.breakpoint()"""
        if self.allow_hint:
            return self.agentic_generate_sequences(prompts, **kwargs)
        else:
            return self.regular_generate_sequences(prompts, **kwargs)

    def construct_partial_inputs(self, prompt_ids, partial_completion):
        if len(partial_completion) == 0:
            return self.tokenizer.decode(prompt_ids)
        else:
            full_text = ""
            for tokens in partial_completion:
                full_text += self.tokenizer.decode(tokens)
            return full_text

    def batched_hint(self, hint_indices, idx_to_num_hints, hint_list_dups):
        
        hint_results = []
        #import debugpy; debugpy.breakpoint()
    
        for curr_idx in hint_indices:
            num_hint = int(idx_to_num_hints[curr_idx])
            if num_hint < len(hint_list_dups[curr_idx]):
                hint = hint_list_dups[curr_idx][num_hint - 1]
            else:
                hint = "We have no more hints to give"
            hint_results.append(hint)

        return hint_results

    def agentic_loop(self, prompts, sampling_params, hint_list, **kwargs):

        raw_prompt_ids = _repeat_interleave(prompts.non_tensor_batch["raw_prompt_ids"], sampling_params.n)
        targets = _repeat_interleave(prompts.non_tensor_batch["target"], sampling_params.n)
        numbers = [nos for nos in prompts.non_tensor_batch["numbers"] for _ in range(sampling_params.n)]

        num_outputs = len(raw_prompt_ids)
        max_hint_structure = [0, 1, 2, 3]
        #max_hint_structure = [0, 1, 2, 3, 0, 1, 2, 3]
        self.max_hints = 3
        #max_hint_structure = [self.max_hints, self.max_hints, self.max_hints, self.max_hints, self.max_hints, self.max_hints, self.max_hints, self.max_hints, self.max_hints, self.max_hints, self.max_hints, self.max_hints, self.max_hints, self.max_hints, self.max_hints, self.max_hints]
        hint_list_logs = [max_hint_structure[i] for _ in range(len(hint_list)) for i in range(len(max_hint_structure))]


        idx_to_partials = [[] for _ in range(num_outputs)]
        idx_to_masks = [[] for _ in range(num_outputs)]
        idx_to_num_hints = [0.0 for _ in range(num_outputs)]
        idx_to_truncation = [0.0 for _ in range(num_outputs)]
        completed_indices = [False for _ in range(num_outputs)]
        hint_list_dups = [h for h in hint_list for _ in range(sampling_params.n)]

        hint_pattern = r'<hint>'
        answer_pattern = r'<answer>(.*?)</answer>' 
        eps = prompts.non_tensor_batch["epsilon"][0]
        sampling_params_nohint = copy.deepcopy(sampling_params)
        sampling_params_nohint.bad_words = ['</h']

        
        for attempt in range(self.max_hints + 1):
            tp_rank = vllm_ps.get_tensor_model_parallel_rank()
            if tp_rank == 0:
                curr_indices_nohint = [idx for idx, completed in enumerate(completed_indices) if not completed and hint_list_logs[idx] == 0]
                inputs_no_hint = [self.construct_partial_inputs(raw_prompt_ids[idx], idx_to_partials[idx]) for idx in curr_indices_nohint]
                curr_indices_hint = [idx for idx, completed in enumerate(completed_indices) if not completed and hint_list_logs[idx] > 0]
                inputs_hint = [self.construct_partial_inputs(raw_prompt_ids[idx], idx_to_partials[idx]) for idx in curr_indices_hint]
                broadcast_data = {'curr_indices_nohint' : curr_indices_nohint, 'curr_indices_hint': curr_indices_hint,  'inputs_no_hint' : inputs_no_hint, 'inputs_hint': inputs_hint}
            else:
                broadcast_data = None

            broadcast_data = vllm_ps._TP.broadcast_object(broadcast_data, src=0)
            curr_indices_nohint, curr_indices_hint, inputs_no_hint, inputs_hint = broadcast_data['curr_indices_nohint'], broadcast_data['curr_indices_hint'], broadcast_data['inputs_no_hint'], broadcast_data['inputs_hint']

            if torch.distributed.is_initialized():
                print(f"Rank {torch.distributed.get_rank()}: Pre-generation barrier")
                torch.distributed.barrier()
                print(f"Rank {torch.distributed.get_rank()}: Post-generation barrier")
            
            outputs_no_hint = self.inference_engine.generate(inputs_no_hint, sampling_params=sampling_params_nohint, use_tqdm=True)
            outputs_hint = self.inference_engine.generate(inputs_hint, sampling_params=sampling_params, use_tqdm=True)

            # First pass: Process the generated text
            hint_indices = []
            for curr_idx, output in zip(curr_indices_nohint + curr_indices_hint, outputs_no_hint + outputs_hint):
                partial_tokens = idx_to_partials[curr_idx]
                partial_masks = idx_to_masks[curr_idx]

                # First append what we have already
                if len(partial_tokens) == 0:
                    partial_tokens.append(output.prompt_token_ids)
                output_text = output.outputs[0].text
                partial_tokens.append(output.outputs[0].token_ids)
                partial_masks.append([1] * len(output.outputs[0].token_ids))

                hint_matches = re.findall(hint_pattern, output_text, re.DOTALL)
                answer_matches = re.findall(answer_pattern, output_text, re.DOTALL)
                if len(hint_matches) != 0:
                    idx_to_num_hints[curr_idx] += 1
                    hint_list_logs[curr_idx] -= 1

                    # Case 1: The model searches
                    if idx_to_num_hints[curr_idx] > self.max_hints:
                        # Add the failure message
                        hint_result = "\n<warning>HINT LIMIT REACHED</warning></hint>\n"
                        doc_tokens = list(self.tokenizer(hint_result, add_special_tokens=False)['input_ids'])
                        partial_tokens.append(doc_tokens)
                        partial_masks.append([0] * len(doc_tokens))

                        # Early truncation
                        if self.exceeds_vllm_length(partial_tokens):
                            completed_indices[curr_idx] = True
                            idx_to_truncation[curr_idx] = 1
                    else:
                        hint_indices.append(curr_idx)
                elif len(answer_matches) != 0:
                    score_curr = compute_score("countdown", output_text, {"target": targets[curr_idx], "numbers": numbers[curr_idx]}, None)["score_wo_hint_penalty"]
                    if random.random() < eps and score_curr < 1 and hint_list_logs[curr_idx] > 0:
                        if idx_to_num_hints[curr_idx] > self.max_hints:
                            completed_indices[curr_idx] = True
                        else:
                            idx_to_num_hints[curr_idx] += 1
                            doc_tokens = list(self.tokenizer('<hint> ', add_special_tokens=False)['input_ids'])
                            partial_generation = re.sub(answer_pattern, '', output_text)

                            partial_tokens[-1] = self.tokenizer(partial_generation)["input_ids"]
                            partial_masks[-1] = partial_masks[-1][:len(partial_tokens[-1])]
                            partial_tokens.append(doc_tokens)
                            partial_masks.append([0] * len(doc_tokens))
                            hint_indices.append(curr_idx)
                    else:
                        completed_indices[curr_idx] = True

                else:
                    # Case 3
                    # Remove eos token 
                    if partial_tokens[-1][-1] == self.tokenizer.eos_token_id:
                        partial_tokens[-1] = partial_tokens[-1][:-1]
                        partial_masks[-1] = partial_masks[-1][:-1]

                    # Append course correction message
                    correction_message = "\nLet me try again.\n" #"\nLet me try again. Maybe ask for a hint?\n"

                    correction_tokens = list(self.tokenizer(correction_message, add_special_tokens=False)['input_ids'])
                    partial_tokens.append(correction_tokens)
                    course_correction_mask_val = 0
                    partial_masks.append([course_correction_mask_val] * len(correction_tokens))
                    # Early truncation
                    if self.exceeds_vllm_length(partial_tokens):
                        completed_indices[curr_idx] = True
                        idx_to_truncation[curr_idx] = 1

            if tp_rank == 0:
                hint_results = self.batched_hint(hint_indices, idx_to_num_hints, hint_list_dups)
                broadcast_data = {
                    'hint_results': hint_results,
                }
            else:
                broadcast_data = None
            hint_results = vllm_ps._TP.broadcast_object(broadcast_data, src=0)['hint_results'] # broadcast tool call results across tp

            for curr_idx, hint_result in zip(hint_indices, hint_results):
                # Add the search results
                partial_tokens = idx_to_partials[curr_idx]
                partial_masks = idx_to_masks[curr_idx]
                doc_tokens = list(self.tokenizer("Correct solution uses " + hint_result + " </hint> Okay, so let me use this.", add_special_tokens=False)['input_ids'])

                partial_tokens.append(doc_tokens)
                partial_masks.append([0] * len(doc_tokens))

                # Early truncation
                if self.exceeds_vllm_length(partial_tokens):
                    completed_indices[curr_idx] = True
                    idx_to_truncation[curr_idx] = 1


        return idx_to_partials, idx_to_masks, idx_to_num_hints, idx_to_truncation        

    def process_agentic_outputs(self, prompts, idx_to_partials, idx_to_masks, idx_to_truncation, device):
        # Get the vllm response and document mask tokens; pad them
        response = []
        document_mask = []
        decoded_responses = []
        for idx, (partial_tokens, partial_masks) in enumerate(zip(idx_to_partials, idx_to_masks)):
            curr_tokens = []
            curr_masks = []
            for tokens, masks in zip(partial_tokens[1:], partial_masks):
                curr_tokens += tokens
                curr_masks += masks

            num_tokens = len(curr_tokens)
            response.append(curr_tokens[:self.config.response_length])
            document_mask.append(curr_masks[:self.config.response_length])
            decoded_responses.append(self.tokenizer.decode(curr_tokens[:self.config.response_length]))
            if num_tokens > self.config.response_length:
                idx_to_truncation[idx] = 1
            
        response = pad_2d_list_to_length(response, self.pad_token_id).to(device)
        document_mask = pad_2d_list_to_length(document_mask, 0).to(device)
        prompts.non_tensor_batch["completions"] = np.array(decoded_responses, dtype='object')

        return response, document_mask

    def exceeds_vllm_length(self, partial_tokens):
        full_text = ""
        for tokens in partial_tokens:
            full_text += self.tokenizer.decode(tokens)
        retokenized_tokens = list(self.tokenizer(full_text, add_special_tokens=False)['input_ids'])
        return len(retokenized_tokens) >= (self.config.prompt_length + self.config.response_length)


    def agentic_generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # Carry over from veRL
        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        hint_list = prompts.non_tensor_batch["hint_exprs"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']
        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        """if 'raw_prompt_ids' not in non_tensor_batch:
            non_tensor_batch['raw_prompt_ids'] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch['raw_prompt_ids']):
            raise RuntimeError('vllm sharding manager is not work properly.')

        #vllm_inputs = [{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")]"""

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        """for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )"""

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            """outputs = self.inference_engine.generate(
                                                    prompts=vllm_inputs,  # because we have already convert it to prompt token id
                                                    sampling_params=self.sampling_params,
                                                    use_tqdm=False,
                                                )"""
            #import debugpy; debugpy.breakpoint()
            idx_to_partials, idx_to_masks, idx_to_num_hints, idx_to_truncation = self.agentic_loop(prompts, self.sampling_params, hint_list=hint_list)
            
            response, document_mask = self.process_agentic_outputs(prompts, idx_to_partials, idx_to_masks, idx_to_truncation, idx.device)

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)
            

            """response = []
            rollout_log_probs = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    response.append(response_ids)
                    if self.config.calculate_log_probs:
                        curr_log_prob = []
                        for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                            curr_log_prob.append(logprob[response_ids[i]].logprob)
                        rollout_log_probs.append(curr_log_prob)

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(
                idx.device
            )
            if self.config.calculate_log_probs:
                rollout_log_probs = pad_2d_list_to_length(
                    rollout_log_probs, -1, max_length=self.config.response_length
                ).to(idx.device)
                rollout_log_probs = rollout_log_probs.to(torch.float32)
            """
            if self.sampling_params.n > 1 and do_sample:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                # NOTE(linjunrong): for multi-turn https://github.com/volcengine/verl/pull/1037
                if "tools_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["tools_kwargs"] = _repeat_interleave(
                        non_tensor_batch["tools_kwargs"], self.sampling_params.n
                    )
                if "interaction_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["interaction_kwargs"] = _repeat_interleave(
                        non_tensor_batch["interaction_kwargs"], self.sampling_params.n
                    )
                if "raw_prompt" in non_tensor_batch.keys():
                    non_tensor_batch["raw_prompt"] = _repeat_interleave(
                        non_tensor_batch["raw_prompt"], self.sampling_params.n
                    )

                if "raw_prompt_ids" in non_tensor_batch.keys():
                    non_tensor_batch["raw_prompt_ids"] = _repeat_interleave(
                        non_tensor_batch["raw_prompt_ids"], self.sampling_params.n
                    )

                if "hint_exprs" in non_tensor_batch.keys():
                    non_tensor_batch["hint_exprs"] = [h for h in hint_list for _ in range(self.sampling_params.n)]

                if "target" in non_tensor_batch.keys():
                    non_tensor_batch["target"] = [h for h in non_tensor_batch["target"] for _ in range(self.sampling_params.n)]

                if "numbers" in non_tensor_batch.keys():
                    non_tensor_batch["numbers"] = [h for h in non_tensor_batch["numbers"] for _ in range(self.sampling_params.n)]

                if "epsilon" in non_tensor_batch.keys():
                    non_tensor_batch["epsilon"] = _repeat_interleave(
                        non_tensor_batch["epsilon"], self.sampling_params.n
                    )
            
            
            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        full_attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        attention_mask_with_documents = torch.cat([attention_mask, document_mask], dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        """batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "attention_mask": full_attention_mask,
                "document_mask": attention_mask_with_documents,
                "position_ids": position_ids,
                "num_hints": idx_to_num_hints
            },
            batch_size=batch_size,
        )"""

        batch_dict = {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "attention_mask": full_attention_mask,
                "document_mask": attention_mask_with_documents,
                "position_ids": position_ids,
                "num_hints": torch.Tensor(idx_to_num_hints)
            }
        """if self.config.calculate_log_probs:
            # we will recompute old log prob with actor
            batch["rollout_log_probs"] = rollout_log_probs"""

        return DataProto.from_dict(batch_dict, non_tensors=non_tensor_batch)

    def regular_generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")
        
        
        vllm_inputs = [{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            rollout_log_probs = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    response.append(response_ids)
                    if self.config.calculate_log_probs:
                        curr_log_prob = []
                        for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                            curr_log_prob.append(logprob[response_ids[i]].logprob)
                        rollout_log_probs.append(curr_log_prob)

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(
                idx.device
            )
            if self.config.calculate_log_probs:
                rollout_log_probs = pad_2d_list_to_length(
                    rollout_log_probs, -1, max_length=self.config.response_length
                ).to(idx.device)
                rollout_log_probs = rollout_log_probs.to(torch.float32)

            if self.sampling_params.n > 1 and do_sample:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                # NOTE(linjunrong): for multi-turn https://github.com/volcengine/verl/pull/1037
                if "tools_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["tools_kwargs"] = _repeat_interleave(
                        non_tensor_batch["tools_kwargs"], self.sampling_params.n
                    )
                if "interaction_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["interaction_kwargs"] = _repeat_interleave(
                        non_tensor_batch["interaction_kwargs"], self.sampling_params.n
                    )
                if "raw_prompt" in non_tensor_batch.keys():
                    non_tensor_batch["raw_prompt"] = _repeat_interleave(
                        non_tensor_batch["raw_prompt"], self.sampling_params.n
                    )
                if "raw_prompt_ids" in non_tensor_batch.keys():
                    non_tensor_batch["raw_prompt_ids"] = _repeat_interleave(
                        non_tensor_batch["raw_prompt_ids"], self.sampling_params.n
                    )

                if "hint_exprs" in non_tensor_batch.keys():
                    non_tensor_batch["hint_exprs"] = [h for h in non_tensor_batch["hint_exprs"] for _ in range(self.sampling_params.n)]
                
                if "target" in non_tensor_batch.keys():
                    non_tensor_batch["target"] = [h for h in non_tensor_batch["target"] for _ in range(self.sampling_params.n)]

                if "numbers" in non_tensor_batch.keys():
                    non_tensor_batch["numbers"] = [h for h in non_tensor_batch["numbers"] for _ in range(self.sampling_params.n)]

                if "epsilon" in non_tensor_batch.keys():
                    non_tensor_batch["epsilon"] = _repeat_interleave(
                        non_tensor_batch["epsilon"], self.sampling_params.n
                    )

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid

        batch_dict = {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
        if self.config.calculate_log_probs:
            # we will recompute old log prob with actor
            batch_dict.update({"rollout_log_probs":rollout_log_probs})

        return DataProto.from_dict(batch_dict, non_tensors=non_tensor_batch)

# https://github.com/vllm-project/vllm/issues/13175
def _monkey_patch_compute_logits(model, vocab_size: int):
    original_compute_logits = model.compute_logits

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        logits = original_compute_logits(hidden_states, sampling_metadata)
        logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)


class vLLMAsyncRollout:
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase,
    which is engine in single worker process.
    """

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        self.tokenizer = tokenizer

        # Engine is deferred to be initialized in init_worker
        self.config = config
        self.inference_engine: WorkerWrapperBase = None
        self.sharding_manager = None
        self.is_sleep = False
        self.address = self._init_zeromq()

    def _init_zeromq(self) -> str:
        tensor_parallel_size = self.config.tensor_model_parallel_size

        # single node: ipc, multi nodes: tcp
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        socket_type = "ipc" if tensor_parallel_size <= local_world_size else "tcp"

        # File lock to prevent multiple workers listen to same port
        with FileLock("/tmp/verl_vllm_zmq.lock"):
            if socket_type == "ipc":
                pid = os.getpid()
                address = f"ipc:///tmp/verl_vllm_zmq_{pid}.ipc"
            else:
                ip, port = self._get_free_port()
                address = f"tcp://{ip}:{port}"
            context = zmq.Context()
            self.socket = context.socket(zmq.REP)
            self.socket.bind(address)

        self.loop_thread = threading.Thread(target=self._loop_forever)
        self.loop_thread.start()

        return address

    def _get_free_port(self):
        ip = ray._private.services.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        return ip, port

    def _loop_forever(self):
        while True:
            message = self.socket.recv()
            method, args, kwargs = pickle.loads(message)
            result = self.execute_method(method, *args, **kwargs)
            self.socket.send(pickle.dumps(result))

    def get_zeromq_address(self):
        return self.address

    def init_worker(self, all_kwargs: List[Dict[str, Any]]):
        """Initialize worker engine."""
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        all_kwargs[0]["local_rank"] = 0

        self.vllm_config = all_kwargs[0]["vllm_config"]
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)

        # inference engine is initialized now, update sharding manager
        self.sharding_manager.inference_engine = self.inference_engine
        self.sharding_manager.model_runner = self.inference_engine.worker.model_runner

        _monkey_patch_compute_logits(self.inference_engine.worker.model_runner.model, len(self.tokenizer))

    def sleep(self, *args, **kwargs):
        """Offload model weights and discard kv cache."""
        if self.is_sleep:
            return
        self.sharding_manager.__exit__(None, None, None)
        self.is_sleep = True

    def wake_up(self, *args, **kwargs):
        """Load model weights and build kv cache."""
        if not self.is_sleep:
            return
        self.sharding_manager.__enter__()  # pylint: disable=C2801
        self.is_sleep = False

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        if method == "init_worker":
            return self.init_worker(*args, **kwargs)
        elif method == "load_model":
            return self.load_model(*args, **kwargs)
        elif method == "sleep":
            return self.sleep(*args, **kwargs)
        elif method == "wake_up":
            return self.wake_up(*args, **kwargs)
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)