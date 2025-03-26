import asyncio
import json
from dataclasses import dataclass
from http import HTTPStatus
from typing import Dict, List, Optional, Union

from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ErrorResponse,
    LogProbs,
    ModelCard,
    ModelList,
    ModelPermission,
    CompletionLogProbs,
)
from vllm.sequence import Logprob
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.delta.request import DeltaRequest
from vllm.swap.request import SwapRequest
from vllm.transformers_utils.tokenizer import get_tokenizer

logger = init_logger(__name__)


@dataclass
class LoRA:
    name: str
    local_path: str

    def to_json(self):
        return {
            "name": self.name,
            "local_path": self.local_path,
        }


@dataclass
class Delta:
    name: str
    local_path: str

    def to_json(self):
        return {
            "name": self.name,
            "local_path": self.local_path,
        }


@dataclass
class SwapModule:
    name: str
    local_path: str

    def to_json(self):
        return {
            "name": self.name,
            "local_path": self.local_path,
        }


class OpenAIServing:

    def __init__(
        self,
        engine: AsyncLLMEngine,
        served_model: str,
        lora_modules=Optional[List[LoRA]],
        delta_modules=Optional[List[Delta]],
        swap_modules=Optional[List[SwapModule]],
    ):
        self.engine = engine
        self.served_model = served_model
        if lora_modules is None:
            self.lora_requests = []
        else:
            self.lora_requests = [
                LoRARequest(
                    lora_name=lora.name,
                    lora_int_id=i,
                    lora_local_path=lora.local_path,
                )
                for i, lora in enumerate(lora_modules, start=1)
            ]
        if delta_modules is None:
            self.delta_requests = []
        else:
            self.delta_requests = [
                DeltaRequest(
                    delta_name=delta.name,
                    delta_int_id=i,
                    delta_local_path=delta.local_path,
                )
                for i, delta in enumerate(delta_modules, start=1)
            ]
        if swap_modules is None:
            self.swap_requests = []
        else:
            self.swap_requests = [
                SwapRequest(
                    swap_name=swap.name,
                    swap_int_id=i,
                    swap_local_path=swap.local_path,
                )
                for i, swap in enumerate(swap_modules, start=1)
            ]
        self.max_model_len = 0
        self.tokenizer = None

        try:
            event_loop = asyncio.get_running_loop()
        except RuntimeError:
            event_loop = None

        if event_loop is not None and event_loop.is_running():
            # If the current is instanced by Ray Serve,
            # there is already a running event loop
            event_loop.create_task(self._post_init())
        else:
            # When using single vLLM without engine_use_ray
            asyncio.run(self._post_init())

    async def _post_init(self):
        engine_model_config = await self.engine.get_model_config()
        self.max_model_len = engine_model_config.max_model_len

        # A separate tokenizer to map token IDs to strings.
        self.tokenizer = get_tokenizer(
            engine_model_config.tokenizer,
            tokenizer_mode=engine_model_config.tokenizer_mode,
            trust_remote_code=engine_model_config.trust_remote_code,
        )

        if len(self.tokenizer) != engine_model_config.get_vocab_size():
            logger.warning(
                f"The tokenizer's vocabulary size {len(self.tokenizer)}"
                f" does not match the model's vocabulary size "
                f"{engine_model_config.get_vocab_size()}. This might "
                f"cause an error in decoding. Please change config.json "
                "to match the tokenizer's vocabulary size."
            )

    async def show_available_models(self) -> ModelList:
        """Show available models. Right now we only have one model."""
        model_cards = [
            ModelCard(
                id=self.served_model,
                root=self.served_model,
                permission=[ModelPermission()],
            )
        ]
        lora_cards = [
            ModelCard(
                id=lora.lora_name,
                root=self.served_model,
                permission=[ModelPermission()],
            )
            for lora in self.lora_requests
        ]
        delta_cards = [
            ModelCard(
                id=delta.delta_name,
                root=self.served_model,
                permission=[ModelPermission()],
            )
            for delta in self.delta_requests
        ]
        swap_cards = [
            ModelCard(
                id=swap.swap_name,
                root=self.served_model,
                permission=[ModelPermission()],
            )
            for swap in self.swap_requests
        ]
        model_cards.extend(lora_cards)
        model_cards.extend(delta_cards)
        model_cards.extend(swap_cards)

        return ModelList(data=model_cards)

    def _create_logprobs(
        self,
        token_ids: List[int],
        top_logprobs: Optional[List[Optional[Dict[int, Logprob]]]] = None,
        num_output_top_logprobs: Optional[int] = None,
        initial_text_offset: int = 0,
    ) -> LogProbs:
        """Create OpenAI-style logprobs."""
        logprobs = LogProbs()
        last_token_len = 0
        if num_output_top_logprobs:
            logprobs.top_logprobs = []
        out_text_offset: List[int] = []
        out_token_logprobs: List[Optional[float]] = []
        out_tokens: List[str] = []
        out_top_logprobs: List[Optional[Dict[str, float]]] = []

        last_token_len = 0

        for i, token_id in enumerate(token_ids):
            step_top_logprobs = top_logprobs[i]
            if step_top_logprobs is None:
                token = self.tokenizer.decode(token_id)
                out_tokens.append(token)
                out_token_logprobs.append(None)
                out_top_logprobs.append(None)
            else:
                token = self._get_decoded_token(step_top_logprobs[token_id],
                                                token_id)
                token_logprob = max(step_top_logprobs[token_id].logprob,
                                    -9999.0)
                out_tokens.append(token)
                out_token_logprobs.append(token_logprob)

                # makes sure to add the top num_output_top_logprobs + 1
                # logprobs, as defined in the openai API
                # (cf. https://github.com/openai/openai-openapi/blob/
                # 893ba52242dbd5387a97b96444ee1c742cfce9bd/openapi.yaml#L7153)
                out_top_logprobs.append({
                    # Convert float("-inf") to the
                    # JSON-serializable float that OpenAI uses
                    self._get_decoded_token(top_lp[1], top_lp[0]):
                    max(top_lp[1].logprob, -9999.0)
                    for i, top_lp in enumerate(step_top_logprobs.items())
                    if num_output_top_logprobs >= i
                })

            if len(out_text_offset) == 0:
                out_text_offset.append(initial_text_offset)
            else:
                out_text_offset.append(out_text_offset[-1] + last_token_len)
            last_token_len = len(token)

        return CompletionLogProbs(
            text_offset=out_text_offset,
            token_logprobs=out_token_logprobs,
            tokens=out_tokens,
            top_logprobs=out_top_logprobs,
        )

    def create_error_response(
        self,
        message: str,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
    ) -> ErrorResponse:
        return ErrorResponse(message=message, type=err_type, code=status_code.value)

    def create_streaming_error_response(
        self,
        message: str,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
    ) -> str:
        json_str = json.dumps(
            {
                "error": self.create_error_response(
                    message=message, err_type=err_type, status_code=status_code
                ).model_dump()
            }
        )
        return json_str

    async def _check_model(self, request) -> Optional[ErrorResponse]:
        if request.model == self.served_model:
            return
        if request.model in [lora.lora_name for lora in self.lora_requests]:
            return
        if request.model in [delta.delta_name for delta in self.delta_requests]:
            return
        if request.model in [swap.swap_name for swap in self.swap_requests]:
            return

        return self.create_error_response(
            message=f"The model [{request.model}] does not exist. Expected one of {self.served_model}, {[lora.lora_name for lora in self.lora_requests]}, {[delta.delta_name for delta in self.delta_requests]}, {[swap.swap_name for swap in self.swap_requests]}",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND,
        )

    def _maybe_get_lora(self, request) -> Optional[LoRARequest]:
        if request.model == self.served_model:
            return
        for lora in self.lora_requests:
            if request.model == lora.lora_name:
                return lora
        return None

    def _maybe_get_delta(self, request) -> Optional[DeltaRequest]:
        if request.model == self.served_model:
            return
        for delta in self.delta_requests:
            if request.model == delta.delta_name:
                return delta
        return None

    def _maybe_get_swap(self, request) -> Optional[SwapRequest]:
        if request.model == self.served_model:
            return
        for swap in self.swap_requests:
            if request.model == swap.swap_name:
                return swap
        return None

    def _validate_prompt_and_tokenize(
        self,
        request: Union[ChatCompletionRequest, CompletionRequest],
        prompt: Optional[str] = None,
        prompt_ids: Optional[List[int]] = None,
    ) -> List[int]:
        if not (prompt or prompt_ids):
            raise ValueError("Either prompt or prompt_ids should be provided.")
        if prompt and prompt_ids:
            raise ValueError("Only one of prompt or prompt_ids should be provided.")

        input_ids = (
            prompt_ids if prompt_ids is not None else self.tokenizer(prompt).input_ids
        )
        token_num = len(input_ids)

        if request.max_tokens is None:
            request.max_tokens = self.max_model_len - token_num

        if token_num + request.max_tokens > self.max_model_len:
            raise ValueError(
                f"This model's maximum context length is "
                f"{self.max_model_len} tokens. However, you requested "
                f"{request.max_tokens + token_num} tokens "
                f"({token_num} in the messages, "
                f"{request.max_tokens} in the completion). "
                f"Please reduce the length of the messages or completion.",
            )
        else:
            return input_ids
    
    def _get_decoded_token(self, logprob: Logprob, token_id: int) -> str:
        if logprob.decoded_token is not None:
            return logprob.decoded_token
        return self.tokenizer.decode(token_id)