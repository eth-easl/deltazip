"""
This file contains the command line arguments for the vLLM's
OpenAI-compatible server. It is kept in a separate file for documentation
purposes.
"""

import argparse
import json
import ssl

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.serving_engine import LoRA, Delta, SwapModule


class LoRAParserAction(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        lora_list = []
        for item in values:
            name, path = item.split("=")
            lora_list.append(LoRA(name, path))
        setattr(namespace, self.dest, lora_list)


class DeltaParserAction(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        delta_list = []
        for item in values:
            name, path = item.split("=")
            delta_list.append(Delta(name, path))
        setattr(namespace, self.dest, delta_list)


class SwapParserAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        swappable_modules = []
        for item in values:
            name, path = item.split("=")
            swappable_modules.append(SwapModule(name, path))
        setattr(namespace, self.dest, swappable_modules)


def make_arg_parser():
    parser = argparse.ArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser.add_argument("--host", type=str, default=None, help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument(
        "--uvicorn-log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical", "trace"],
        help="log level for uvicorn",
    )
    parser.add_argument(
        "--allow-credentials", action="store_true", help="allow credentials"
    )
    parser.add_argument(
        "--allowed-origins", type=json.loads, default=["*"], help="allowed origins"
    )
    parser.add_argument(
        "--allowed-methods", type=json.loads, default=["*"], help="allowed methods"
    )
    parser.add_argument(
        "--allowed-headers", type=json.loads, default=["*"], help="allowed headers"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="If provided, the server will require this key "
        "to be presented in the header.",
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="The model name used in the API. If not "
        "specified, the model name will be the same as "
        "the huggingface name.",
    )
    parser.add_argument(
        "--lora-modules",
        type=str,
        default=[],
        nargs="+",
        action=LoRAParserAction,
        help="LoRA module configurations in the format name=path. "
        "Multiple modules can be specified.",
    )
    parser.add_argument(
        "--delta-modules",
        type=str,
        default=[],
        nargs="+",
        action=DeltaParserAction,
        help="Delta module configurations in the format name=path. "
        "Multiple modules can be specified.",
    )
    parser.add_argument(
        "--swap-modules",
        type=str,
        default=[],
        nargs="+",
        action=SwapParserAction,
        help="Modules that can be reloaded during requests",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default=None,
        help="The file path to the chat template, "
        "or the template in single-line form "
        "for the specified model",
    )
    parser.add_argument(
        "--response-role",
        type=str,
        default="assistant",
        help="The role name to return if " "`request.add_generation_prompt=true`.",
    )
    parser.add_argument(
        "--ssl-keyfile",
        type=str,
        default=None,
        help="The file path to the SSL key file",
    )
    parser.add_argument(
        "--ssl-certfile",
        type=str,
        default=None,
        help="The file path to the SSL cert file",
    )
    parser.add_argument(
        "--ssl-ca-certs", type=str, default=None, help="The CA certificates file"
    )
    parser.add_argument(
        "--ssl-cert-reqs",
        type=int,
        default=int(ssl.CERT_NONE),
        help="Whether client certificate is required (see stdlib ssl module's)",
    )
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy",
    )
    parser.add_argument(
        "--middleware",
        type=str,
        action="append",
        default=[],
        help="Additional ASGI middleware to apply to the app. "
        "We accept multiple --middleware arguments. "
        "The value should be an import path. "
        "If a function is provided, vLLM will add it to the server "
        "using @app.middleware('http'). "
        "If a class is provided, vLLM will add it to the server "
        "using app.add_middleware(). ",
    )

    parser = AsyncEngineArgs.add_cli_args(parser)
    return parser
