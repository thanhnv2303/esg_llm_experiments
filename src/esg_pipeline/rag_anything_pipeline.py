from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from .config import ExperimentConfig

@dataclass
class ModelEndpointConfig:
    """Configuration for an LLM/VLM endpoint using an OpenAI-compatible API."""

    model: str
    provider: str = "openai"
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    api_key_file: Optional[Path] = None
    api_base: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)
    _cached_key: Optional[str] = field(default=None, init=False, repr=False)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any], default_provider: str = "openai") -> "ModelEndpointConfig":
        data = dict(payload)
        model = data.pop("model", None)
        if not model:
            raise ValueError("ModelEndpointConfig requires a 'model' field.")
        provider = data.pop("provider", default_provider)
        api_key = data.pop("api_key", None)
        api_key_env = data.pop("api_key_env", None)
        api_key_file_value = data.pop("api_key_file", None)
        api_key_file = Path(api_key_file_value).expanduser() if api_key_file_value else None
        api_base = data.pop("api_base", None)
        if api_key_env is None and not api_key and api_key_file is None and provider.lower() in {"openai", "vllm"}:
            api_key_env = "OPENAI_API_KEY"
        temperature = data.pop("temperature", None)
        top_p = data.pop("top_p", None)
        max_tokens = data.pop("max_tokens", None)
        extra_params = data.pop("extra_params", {})
        if not isinstance(extra_params, dict):
            raise ValueError("extra_params must be a dictionary when provided.")
        for key, value in data.items():
            extra_params.setdefault(key, value)
        return cls(
            model=model,
            provider=provider,
            api_key=api_key,
            api_key_env=api_key_env,
            api_key_file=api_key_file,
            api_base=api_base,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            extra_params=extra_params,
        )

    def resolve_api_key(self, fallback_env: Optional[str] = None) -> Optional[str]:
        if self.api_key:
            return self.api_key
        if self.api_key_file:
            if self._cached_key is None:
                self._cached_key = self._read_key_file(self.api_key_file)
            return self._cached_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        if fallback_env:
            return os.environ.get(fallback_env)
        return None

    @staticmethod
    def _read_key_file(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8").strip()
        except FileNotFoundError as exc:
            raise RuntimeError(f"API key file not found: {path}") from exc
        except OSError as exc:
            raise RuntimeError(f"Failed to read API key file {path}: {exc}") from exc

    def call_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = dict(self.extra_params)
        if self.temperature is not None:
            kwargs.setdefault("temperature", self.temperature)
        if self.top_p is not None:
            kwargs.setdefault("top_p", self.top_p)
        if self.max_tokens is not None:
            kwargs.setdefault("max_tokens", self.max_tokens)
        return kwargs


@dataclass
class EmbeddingConfig:
    """Configuration for the embedding function used by LightRAG."""

    model: str = "text-embedding-3-large"
    provider: str = "openai"
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    api_key_file: Optional[Path] = None
    api_base: Optional[str] = None
    embedding_dim: int = 3072
    max_token_size: int = 8192
    extra_params: Dict[str, Any] = field(default_factory=dict)
    _cached_key: Optional[str] = field(default=None, init=False, repr=False)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EmbeddingConfig":
        data = dict(payload)
        model = data.pop("model", "text-embedding-3-large")
        provider = data.pop("provider", "openai")
        api_key = data.pop("api_key", None)
        api_key_env = data.pop("api_key_env", None)
        api_key_file_value = data.pop("api_key_file", None)
        api_key_file = Path(api_key_file_value).expanduser() if api_key_file_value else None
        api_base = data.pop("api_base", None)
        if api_key_env is None and not api_key and api_key_file is None:
            api_key_env = "OPENAI_API_KEY"
        embedding_dim = int(data.pop("embedding_dim", 3072))
        max_token_size = int(data.pop("max_token_size", 8192))
        extra_params = data.pop("extra_params", {})
        if not isinstance(extra_params, dict):
            raise ValueError("EmbeddingConfig.extra_params must be a dictionary when provided.")
        for key, value in data.items():
            extra_params.setdefault(key, value)
        return cls(
            model=model,
            provider=provider,
            api_key=api_key,
            api_key_env=api_key_env,
            api_key_file=api_key_file,
            api_base=api_base,
            embedding_dim=embedding_dim,
            max_token_size=max_token_size,
            extra_params=extra_params,
        )

    def resolve_api_key(self, fallback_env: Optional[str] = None) -> Optional[str]:
        if self.api_key:
            return self.api_key
        if self.api_key_file:
            if self._cached_key is None:
                self._cached_key = self._read_key_file(self.api_key_file)
            return self._cached_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        if fallback_env:
            return os.environ.get(fallback_env)
        return None

    @staticmethod
    def _read_key_file(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8").strip()
        except FileNotFoundError as exc:
            raise RuntimeError(f"API key file not found: {path}") from exc
        except OSError as exc:
            raise RuntimeError(f"Failed to read API key file {path}: {exc}") from exc


@dataclass
class DocumentProcessingConfig:
    """Options used when processing a document through RAG-Anything."""

    output_dir: Optional[Path] = None
    parse_method: Optional[str] = None
    display_stats: Optional[bool] = None
    split_by_character: Optional[str] = None
    split_by_character_only: bool = False
    doc_id: Optional[str] = None
    device: Optional[str] = None
    device_enforce: bool = False
    device_virtual_memory_gb: Optional[int] = None
    restrict_to_experiment_pages: bool = False
    experiment_page_padding: int = 0
    extra_pages: List[int] = field(default_factory=list)
    page_ranges: List[Tuple[int, int]] = field(default_factory=list)
    clear_existing_document: bool = False
    parser_kwargs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DocumentProcessingConfig":
        data = dict(payload)
        parser_kwargs = data.pop("parser_kwargs", {})
        if parser_kwargs and not isinstance(parser_kwargs, dict):
            raise ValueError("document_processing.parser_kwargs must be a dictionary when provided.")

        extra_pages_payload = data.pop("extra_pages", [])
        extra_pages: List[int] = []
        if isinstance(extra_pages_payload, list):
            for value in extra_pages_payload:
                try:
                    page = int(value)
                except (TypeError, ValueError):
                    continue
                if page > 0:
                    extra_pages.append(page)

        page_ranges_payload = data.pop("page_ranges", [])
        page_ranges: List[Tuple[int, int]] = []
        if isinstance(page_ranges_payload, list):
            for entry in page_ranges_payload:
                if isinstance(entry, dict):
                    start = entry.get("start")
                    end = entry.get("end", start)
                elif isinstance(entry, (list, tuple)) and entry:
                    start = entry[0]
                    end = entry[-1]
                else:
                    start = entry
                    end = entry
                try:
                    start_int = int(start) if start is not None else None
                    end_int = int(end) if end is not None else None
                except (TypeError, ValueError):
                    continue
                if start_int is None and end_int is None:
                    continue
                if start_int is None:
                    start_int = end_int
                if end_int is None:
                    end_int = start_int
                if start_int <= 0 and end_int <= 0:
                    continue
                if start_int <= 0:
                    start_int = 1
                if end_int <= 0:
                    end_int = 1
                if end_int < start_int:
                    start_int, end_int = end_int, start_int
                page_ranges.append((start_int, end_int))

        restrict_pages = bool(data.pop("restrict_to_experiment_pages", False))
        padding_value = data.pop("experiment_page_padding", 0)
        clear_existing = bool(data.pop("clear_existing_document", False))

        try:
            experiment_page_padding = int(padding_value)
        except (TypeError, ValueError):
            experiment_page_padding = 0
        if experiment_page_padding < 0:
            experiment_page_padding = 0

        # Remaining keys that are not explicitly mapped are treated as parser kwargs
        known = {
            "output_dir",
            "parse_method",
            "display_stats",
            "split_by_character",
            "split_by_character_only",
            "doc_id",
            "device",
            "device_enforce",
            "device_virtual_memory_gb",
            "restrict_to_experiment_pages",
            "experiment_page_padding",
            "extra_pages",
            "page_ranges",
            "clear_existing_document",
        }
        for key in list(data.keys()):
            if key not in known:
                parser_kwargs.setdefault(key, data.pop(key))
        output_dir = data.get("output_dir")
        return cls(
            output_dir=Path(output_dir) if output_dir else None,
            parse_method=data.get("parse_method"),
            display_stats=data.get("display_stats"),
            split_by_character=data.get("split_by_character"),
            split_by_character_only=bool(data.get("split_by_character_only", False)),
            doc_id=data.get("doc_id"),
            device=data.get("device"),
            device_enforce=bool(data.get("device_enforce", False)),
            device_virtual_memory_gb=(
                int(data.get("device_virtual_memory_gb"))
                if data.get("device_virtual_memory_gb") is not None
                else None
            ),
            restrict_to_experiment_pages=restrict_pages,
            experiment_page_padding=experiment_page_padding,
            extra_pages=extra_pages,
            page_ranges=page_ranges,
            clear_existing_document=clear_existing,
            parser_kwargs=parser_kwargs,
        )


@dataclass
class QueryConfig:
    """Default query behaviour for the RAG pipeline."""

    mode: str = "mix"
    use_multimodal: bool = False
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "QueryConfig":
        data = dict(payload)
        params = data.pop("params", {})
        if params and not isinstance(params, dict):
            raise ValueError("query.params must be a dictionary when provided.")
        for key, value in data.items():
            if key not in {"mode", "use_multimodal"}:
                params.setdefault(key, value)
        return cls(
            mode=data.get("mode", "mix"),
            use_multimodal=bool(data.get("use_multimodal", False)),
            params=params,
        )


@dataclass
class RAGAnythingPipelineConfig:
    """Top-level configuration for integrating RAG-Anything into the ESG pipeline."""

    rag_settings: Dict[str, Any] = field(default_factory=dict)
    llm: Optional[ModelEndpointConfig] = None
    embedding: Optional[EmbeddingConfig] = None
    vision_model: Optional[ModelEndpointConfig] = None
    lightrag_options: Dict[str, Any] = field(default_factory=dict)
    document_processing: DocumentProcessingConfig = field(default_factory=DocumentProcessingConfig)
    query: QueryConfig = field(default_factory=QueryConfig)

    @classmethod
    def from_json(cls, path: Path) -> "RAGAnythingPipelineConfig":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rag_settings = dict(payload.get("rag_anything", {}))
        llm_payload = payload.get("llm")
        if not llm_payload:
            raise ValueError("RAGAnything pipeline config must include an 'llm' section.")
        llm_config = ModelEndpointConfig.from_dict(llm_payload)

        embedding_config = EmbeddingConfig.from_dict(payload.get("embedding", {}))

        vision_payload = payload.get("vision_model")
        vision_config = (
            ModelEndpointConfig.from_dict(vision_payload)
            if vision_payload
            else None
        )

        lightrag_options = dict(payload.get("lightrag", {}))

        document_processing = DocumentProcessingConfig.from_dict(
            payload.get("document_processing", {})
        )
        query_config = QueryConfig.from_dict(payload.get("query", {}))

        return cls(
            rag_settings=rag_settings,
            llm=llm_config,
            embedding=embedding_config,
            vision_model=vision_config,
            lightrag_options=lightrag_options,
            document_processing=document_processing,
            query=query_config,
        )

    def build_rag_config(self):
        from raganything import RAGAnythingConfig

        valid_fields = set(RAGAnythingConfig.__dataclass_fields__.keys())
        config_kwargs = {
            key: value
            for key, value in self.rag_settings.items()
            if key in valid_fields
        }
        return RAGAnythingConfig(**config_kwargs)


class RAGAnythingPipeline:
    """Thin wrapper around the upstream RAG-Anything pipeline."""

    def __init__(self, config: RAGAnythingPipelineConfig) -> None:
        self.config = config
        self._rag = None
        self._prepared_document: Optional[Path] = None
        self._parser_name: Optional[str] = None
        self.logger = LOGGER.getChild("RAGAnythingPipeline")

    @property
    def rag(self):
        if self._rag is None:
            self._rag = self._create_pipeline()
        return self._rag

    def _create_pipeline(self):
        try:
            from raganything import RAGAnything
            from lightrag.utils import EmbeddingFunc
            from lightrag.llm.openai import openai_complete_if_cache, create_openai_async_client
        except ImportError as exc:
            raise RuntimeError(
                "raganything and lightrag must be installed to use the RAG pipeline."
            ) from exc

        rag_config = self.config.build_rag_config()
        parser_name = getattr(rag_config, "parser", None)
        if isinstance(parser_name, str) and parser_name.strip():
            self._parser_name = parser_name.strip().lower()
        else:
            self._parser_name = "mineru"

        self._enable_docling_verbose_patch()

        # Ensure Milvus database exists when requested before LightRAG initialisation
        self._ensure_milvus_database(rag_config)

        llm_config = self.config.llm
        if llm_config is None:
            raise RuntimeError(
                "LLM configuration is missing. Provide an 'llm' section in the RAG config."
            )

        llm_key = llm_config.resolve_api_key()
        if not llm_key:
            raise RuntimeError(
                "Unable to resolve API key for the LLM endpoint. Set 'api_key' or 'api_key_env'."
            )

        def llm_model_func(
            prompt: str,
            system_prompt: Optional[str] = None,
            history_messages: Optional[List[Dict[str, Any]]] = None,
            **kwargs: Any,
        ):
            call_kwargs = llm_config.call_kwargs()
            call_kwargs.update(kwargs)
            return openai_complete_if_cache(
                llm_config.model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                api_key=llm_key,
                base_url=llm_config.api_base,
                **call_kwargs,
            )

        vision_model_config = self.config.vision_model
        vision_key: Optional[str] = None
        if vision_model_config is not None:
            vision_key = vision_model_config.resolve_api_key() or llm_key

        def vision_model_func(
            prompt: str,
            system_prompt: Optional[str] = None,
            history_messages: Optional[List[Dict[str, Any]]] = None,
            image_data: Optional[str] = None,
            messages: Optional[List[Dict[str, Any]]] = None,
            **kwargs: Any,
        ):
            if vision_model_config is None:
                return llm_model_func(
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    **kwargs,
                )
            if vision_key is None:
                raise RuntimeError(
                    "Vision model configuration is missing credentials. Set 'api_key' or 'api_key_env'."
                )
            call_kwargs = vision_model_config.call_kwargs()
            call_kwargs.update(kwargs)
            if messages is not None:
                return openai_complete_if_cache(
                    vision_model_config.model,
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=messages,
                    api_key=vision_key,
                    base_url=vision_model_config.api_base,
                    **call_kwargs,
                )
            if image_data is not None:
                payload: List[Dict[str, Any]] = []
                if system_prompt:
                    payload.append({"role": "system", "content": system_prompt})
                payload.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                            },
                        ],
                    }
                )
                return openai_complete_if_cache(
                    vision_model_config.model,
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=payload,
                    api_key=vision_key,
                    base_url=vision_model_config.api_base,
                    **call_kwargs,
                )
            return llm_model_func(
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                **kwargs,
            )

        embedding_config = self.config.embedding
        if embedding_config is None:
            raise RuntimeError(
                "Embedding configuration is missing. Provide an 'embedding' section or rely on defaults."
            )
        embedding_key = embedding_config.resolve_api_key(
            fallback_env=llm_config.api_key_env
        )
        if not embedding_key:
            raise RuntimeError(
                "Unable to resolve API key for embedding function. Set 'api_key' or 'api_key_env' in embedding config."
            )

        async def embedding_callable(texts: List[str]):
            call_kwargs = dict(embedding_config.extra_params)
            encoding_format = call_kwargs.pop("encoding_format", "base64")
            input_type = call_kwargs.pop("input_type", None)
            if input_type not in {"passage", "query"}:
                raise RuntimeError("embedding.extra_params.input_type must be 'passage' or 'query'")
            extra_body = dict(call_kwargs.pop("extra_body", {}))
            extra_body.setdefault("input_type", input_type)
            client_configs = call_kwargs.pop("client_configs", None)

            request_kwargs: Dict[str, Any] = {
                "model": embedding_config.model,
                "input": texts,
                "encoding_format": encoding_format,
                **call_kwargs,
            }
            if extra_body:
                request_kwargs["extra_body"] = extra_body

            async_client = create_openai_async_client(
                api_key=embedding_key,
                base_url=embedding_config.api_base,
                client_configs=client_configs,
            )

            async with async_client as client:
                response = await client.embeddings.create(**request_kwargs)

            vectors = []
            for data_point in response.data:
                embedding = data_point.embedding
                if isinstance(embedding, list):
                    vectors.append(np.asarray(embedding, dtype=np.float32))
                else:
                    vectors.append(
                        np.frombuffer(base64.b64decode(embedding), dtype=np.float32)
                    )
            return np.asarray(vectors)

        embedding_func = EmbeddingFunc(
            embedding_dim=embedding_config.embedding_dim,
            max_token_size=embedding_config.max_token_size,
            func=embedding_callable,
        )

        lightrag_kwargs = dict(self.config.lightrag_options)

        self.logger.info(
            "Initialising RAG-Anything with working directory %s",
            rag_config.working_dir,
        )
        return RAGAnything(
            config=rag_config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func if vision_model_config else None,
            embedding_func=embedding_func,
            lightrag_kwargs=lightrag_kwargs,
        )

    def _ensure_milvus_database(self, rag_config) -> None:
        vector_storage = self.config.lightrag_options.get("vector_storage")
        if vector_storage != "MilvusVectorDBStorage":
            return

        db_name = os.environ.get("MILVUS_DB_NAME")
        if not db_name:
            return

        try:
            from pymilvus import MilvusClient  # type: ignore
        except ImportError:
            self.logger.warning(
                "Milvus vector storage requested but pymilvus is not installed; "
                "unable to ensure database '%s' exists.",
                db_name,
            )
            return

        uri = os.environ.get("MILVUS_URI")
        user = os.environ.get("MILVUS_USER")
        password = os.environ.get("MILVUS_PASSWORD")
        token = os.environ.get("MILVUS_TOKEN")

        if not uri:
            import configparser

            config = configparser.ConfigParser()
            config.read("config.ini", encoding="utf-8")
            uri = config.get("milvus", "uri", fallback=None)

        if not uri:
            uri = os.path.join(rag_config.working_dir, "milvus_lite.db")

        client = None
        try:
            client = MilvusClient(uri=uri, user=user, password=password, token=token)
            existing_databases = client.list_databases()
            if db_name not in existing_databases:
                client.create_database(db_name)
                self.logger.info("Created Milvus database '%s'", db_name)
        except Exception as exc:  # pragma: no cover - depends on external service
            self.logger.warning(
                "Unable to ensure Milvus database '%s' exists: %s", db_name, exc
            )
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass

    def _select_parser_device(
        self, requested_device: Optional[str], enforce: bool
    ) -> Optional[str]:
        if not requested_device:
            return None

        device = requested_device.strip()
        if not device:
            return None

        lowered = device.lower()

        def _fallback(reason: str, cpu_value: str = "cpu") -> str:
            if enforce:
                self.logger.warning(
                    "%s; continuing with requested device '%s' due to enforcement",
                    reason,
                    device,
                )
                return device
            self.logger.warning(
                "%s; falling back to CPU for parser execution", reason
            )
            return cpu_value

        if lowered.startswith("cuda"):
            allocator_key = "expandable_segments"
            allocator_entry = f"{allocator_key}:True"
            existing_allocator = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
            if not existing_allocator:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = allocator_entry
            else:
                parts = [part.strip() for part in existing_allocator.split(",") if part.strip()]
                replaced = False
                seen = False
                for index, part in enumerate(parts):
                    key, _, value = part.partition(":")
                    if key == allocator_key:
                        seen = True
                        if value.lower() != "true":
                            parts[index] = allocator_entry
                            replaced = True
                if not seen:
                    parts.append(allocator_entry)
                    replaced = True
                if replaced:
                    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(parts)
            try:
                import torch  # type: ignore

                try:
                    available = torch.cuda.is_available()
                except Exception as exc:  # pragma: no cover - depends on CUDA runtime
                    self.logger.warning(
                        "CUDA availability check failed (%s)", exc
                    )
                    return _fallback("CUDA availability check failed")

                if not available:
                    return _fallback("CUDA requested but no GPU is available")

                if ":" in lowered:
                    index_text = lowered.split(":", 1)[1]
                    if index_text.isdigit():
                        try:
                            device_index = int(index_text)
                            device_count = torch.cuda.device_count()
                        except Exception as exc:  # pragma: no cover - driver dependent
                            self.logger.warning(
                                "Failed to query CUDA device count (%s)", exc
                            )
                            return _fallback("Unable to query CUDA device count")

                        if device_index >= device_count:
                            if device_count > 0:
                                message = (
                                    "CUDA device index %s requested but only %s device(s) detected"
                                    % (index_text, device_count)
                                )
                                if enforce:
                                    self.logger.warning(
                                        "%s; continuing with requested device '%s'",
                                        message,
                                        device,
                                    )
                                    return device
                                self.logger.warning("%s; using cuda:0 instead.", message)
                                return "cuda:0"
                            return _fallback(
                                "CUDA device index out of range", cpu_value="cpu"
                            )

                return device
            except ImportError:
                return _fallback(
                    "CUDA requested but PyTorch with CUDA support is not installed"
                )

        if lowered.startswith("mps"):
            try:
                import torch  # type: ignore

                mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
                if not mps_backend or not mps_backend.is_available():
                    return _fallback("MPS requested but not available")
                return device
            except ImportError:
                return _fallback(
                    "MPS requested but PyTorch with MPS support is not installed"
                )

        if lowered.startswith("npu"):
            try:
                import torch  # type: ignore

                if not hasattr(torch, "npu") or not torch.npu.is_available():
                    return _fallback("NPU requested but not available")
                return device
            except ImportError:
                return _fallback(
                    "NPU requested but PyTorch with NPU support is not installed"
                )

        return device

    def _enable_docling_verbose_patch(self) -> None:
        try:
            from raganything.parser import DoclingParser
        except ImportError:
            return

        if getattr(DoclingParser, "_esg_docling_verbose_patch", False):
            return

        original_impl = DoclingParser._run_docling_command

        def _compute_verbose_flags(value: Optional[str]) -> List[str]:
            if value is None:
                return []
            text = str(value).strip().lower()
            if text in {"", "0", "none", "false"}:
                return []
            if text in {"debug", "trace"}:
                return ["-v", "-v"]
            try:
                level = int(text)
            except ValueError:
                level = 1
            level = max(0, min(level, 2))
            return ["-v"] * level

        def patched_run_docling_command(self, input_path, output_dir, file_stem, **kwargs):
            verbose_override = kwargs.pop("verbose", None)
            verbose_env = os.environ.get("DOCLING_VERBOSE_LEVEL")
            if verbose_env is None:
                verbose_env = os.environ.get("DOCLING_VERBOSE")

            verbose_flags = _compute_verbose_flags(verbose_override)
            if not verbose_flags:
                verbose_flags = _compute_verbose_flags(verbose_env)
            if (
                not verbose_flags
                and verbose_override is None
                and verbose_env is None
            ):
                verbose_flags = ["-v"]
            if not verbose_flags and os.environ.get("DOCLING_ENABLE_LOGS", "").lower() in {"1", "true", "yes"}:
                verbose_flags = ["-v"]

            file_output_dir = Path(output_dir) / file_stem / "docling"
            file_output_dir.mkdir(parents=True, exist_ok=True)

            cmd_json = [
                "docling",
                *verbose_flags,
                "--output",
                str(file_output_dir),
                "--to",
                "json",
                str(input_path),
            ]
            cmd_md = [
                "docling",
                *verbose_flags,
                "--output",
                str(file_output_dir),
                "--to",
                "md",
                str(input_path),
            ]

            try:
                import platform

                docling_subprocess_kwargs = {
                    "capture_output": True,
                    "text": True,
                    "check": True,
                    "encoding": "utf-8",
                    "errors": "ignore",
                }

                if platform.system() == "Windows":
                    docling_subprocess_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

                result_json = subprocess.run(cmd_json, **docling_subprocess_kwargs)
                result_md = subprocess.run(cmd_md, **docling_subprocess_kwargs)
                logging.info("Docling command executed successfully")
                if result_json.stdout:
                    logging.info("[Docling] %s", result_json.stdout.strip())
                if result_json.stderr:
                    logging.warning("[Docling] %s", result_json.stderr.strip())
                if result_md.stdout:
                    logging.debug("[Docling] %s", result_md.stdout.strip())
                if result_md.stderr:
                    logging.warning("[Docling] %s", result_md.stderr.strip())
            except subprocess.CalledProcessError as exc:
                logging.error("Error running docling command: %s", exc)
                if exc.stderr:
                    logging.error("Error details: %s", exc.stderr.strip())
                raise
            except FileNotFoundError:
                raise RuntimeError(
                    "docling command not found. Please ensure Docling is properly installed."
                )

        DoclingParser._run_docling_command = patched_run_docling_command  # type: ignore[assignment]
        DoclingParser._esg_docling_verbose_patch = True

    def _determine_parser_name(self, parser_hint: Optional[str]) -> str:
        if parser_hint and parser_hint.strip():
            return parser_hint.strip().lower()
        if self._parser_name:
            return self._parser_name
        return "mineru"

    def _resolve_page_range(
        self,
        doc_config: DocumentProcessingConfig,
        experiment: Optional["ExperimentConfig"],
    ) -> Optional[Tuple[int, int]]:
        pages: set[int] = set()

        for start, end in doc_config.page_ranges:
            if end < start:
                start, end = end, start
            for page in range(start, end + 1):
                if page > 0:
                    pages.add(page)

        for page in doc_config.extra_pages:
            if page > 0:
                pages.add(page)

        if doc_config.restrict_to_experiment_pages:
            if experiment is None:
                self.logger.warning(
                    "restrict_to_experiment_pages is enabled but no experiment metadata was provided; skipping task-derived pages."
                )
            else:
                padding = max(0, doc_config.experiment_page_padding)
                for task in getattr(experiment, "tasks", []):
                    task_page = getattr(task, "page", None)
                    if task_page is None:
                        continue
                    for offset in range(-padding, padding + 1):
                        candidate = task_page + offset
                        if candidate > 0:
                            pages.add(candidate)

        if not pages:
            return None

        start_page = min(pages)
        end_page = max(pages)
        return start_page, end_page

    def _slice_document_to_range(
        self,
        document_path: Path,
        start_page: int,
        end_page: int,
    ) -> Tuple[Path, Optional[Callable[[], None]]]:
        if start_page >= end_page:
            end_page = start_page

        try:
            import fitz  # type: ignore
        except ImportError:  # pragma: no cover - optional dependency
            self.logger.warning(
                "PyMuPDF is required to slice PDFs for docling page filtering; processing full document instead."
            )
            return document_path, None

        try:
            source = fitz.open(str(document_path))
        except Exception as exc:  # pragma: no cover - environment dependent
            self.logger.warning(
                "Unable to open '%s' for page slicing (%s); processing full document instead.",
                document_path,
                exc,
            )
            return document_path, None

        try:
            total_pages = source.page_count
            if total_pages <= 0:
                return document_path, None

            start_index = max(0, min(start_page, total_pages) - 1)
            end_index = max(0, min(end_page, total_pages) - 1)
            if start_index == 0 and end_index >= total_pages - 1:
                return document_path, None

            subset = fitz.open()
            try:
                for index in range(start_index, end_index + 1):
                    subset.insert_pdf(source, from_page=index, to_page=index)
                temp_dir = Path(tempfile.mkdtemp(prefix="rag_slice_"))
                temp_path = temp_dir / f"{document_path.stem}_p{start_page}-{end_page}{document_path.suffix}"
                subset.save(temp_path)
            finally:
                subset.close()

        except Exception as exc:  # pragma: no cover - environment dependent
            self.logger.warning(
                "Failed to generate page slice for '%s' (%s); processing full document instead.",
                document_path,
                exc,
            )
            return document_path, None
        finally:
            source.close()

        def cleanup() -> None:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass

        return temp_path, cleanup

    def _configure_parser_environment(
        self,
        parser_name: Optional[str],
        device: Optional[str],
        virtual_memory_gb: Optional[int],
    ) -> None:
        parser_value = parser_name or self._parser_name or "mineru"
        parser_kind = parser_value.strip().lower()

        # Sanitise stale environment overrides before setting parser-specific hints
        if parser_kind != "mineru":
            for key in ("MINERU_DEVICE_MODE", "MINERU_VIRTUAL_VRAM_SIZE"):
                if key in os.environ:
                    os.environ.pop(key)
        if parser_kind != "docling" and "DOCLING_DEVICE" in os.environ:
            os.environ.pop("DOCLING_DEVICE")

        if parser_kind == "mineru" and not device:
            for key in ("MINERU_DEVICE_MODE", "MINERU_VIRTUAL_VRAM_SIZE"):
                if key in os.environ:
                    os.environ.pop(key)
            return

        if parser_kind == "docling":
            if device:
                self._configure_docling_environment(device)
            elif "DOCLING_DEVICE" in os.environ:
                os.environ.pop("DOCLING_DEVICE")
            return

        if parser_kind != "mineru" or not device:
            return

        lowered = device.lower()
        env_updates: Dict[str, str] = {"MINERU_DEVICE_MODE": device}

        if lowered.startswith(("cuda", "npu")):
            if virtual_memory_gb and virtual_memory_gb > 0:
                env_updates["MINERU_VIRTUAL_VRAM_SIZE"] = str(virtual_memory_gb)
            else:
                detected = self._detect_virtual_memory_gb(device)
                if detected:
                    env_updates["MINERU_VIRTUAL_VRAM_SIZE"] = str(detected)
                else:
                    # MinerU requires a VRAM hint when running with accelerator backends.
                    env_updates.setdefault("MINERU_VIRTUAL_VRAM_SIZE", "12")

        for key in ("MINERU_DEVICE_MODE", "MINERU_VIRTUAL_VRAM_SIZE"):
            if key not in env_updates and key in os.environ:
                os.environ.pop(key)

        for key, value in env_updates.items():
            os.environ[key] = value

    def _configure_docling_environment(self, device: str) -> None:
        normalized = device.strip()
        if not normalized:
            return

        normalized_value = normalized.lower()
        desired = normalized_value
        if normalized_value.startswith("cuda") and normalized != normalized_value:
            desired = normalized_value

        current = os.environ.get("DOCLING_DEVICE")
        if current != desired:
            os.environ["DOCLING_DEVICE"] = desired

    def _flush_accelerator_cache(self, device: str) -> None:
        lowered = device.lower()

        if lowered.startswith("cuda"):
            try:
                import torch  # type: ignore

                index = torch.cuda.current_device()
                if ":" in lowered:
                    index_text = lowered.split(":", 1)[1]
                    if index_text.isdigit():
                        index = int(index_text)

                with torch.cuda.device(index):
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()
                    if hasattr(torch.cuda, "reset_peak_memory_stats"):
                        torch.cuda.reset_peak_memory_stats()
            except Exception as exc:  # pragma: no cover - environment specific
                self.logger.debug(
                    "Unable to clear CUDA allocator before parser execution: %s",
                    exc,
                )

    def _detect_virtual_memory_gb(self, device: str) -> Optional[int]:
        lowered = device.lower()
        if lowered.startswith("cuda"):
            try:
                import torch  # type: ignore

                if not torch.cuda.is_available():
                    return None

                index: int
                if lowered == "cuda":
                    index = torch.cuda.current_device()
                elif ":" in lowered:
                    index_text = lowered.split(":", 1)[1]
                    if index_text.isdigit():
                        index = int(index_text)
                    else:
                        index = torch.cuda.current_device()
                else:
                    index = torch.cuda.current_device()

                props = torch.cuda.get_device_properties(index)
                memory_gb = max(1, int(props.total_memory / (1024**3)))
                return memory_gb
            except Exception:
                return None

        if lowered.startswith("npu"):
            try:
                import torch  # type: ignore

                if not hasattr(torch, "npu") or not torch.npu.is_available():
                    return None
                props = torch.npu.get_device_properties(device)
                return max(1, int(props.total_memory / (1024**3)))
            except Exception:
                return None

        return None

    def prepare_document(
        self,
        document_path: Path,
        experiment: Optional["ExperimentConfig"] = None,
    ) -> None:
        if not document_path.exists():
            raise FileNotFoundError(f"Document not found: {document_path}")
        if self._prepared_document and document_path.resolve() == self._prepared_document:
            return
        processor = self.rag
        doc_config = self.config.document_processing
        kwargs = dict(doc_config.parser_kwargs)
        parser_override = kwargs.get("parser")
        parser_hint = parser_override if isinstance(parser_override, str) else None
        parser_name = self._determine_parser_name(parser_hint)
        resolved_device: Optional[str] = None
        if doc_config.device and "device" not in kwargs:
            resolved_device = self._select_parser_device(
                doc_config.device, doc_config.device_enforce
            )
            if resolved_device:
                kwargs["device"] = resolved_device
        self._configure_parser_environment(
            parser_hint,
            resolved_device,
            doc_config.device_virtual_memory_gb,
        )
        if resolved_device:
            self._flush_accelerator_cache(resolved_device)

        page_range = self._resolve_page_range(doc_config, experiment)
        source_document = document_path
        cleanup: Optional[Callable[[], None]] = None

        if page_range:
            start_page, end_page = page_range
            self.logger.info(
                "Limiting parser '%s' to document pages %s-%s",
                parser_name,
                start_page,
                end_page,
            )
            if parser_name == "docling":
                source_document, cleanup = self._slice_document_to_range(
                    document_path, start_page, end_page
                )
                if source_document is document_path:
                    self.logger.warning(
                        "Docling page slicing unavailable; continuing with the full document."
                    )
            elif "start_page" not in kwargs and "end_page" not in kwargs:
                kwargs["start_page"] = max(0, start_page - 1)
                kwargs["end_page"] = max(0, end_page - 1)
            else:
                self.logger.debug(
                    "Start/end page hints already supplied in parser kwargs; using configured values."
                )

        if doc_config.clear_existing_document:
            doc_id_hint = doc_config.doc_id
            if doc_id_hint:
                try:
                    self._run_async(self.rag.lightrag.adelete_by_doc_id(doc_id_hint))
                    self.logger.info(
                        "Cleared existing LightRAG artefacts for doc_id '%s' before ingestion.",
                        doc_id_hint,
                    )
                except Exception as exc:  # pragma: no cover - storage dependent
                    self.logger.warning(
                        "Failed to clear LightRAG document '%s': %s",
                        doc_id_hint,
                        exc,
                    )
            else:
                self.logger.warning(
                    "clear_existing_document is enabled but document_processing.doc_id is not set; skipping cleanup."
                )

        try:
            self._run_async(
                processor.process_document_complete(
                    str(source_document),
                    output_dir=str(doc_config.output_dir)
                    if doc_config.output_dir
                    else None,
                    parse_method=doc_config.parse_method,
                    display_stats=doc_config.display_stats,
                    split_by_character=doc_config.split_by_character,
                    split_by_character_only=doc_config.split_by_character_only,
                    doc_id=doc_config.doc_id,
                    **kwargs,
                )
            )
        finally:
            if cleanup:
                cleanup()

        self._prepared_document = document_path.resolve()

    def query(
        self,
        prompt: str,
        image_paths: Optional[List[Path]] = None,
    ) -> Dict[str, Any]:
        query_config = self.config.query
        use_multimodal = bool(query_config.use_multimodal and image_paths)
        payload: Dict[str, Any] = {
            "mode": query_config.mode,
            "params": dict(query_config.params),
        }
        if use_multimodal:
            payload["multimodal"] = [
                {"type": "image", "img_path": str(path)}
                for path in image_paths or []
                if path.exists()
            ]
            if not payload["multimodal"]:
                use_multimodal = False

        if not use_multimodal and "multimodal" in payload:
            payload.pop("multimodal", None)
        return self._execute_query(prompt, payload)

    def _execute_query(self, prompt: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        params = dict(payload.get("params", {}))
        mode = payload.get("mode", "mix")
        multimodal = payload.get("multimodal")
        processor = self.rag
        if multimodal:
            result = self._run_async(
                processor.aquery_with_multimodal(
                    prompt,
                    multimodal_content=multimodal,
                    mode=mode,
                    **params,
                )
            )
        else:
            result = self._run_async(
                processor.aquery(
                    prompt,
                    mode=mode,
                    **params,
                )
            )
        return {
            "result": result,
            "mode": mode,
            "multimodal": bool(multimodal),
            "params": params,
        }

    @staticmethod
    def _run_async(coro) -> Any:
        try:
            return asyncio.run(coro)
        except RuntimeError as exc:  # pragma: no cover - defensive fallback
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()


__all__ = [
    "ModelEndpointConfig",
    "EmbeddingConfig",
    "DocumentProcessingConfig",
    "QueryConfig",
    "RAGAnythingPipelineConfig",
    "RAGAnythingPipeline",
]
