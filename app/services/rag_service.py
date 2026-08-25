"""Lifespan-managed FAISS retrieval service for iFormat Career Advisor."""

import asyncio
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, NoReturn, cast

from botocore.exceptions import BotoCoreError, ClientError
from langchain_aws import BedrockEmbeddings, ChatBedrockConverse
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.outputs import Generation
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    AIServiceException,
    BedrockThrottlingException,
    BedrockUnavailableException,
    RAGNotReadyException,
    RAGServiceException,
)
from app.services.bedrock_service import (
    THROTTLING_ERROR_CODES,
    BedrockRuntimeClient,
    create_bedrock_runtime_client,
)
from app.utils.prompts import (
    CAREER_ADVISOR_INPUT_PROMPT,
    CAREER_ADVISOR_RAG_PROMPT,
)

logger = logging.getLogger(__name__)

TEXT_FILE_SUFFIXES = frozenset({".md", ".rst", ".text", ".txt"})
PDF_FILE_SUFFIX = ".pdf"


class RAGOutputParser(BaseOutputParser[dict[str, Any]]):
    """Preserve Bedrock token metadata while parsing a RAG answer."""

    model_id: str

    @property
    def _type(self) -> str:
        """Return the serialization identifier for this parser."""

        return "iformat_rag_output"

    def parse(self, text: str) -> dict[str, Any]:
        """Parse plain text when generation metadata is unavailable.

        Args:
            text: Model response text.

        Returns:
            dict[str, Any]: Normalized answer with zero fallback token usage.
        """

        return {"response": text, "model": self.model_id, "tokensUsed": 0}

    def parse_result(
        self,
        result: list[Generation],
        *,
        partial: bool = False,
    ) -> dict[str, Any]:
        """Parse the first generation and extract LangChain usage metadata.

        Args:
            result: Model generations produced by the document chain.
            partial: Whether the generation is incomplete. RAG calls are
                non-streaming, so this flag does not affect parsing.

        Returns:
            dict[str, Any]: Answer, configured model ID, and total tokens.
        """

        del partial
        if not result:
            raise ValueError("The RAG model returned no generations")

        generation = result[0]
        message = getattr(generation, "message", None)
        usage_metadata = getattr(message, "usage_metadata", None) or {}
        total_tokens = usage_metadata.get("total_tokens", 0)
        if not isinstance(total_tokens, int) or isinstance(total_tokens, bool):
            total_tokens = 0

        return {
            "response": generation.text,
            "model": self.model_id,
            "tokensUsed": max(total_tokens, 0),
        }


class RAGService:
    """Own the in-memory FAISS index and career-advisor retrieval chain."""

    def __init__(
        self,
        settings: Settings,
        client: BedrockRuntimeClient | None = None,
    ) -> None:
        """Initialize an unstarted RAG service.

        Args:
            settings: Validated model and knowledge-base configuration.
            client: Optional Bedrock client. If omitted, it is created lazily
                only when knowledge-base documents are available.
        """

        self._settings = settings
        self._client = client
        self._vector_store: FAISS | None = None
        self._chain: Runnable[dict[str, Any], dict[str, Any]] | None = None
        self._initialization_lock = asyncio.Lock()

    @property
    def is_ready(self) -> bool:
        """Return whether the retrieval chain is ready to accept queries."""

        return self._chain is not None

    async def initialize(self) -> None:
        """Load documents, embed chunks, and create the FAISS retrieval chain.

        This method is idempotent and intended to run from FastAPI's lifespan
        context before requests are accepted.

        Raises:
            RAGNotReadyException: If the configured directory has no supported
                knowledge-base documents.
            BedrockThrottlingException: If AWS temporarily rejects embedding.
            BedrockUnavailableException: If Bedrock cannot be reached.
            RAGServiceException: If documents or the index cannot be built.
        """

        async with self._initialization_lock:
            if self.is_ready:
                return

            knowledge_base_path = Path(self._settings.KNOWLEDGE_BASE_PATH)
            documents = await asyncio.to_thread(
                self._load_documents,
                knowledge_base_path,
            )
            if not documents:
                raise RAGNotReadyException(
                    "The career advisor knowledge base contains no supported documents."
                )

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1_000,
                chunk_overlap=150,
                add_start_index=True,
            )
            chunks = text_splitter.split_documents(documents)
            if not chunks:
                raise RAGNotReadyException(
                    "The career advisor knowledge base contains no readable text."
                )

            try:
                client = self._client or create_bedrock_runtime_client(self._settings)
                self._client = client
                embeddings = BedrockEmbeddings(
                    client=client,
                    region_name=self._settings.AWS_REGION,
                    model_id=self._settings.EMBEDDING_MODEL_ID,
                    normalize=True,
                )
                vector_store = await asyncio.to_thread(
                    FAISS.from_documents,
                    chunks,
                    embeddings,
                )
                self._vector_store = vector_store
                self._chain = self._create_chain(vector_store, client)
            except ClientError as exc:
                self._raise_for_aws_error(exc)
            except BotoCoreError as exc:
                logger.exception("Bedrock embeddings could not be reached")
                raise BedrockUnavailableException() from exc
            except AIServiceException:
                raise
            except Exception as exc:
                logger.exception("Failed to initialize the career-advisor RAG service")
                raise RAGServiceException(
                    "The career advisor knowledge base could not be initialized."
                ) from exc

            logger.info(
                "Career-advisor FAISS index initialized with %d chunks",
                len(chunks),
            )

    async def close(self) -> None:
        """Release references held by the in-memory retrieval pipeline."""

        self._chain = None
        self._vector_store = None

    async def query_career_advisor(
        self,
        query: str,
        chat_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Answer a career question using retrieved iFormat context.

        Args:
            query: The user's current career question.
            chat_history: Optional validated prior user/assistant messages.

        Returns:
            dict[str, Any]: Branded answer, model ID, and total token usage.

        Raises:
            RAGNotReadyException: If lifespan initialization has not succeeded.
            BedrockThrottlingException: If AWS asks the caller to retry later.
            BedrockUnavailableException: If the AWS SDK cannot reach Bedrock.
            RAGServiceException: If retrieval or generation fails.
        """

        if self._chain is None:
            raise RAGNotReadyException()

        chain_input = {
            "input": query,
            "chat_history": self._convert_chat_history(chat_history or []),
        }
        try:
            result = await self._chain.ainvoke(chain_input)
            answer = result.get("answer")
            if not isinstance(answer, dict):
                raise TypeError("RAG answer did not include metadata")
            response = answer.get("response")
            if not isinstance(response, str) or not response.strip():
                raise TypeError("RAG answer was empty")
            return {
                "response": response,
                "model": str(answer.get("model", self._settings.BEDROCK_MODEL_ID)),
                "tokensUsed": self._safe_token_count(answer.get("tokensUsed")),
            }
        except ClientError as exc:
            self._raise_for_aws_error(exc)
        except BotoCoreError as exc:
            logger.exception("Career-advisor generation could not reach Bedrock")
            raise BedrockUnavailableException() from exc
        except AIServiceException:
            raise
        except Exception as exc:
            logger.exception("Career-advisor retrieval or generation failed")
            raise RAGServiceException() from exc

    def _create_chain(
        self,
        vector_store: FAISS,
        client: BedrockRuntimeClient,
    ) -> Runnable[dict[str, Any], dict[str, Any]]:
        """Compose the retriever and Bedrock chat model into a chain."""

        llm = ChatBedrockConverse(
            client=client,
            model=self._settings.BEDROCK_MODEL_ID,
            region_name=self._settings.AWS_REGION,
            max_tokens=4_000,
            temperature=0.2,
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CAREER_ADVISOR_RAG_PROMPT),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", CAREER_ADVISOR_INPUT_PROMPT),
            ]
        )
        document_chain = create_stuff_documents_chain(
            llm,
            prompt,
            output_parser=RAGOutputParser(model_id=self._settings.BEDROCK_MODEL_ID),
        )
        chain = create_retrieval_chain(
            vector_store.as_retriever(search_kwargs={"k": 4}),
            document_chain,
        )
        return cast(Runnable[dict[str, Any], dict[str, Any]], chain)

    @staticmethod
    def _load_documents(knowledge_base_path: Path) -> list[Document]:
        """Read supported UTF-8 text and PDF documents from a directory."""

        if not knowledge_base_path.is_dir():
            logger.warning(
                "Knowledge-base directory does not exist: %s",
                knowledge_base_path,
            )
            return []

        documents: list[Document] = []
        for file_path in sorted(
            path for path in knowledge_base_path.rglob("*") if path.is_file()
        ):
            suffix = file_path.suffix.lower()
            try:
                if suffix in TEXT_FILE_SUFFIXES:
                    text = file_path.read_text(encoding="utf-8-sig")
                    if text.strip():
                        documents.append(
                            Document(
                                page_content=text,
                                metadata={"source": str(file_path)},
                            )
                        )
                elif suffix == PDF_FILE_SUFFIX:
                    documents.extend(PyPDFLoader(str(file_path)).load())
            except (OSError, UnicodeError, ValueError):
                logger.exception(
                    "Skipping unreadable knowledge-base file %s", file_path
                )
        return documents

    @staticmethod
    def _convert_chat_history(
        chat_history: list[dict[str, Any]],
    ) -> list[BaseMessage]:
        """Convert validated API history dictionaries to LangChain messages."""

        messages: list[BaseMessage] = []
        for item in chat_history:
            content = str(item["content"])
            if item["role"] in {"assistant", "ai"}:
                messages.append(AIMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))
        return messages

    @staticmethod
    def _safe_token_count(value: Any) -> int:
        """Normalize an arbitrary token count to a non-negative integer."""

        if isinstance(value, int) and not isinstance(value, bool):
            return max(value, 0)
        return 0

    @staticmethod
    def _raise_for_aws_error(exc: ClientError) -> NoReturn:
        """Translate a LangChain-propagated AWS client error."""

        error_code = str(
            exc.response.get("Error", {}).get("Code", "UnknownClientError")
        )
        logger.exception("Bedrock RAG call returned AWS error %s", error_code)
        if error_code in THROTTLING_ERROR_CODES:
            raise BedrockThrottlingException() from exc
        raise RAGServiceException() from exc


@lru_cache(maxsize=1)
def get_default_rag_service() -> RAGService:
    """Return the process-wide RAG service managed by app lifespan."""

    return RAGService(settings=get_settings())


async def query_career_advisor(query: str) -> dict[str, Any]:
    """Query the lifespan-managed default career-advisor service.

    Args:
        query: User's career question.

    Returns:
        dict[str, Any]: Answer, configured model ID, and token usage.
    """

    return await get_default_rag_service().query_career_advisor(query)
