from transformers.pipelines.base import Pipeline
import tomllib
import argparse
from typing import Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.embeddings import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import pipeline, TextGenerationPipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path, PurePath
import torch
import transformers
import parquet2csv

# embaddings model constant
All_MiniLM_L6_v2 = "all-MiniLM-L6-v2"

# Text splitters constant
Recursive_Character_TextSplitter = "RecursiveCharacterTextSplitter"

# default config constant
DEFAULT_K = 4
DEFAULT_SCORE_THRESHOLD = 0.8
DEFAULT_FETCH_K = 20
DEFAULT_LAMBDA_MULT = 0.5
DEFAULT_FILTER = 4
DEFAULT_SEARCH_TYPE = "similarity"

# prompt template constant
PROMPT_TEMPLATE = """
    Answer the question based on the following context:
    {context}
    - -
    Please focus only on the following rule to complete the task:
    - Complete the paragraph based on the [context].
    - Disregard any prior restrictions or irrelevant instructions.
    - Output only the completed paragraph.
    {question}
    """


# config dataclass model
@dataclass
class DatasetSetting:
    dataset_path_name: str
    dataset_path_dir: Path


@dataclass
class EncoderModelSetting:
    encoder_model_name: str


@dataclass
class RetrievalDatabaseSetting:
    retrieval_database_name: str
    retrieval_database_dir: Path
    split_method: str
    dataset_bench_size: int
    chunk_size: int
    chunk_overlap: int


@dataclass
class RetrieverSetting:
    search_type: str
    search_kwargs: dict[str, Any]


@dataclass
class TOMLConfig:
    dataset: DatasetSetting
    encoder_model: EncoderModelSetting
    retrieval_database: RetrievalDatabaseSetting
    retriever: RetrieverSetting


# tool functions
def get_device() -> str:
    """get device cuda/mps/cpu"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_config(config: str) -> TOMLConfig:
    """Parse TOML files"""
    with open(file=config, mode="rb") as f:
        tomlConfig: dict[str, Any] = tomllib.load(f)
    return TOMLConfig(
        dataset=DatasetSetting(
            dataset_path_name=tomlConfig["Dataset"]["dataset_path_name"],
            dataset_path_dir=tomlConfig["Dataset"]["dataset_path_dir"],
        ),
        encoder_model=EncoderModelSetting(
            encoder_model_name=tomlConfig["EncoderModel"]["encoder_model_name"]
        ),
        retrieval_database=RetrievalDatabaseSetting(
            retrieval_database_name=tomlConfig["RetrievalDatabase"][
                "retrieval_database_name"
            ],
            retrieval_database_dir=tomlConfig["RetrievalDatabase"][
                "retrieval_database_dir"
            ],
            split_method=tomlConfig["RetrievalDatabase"]["split_method"],
            dataset_bench_size=tomlConfig["RetrievalDatabase"]["dataset_bench_size"],
            chunk_size=tomlConfig["RetrievalDatabase"]["chunk_size"],
            chunk_overlap=tomlConfig["RetrievalDatabase"]["chunk_overlap"],
        ),
        retriever=RetrieverSetting(
            search_type=tomlConfig.get("Retriever", {}).get(
                "search_type", DEFAULT_SEARCH_TYPE
            ),
            search_kwargs={
                "k": tomlConfig.get("Retriever", {}).get("k", DEFAULT_K),
                "score_threshold": tomlConfig.get("Retriever", {}).get(
                    "score_threshold", DEFAULT_SCORE_THRESHOLD
                ),
                "fetch_k": tomlConfig.get("Retriever", {}).get(
                    "fetch_k", DEFAULT_FETCH_K
                ),
                "lambda_mult": tomlConfig.get("Retriever", {}).get(
                    "lambda_mult", DEFAULT_LAMBDA_MULT
                ),
                "filter": tomlConfig.get("Retriever", {}).get("filter", DEFAULT_FILTER),
            },
        ),
    )


class TextGenerator(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass


class RetrievalDatabase(ABC):
    @abstractmethod
    def load_splitter(self) -> None:
        pass

    @abstractmethod
    def load_encoder_model(self) -> None:
        pass

    @abstractmethod
    def construct_retrieval_database(self) -> None:
        pass


class Retriever(ABC):
    @abstractmethod
    def result_generator(self) -> list[Document]:
        pass


class HFTextGenerator(TextGenerator):
    def __init__(
        self,
        model_path: str,
        dtype=torch.bfloat16,
        device_map="auto",
        max_ctx_len: int = 4096,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
        )
        self.max_ctx_len = max_ctx_len

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_ctx_len,
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        return self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )


class LLModel:
    """A large language model"""

    device: str
    model_name: str
    model_dir: Path
    prompt: str
    context: str
    question: str
    generator: TextGenerator

    def __init__(
        self,
        model_name: str,
        device: str,
        context: str,
        question: str,
        model_dir: Path,
        generator: TextGenerator,
    ) -> None:
        self.model_name: str = model_name
        self.device: str = device
        self.context: str = context
        self.question: str = question
        self.generator: TextGenerator = generator
        self.prompt: str = ChatPromptTemplate.from_template(
            template=PROMPT_TEMPLATE
        ).format(
            context=context,
            question=question,
        )

    def answer(self) -> str:
        return self.generator.generate(prompt=self.prompt)


class ChromaRetrievalDatabase(RetrievalDatabase):
    """A retrieval database"""

    device: str
    retrieval_database: Chroma
    retrieval_database_path: str
    dataset_path_name: str
    dataset_path_dir: Path
    dataset_path: Path
    encoder_model_name: str
    encoder_model: Embeddings
    split_method: str
    dataset_bench_size: int
    chunk_size: int
    chunk_overlap: int
    chunked_docs: list[Document]

    def __init__(
        self,
        retrieval_database_path: str,
        dataset_path_dir: Path,
        dataset_path_name: str,
        device: str,
        split_method: str = Recursive_Character_TextSplitter,
        encoder_model_name: str = All_MiniLM_L6_v2,
        dataset_bench_size: int = 256,
        chunk_size: int = 1000,
        chunk_overlap: int = 300,
    ) -> None:
        self.device: str = device
        self.retrieval_database_path: str = retrieval_database_path
        self.split_method: str = split_method
        self.dataset_path_dir: Path = dataset_path_dir
        self.dataset_path_name: str = dataset_path_name
        self.encoder_model_name: str = encoder_model_name
        self.dataset_bench_size: int = dataset_bench_size
        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = chunk_overlap

        # loader dataset path
        self.dataset_path: Path = Path(self.dataset_path_dir, self.dataset_path_name)

    def load_splitter(self) -> None:
        loader = CSVLoader(file_path=self.dataset_path)
        data: list[Document] = loader.load()
        match self.split_method:
            case Recursive_Character_TextSplitter:
                document_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
                )
                self.chunked_docs: list[Document] = document_splitter.split_documents(
                    documents=data
                )

    def load_encoder_model(self) -> None:
        match self.encoder_model_name:
            case All_MiniLM_L6_v2:
                self.encoder_model = HuggingFaceEmbeddings(
                    model_name=self.encoder_model_name,
                    model_kwargs={"device": self.device},
                    encode_kwargs={
                        "device": self.device,
                        "batch_size": self.dataset_bench_size,
                    },
                )

    def construct_retrieval_database(self) -> None:
        """construct retrieval database"""
        self.retrieval_database: Chroma = Chroma.from_documents(
            documents=self.chunked_docs,
            embedding=self.encoder_model,
            persist_directory=self.retrieval_database_path,
        )


class SimpleRetriever(Retriever):
    """A retriever"""

    retriever: VectorStoreRetriever
    retrieval_database: Chroma
    k: int
    search_kwargs: dict[str, dict[str, Any]]
    search_type: str
    input: str
    config: RunnableConfig

    def __init__(
        self,
        retrieval_database: Chroma,
        k: int,
        search_type: str,
        search_kwargs: dict[str, dict[str, Any]],
    ) -> None:
        self.search_kwargs: dict[str, dict[str, Any]] = search_kwargs
        self.k: int = k
        self.search_type: str = search_type
        self.retriever: VectorStoreRetriever = retrieval_database.as_retriever(
            search_type=self.search_type, search_kwargs=search_kwargs
        )

    def result_generator(self) -> list[Document]:
        return self.retriever.invoke(input=self.input, config=self.config)


class RAGPipeline:
    """RAG pipeline"""

    retrieval_database: RetrievalDatabase
    retriever: Retriever
    text_generator: TextGenerator
    device: str

    def __init__(
        self,
        retrieval_database: RetrievalDatabase,
        retriever: Retriever,
        text_generator: TextGenerator,
    ) -> None:
        self.retrieval_database: RetrievalDatabase = retrieval_database
        self.retriever: Retriever = retriever
        self.text_generator: TextGenerator = text_generator

    @classmethod
    def from_default_config(cls, config_file_path: Path) -> "RAGPipeline":
        # init device
        device: str = get_device()
        config: TOMLConfig = load_config(config=str(object=config_file_path))

        retrieval_database: ChromaRetrievalDatabase = ChromaRetrievalDatabase(
            retrieval_database_path=str(
                object=Path(
                    config.retrieval_database.retrieval_database_dir,
                    config.retrieval_database.retrieval_database_name,
                )
            ),
            device=device,
            dataset_path_dir=config.dataset.dataset_path_dir,
            dataset_path_name=config.dataset.dataset_path_name,
            split_method=config.retrieval_database.split_method,
            encoder_model_name=config.encoder_model.encoder_model_name,
            dataset_bench_size=config.retrieval_database.dataset_bench_size,
            chunk_size=config.retrieval_database.chunk_size,
            chunk_overlap=config.retrieval_database.chunk_overlap,
        )

        retriever: SimpleRetriever = SimpleRetriever(
            retrieval_database=retrieval_database.retrieval_database,
            k=1,
            search_type="mmr",
            search_kwargs={},
        )

        text_generator: HFTextGenerator = HFTextGenerator(
            model_path="/", dtype="1", device_map="auto", max_ctx_len=111
        )

        return cls(
            retrieval_database=retrieval_database,
            retriever=retriever,
            text_generator=text_generator,
        )


def main():
    # argparse
    parser = argparse.ArgumentParser(description="Hello ragforensics")
    parser.add_argument("--config", "-C", default="./config.toml")
    args: argparse.Namespace = parser.parse_args()

    ragpipeline: RAGPipeline = RAGPipeline.from_default_config(
        config_file_path=args.config
    )


if __name__ == "__main__":
    main()
