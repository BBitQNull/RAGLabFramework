import tomllib
import argparse
from typing import Any, Optional
from dataclasses import dataclass
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_core.runnables import RunnableConfig
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.embeddings import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pathlib import Path, PurePath
import torch
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


# config dataclass model
@dataclass
class Dataset:
    dataset_path_name: str
    dataset_path_dir: Path


@dataclass
class EncoderModel:
    encoder_model_name: str


@dataclass
class RetrievalDatabase:
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
    dataset: Dataset
    encoder_model: EncoderModel
    retrieval_database: RetrievalDatabase
    retriever: RetrieverSetting


def get_device() -> str:
    """get device cuda/mps/cpu"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_config(config: argparse.Namespace) -> TOMLConfig:
    """Parse TOML files"""
    with open(file=config.config, mode="rb") as f:
        tomlConfig: dict[str, Any] = tomllib.load(f)
    return TOMLConfig(
        dataset=Dataset(
            dataset_path_name=tomlConfig["Dataset"]["dataset_path_name"],
            dataset_path_dir=tomlConfig["Dataset"]["dataset_path_dir"],
        ),
        encoder_model=EncoderModel(
            encoder_model_name=tomlConfig["EncoderModel"]["encoder_model_name"]
        ),
        retrieval_database=RetrievalDatabase(
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
            search_type=tomlConfig.get("Retriever", {}).get("search_type", 4),
            search_kwargs={
                "k": tomlConfig.get("Retriever", {}).get("k", 4),
                "score_threshold": tomlConfig.get("Retriever", {}).get(
                    "score_threshold", 0.8
                ),
                "fetch_k": tomlConfig.get("Retriever", {}).get("fetch_k", 20),
                "lambda_mult": tomlConfig.get("Retriever", {}).get("lambda_mult", 0.5),
                "filter": tomlConfig.get("Retriever", {}).get("filter", None),
            },
        ),
    )


class LLModel:
    """A large language model"""

    pass


class RDatabase:
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


class Retriever:
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


def main():
    # argparse
    parser = argparse.ArgumentParser(description="Hello ragforensics")
    parser.add_argument("--config", "-C", default="./config.toml")
    args: argparse.Namespace = parser.parse_args()

    # parse toml config
    config: TOMLConfig = load_config(config=args)

    # init device
    device: str = get_device()
    # create retrieval database instance
    retrieval_database: Chroma = RDatabase(
        retrieval_database_path=str(
            object=Path(
                TOMLConfig.retrieval_database.retrieval_database_dir,
                TOMLConfig.retrieval_database.retrieval_database_name,
            )
        ),
        device=device,
        dataset_path_dir=TOMLConfig.dataset.dataset_path_dir,
        dataset_path_name=TOMLConfig.dataset.dataset_path_name,
        split_method=TOMLConfig.retrieval_database.split_method,
        encoder_model_name=TOMLConfig.encoder_model.encoder_model_name,
        dataset_bench_size=TOMLConfig.retrieval_database.dataset_bench_size,
        chunk_size=TOMLConfig.retrieval_database.chunk_size,
        chunk_overlap=TOMLConfig.retrieval_database.chunk_overlap,
    ).retrieval_database


if __name__ == "__main__":
    main()
