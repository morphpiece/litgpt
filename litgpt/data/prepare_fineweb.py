# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import os
import time
import traceback
from pathlib import Path
from typing import Literal, Union
from lightning_utilities.core.imports import RequirementCache
from tqdm import tqdm
from litgpt import Tokenizer
from morphpiece import MorphPiece
from litgpt.utils import CLI

_LITDATA_AVAILABLE = RequirementCache("litdata")
if _LITDATA_AVAILABLE:
    from litdata.processing.data_processor import DataChunkRecipe
else:
    DataChunkRecipe = object


class FineWebDataRecipe(DataChunkRecipe):
    is_generator = True
    
    def __init__(self, tokenizer: Union[Tokenizer,MorphPiece], chunk_size: int):
        super().__init__(chunk_size)
        
        self.tokenizer = tokenizer
        self.tokenizer_type = "morph" if isinstance(tokenizer, MorphPiece) else "llama"

    def prepare_structure(self, input_dir):
        files = Path(input_dir).rglob("*.parquet")
        return [str(file) for file in files]

    def prepare_item(self, item_metadata):
        import pyarrow.parquet as pq

        filepath = item_metadata
        start = time.time()

        try:
            parquet_file = pq.ParquetFile(filepath)
            # reduce RAM usage
            for batch in tqdm(parquet_file.iter_batches(batch_size=8192, columns=["text"])):
                for text in batch.to_pandas()["text"]:
                    if self.tokenizer_type == "morph":
                        yield self.tokenizer.encode(text)
                    else:
                        yield self.tokenizer.encode(text, bos=False, eos=True)

        except Exception:
            print(traceback.format_exc())
            print(f"Error reading {filepath}")
            return

        parquet_file.close()
        end = time.time()
        print(f"Took {end - start:.2f} seconds total", filepath)


def prepare(
    input_dir: Path,
    output_dir: Path,
    tokenizer_path: Path,
    tokenizer_type: Literal["llama", "morph"] = "llama",
    num_workers: int = -1,
    chunk_size: int = (2049 * 8192),
    fast_dev_run: bool = False,
) -> None:
    from litdata.processing.data_processor import DataProcessor
    

    if tokenizer_type == "llama":
        print('Using Llama tokenizer from ', tokenizer_path)
        tokenizer = Tokenizer(tokenizer_path)
    else:
        print('Using MorphPiece tokenizer from ', tokenizer_path)
        tokenizer = MorphPiece(tokenizer_path)
        
    num_workers = num_workers if num_workers > 0 else os.cpu_count()
        
    data_recipe = FineWebDataRecipe(tokenizer=tokenizer, chunk_size=chunk_size)
    data_processor = DataProcessor(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
        num_workers=num_workers,
        num_downloaders=1,
    )

    start_time = time.time()
    data_processor.run(data_recipe)
    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    CLI(prepare)
