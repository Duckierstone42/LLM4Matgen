

import os
import json
import copy
import glob
import torch
import random
import numpy as np
from pathlib import Path
import pandas as pd
from typing import Any, Dict, List, Mapping, Optional
import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from torchtune.data import truncate
from torchtune.datasets._packed import PackedDataset
from torchtune.modules.tokenizers import ModelTokenizer
from llm4structgen.datasets.prompts import CIF_GENERATION_PROMPT_HEADER

prompt_lookup = {
    "formation_energy_per_atom": "The formation energy per atom is",
    "band_gap": "The band gap is",
    "e_above_hull": "The energy above the convex hull is",
    "spacegroup_number": "The spacegroup number is",
}



class TextCompletionMPDataset(Dataset):
    def __init__(
        self,
        tokenizer: ModelTokenizer,
        data_files: str,
        initial_bandgap_file: str,
        cif_files: str,
        max_seq_len: Optional[int] = None,
        add_eos: bool = True,
        attributes: Any = False,
        translate: bool = False,
        rotate: bool = False,
        permute: bool = False,
        decimals: int = 2,
        duplicate_count: int = 1,
    ) -> None:
        self._tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.add_eos = add_eos
        self.attributes = attributes
        self.duplicate_count = duplicate_count
        self.cif_files = cif_files
        
        # self._data = load_dataset(source, data_files=data_files)
        #How will i load datat
        self.mp_to_bandgap = self.load_initial_bandgap(initial_bandgap_file)
        self.data = self._load_data(data_files)

        # No encoder needed, data already in cif format
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        #Getting from data will give me mp_name, mod_type, mod_atom, initial_bandgap, modified_band_gap

        #Use cif fil
        datum = self.data[idx]
        return self._prepare_sample(datum)

    def load_initial_bandgap(self,bandgap_file):
        hashmap = {}
        df = pd.read_csv(bandgap_file,header=None)
        for index, row in df.iterrows():
            hashmap[row[0]] = row[2]
        return hashmap

    def _load_data(self,dataset_path):
        dataset_path = Path(dataset_path)
        data = []
        for file in dataset_path.iterdir():
            #Also need to record file_path.
            mol_name = file.name.split("_")[0]
            initial_bandgap = self.mp_to_bandgap[mol_name]
            #Now, how to go from mol_name to actual name?
            df = pd.read_csv(file,header=None)
            for index, row in df.iterrows():
                
                band_gap = row[1]
                mod = row[0]
                elem = [mol_name,mod,initial_bandgap,band_gap]
                data.append(elem)

        return data

    def prepare_prompt(self,sample):
        #Sample is of format -> ["mvc-13180","exchange","Pr2","Pr1","1.7","1.4"]

        cif_file = Path(sample[0] + ".cif")
        cif_path = self.cif_files / cif_file
        cif_str = cif_path.read_text()
        initial_bandgap = sample[2]
        prompt_header = CIF_GENERATION_PROMPT_HEADER.replace("<material_cif>",cif_str)
        prompt_header = prompt_header.replace("<band_gap>",str(initial_bandgap))

        dict_output = {"Modification" : sample[1]}

        prompt_header = prompt_header.replace("<dictionary_output>",str(dict_output))
        return prompt_header


    def _prepare_sample(self, sample) -> Dict[str, List[int]]:
        prompt = self.prepare_prompt(sample)

        tokens = self._tokenizer.encode(text=prompt, add_bos=True, add_eos=self.add_eos)

        # Truncate if needed, but don't coerce EOS id
        if self._tokenizer.max_seq_len is not None:
            tokens = truncate(tokens, self._tokenizer.max_seq_len - 1)

        # No need to offset labels by 1 - happens in the recipe
        labels = tokens.copy()

        return {"tokens": tokens, "labels": labels}


def text_completion_dataset(
    tokenizer: ModelTokenizer,
    data_files: str,
    initial_bandgap: str,
    cif_files: str,
    max_seq_len: Optional[int] = None,
    add_eos: bool = True,
    packed: bool = False,
    attributes: Any = False,
    translate: bool = False,
    rotate: bool = False,
    permute: bool = False,
    decimals: int = 2,
    duplicate_count: int = 1,
) -> TextCompletionMPDataset:
    """
    Build a configurable dataset from a freeform, unstructured text corpus similar
    to datasets used in pre-training. This method should be
    used to configure a custom text dataset from the yaml config instead of
    using :class:`~torchtune.datasets.TextCompletionDataset` directly, as it is made to be config friendly.
    """
    ds = TextCompletionMPDataset(
        tokenizer=tokenizer,
        data_files=data_files,
        initial_bandgap_file=initial_bandgap,
        cif_files = cif_files,
        max_seq_len=max_seq_len,
        add_eos=add_eos,
        attributes=attributes,
        translate=translate,
        rotate=rotate,
        permute=permute,
        decimals=decimals,
        duplicate_count=duplicate_count
    )

    return (
        PackedDataset(ds, max_seq_len=max_seq_len, padding_idx=tokenizer.pad_id)
        if packed
        else ds
    )


