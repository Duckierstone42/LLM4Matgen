from typing import Any, Callable, Dict, Optional, Union

from torchtune.data import InputOutputToMessages
from torchtune.datasets._packed import PackedDataset
from torchtune.datasets._sft import SFTDataset
from torchtune.modules.transforms.tokenizers import ModelTokenizer
from torchtune.data import truncate
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
from datasets import load_dataset
from torch.utils.data import Dataset
from llm4structgen.datasets.prompts import CIF_INSTRUCT_PROMPT_HEADER_INPUT,CIF_INSTRUCT_PROMPT_HEADER_OUTPUT


class TextInstructionMPDataset(Dataset):
    def __init__(
        self,
        tokenizer: ModelTokenizer,
        message_transform: InputOutputToMessages,
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
        self.message_transform = message_transform
        # self._data = load_dataset(source, data_files=data_files)
        #How will i load datat
        self.mp_to_bandgap = self.load_initial_bandgap(initial_bandgap_file)
        self.data = self._load_data(data_files)

       
    def __len__(self):
        return min(len(self.data),100000)
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

            #Randomly chooses one from each of the top5
            rand_index = random.randint(0,len(df)-1)
            row = df.iloc[rand_index]
                
            band_gap = row[1]
            mod = row[0]
            elem = [mol_name,mod,initial_bandgap,band_gap]
            data.append(elem)

        return data

    def prepare_input_prompt(self,sample):
        #Sample is of format -> ["mvc-13180","exchange","Pr2","Pr1","1.7","1.4"]
        #Use reduced cif format

        cif_file = Path(sample[0] + ".cif")
        cif_path = self.cif_files / cif_file
        cif_str = cif_path.read_text()
        initial_bandgap = sample[2]
        prompt_header = CIF_INSTRUCT_PROMPT_HEADER_INPUT.replace("<material_cif>",cif_str)
        prompt_header = prompt_header.replace("<band_gap>",str(initial_bandgap))


        return prompt_header

    def prepare_output_prompt(self,sample):
        dict_output = {"Modification" : sample[1]}

        output = CIF_INSTRUCT_PROMPT_HEADER_OUTPUT.replace("<dictionary_output>",str(dict_output))

        return output

    def _prepare_sample(self, sample) -> Dict[str, List[int]]:
        input_text = self.prepare_input_prompt(sample)
        output_text = self.prepare_output_prompt(sample)
        sample_dict = {"input": input_text, "output": output_text}
        messages = self.message_transform(sample_dict)["messages"]
        # print(messages)
        #tokens = self._tokenizer.encode(text=prompt, add_bos=True, add_eos=self.add_eos)
        tokenized = self._tokenizer.tokenize_messages(messages)
        # print("Tokens: ",tokenized)
        tokens = tokenized[0]
        labels = tokenized[1]

        # Truncate if needed, but don't coerce EOS id
        if self._tokenizer.max_seq_len is not None:
            tokens = truncate(tokens, self._tokenizer.max_seq_len - 1)
            labels = truncate(labels,self._tokenizer.max_seq_len - 1)


        return {"tokens": tokens, "labels": labels}
    

def instruct_dataset(
    tokenizer: ModelTokenizer,
    data_files: str,
    initial_bandgap: str,
    cif_files: str,
    max_seq_len: Optional[int] = None,
    add_eos: bool = True,
    attributes: Any = False,
    translate: bool = False,
    rotate: bool = False,
    permute: bool = False,
    decimals: int = 2,
    duplicate_count: int = 1,
    column_map: Optional[Dict[str, str]] = None,
    train_on_input: bool = False,
    new_system_prompt: Optional[str] = None,
    packed: bool = False,
) -> Union[SFTDataset, PackedDataset]:
    """
    Configure a custom dataset with user instruction prompts and model responses.

    This builder function can be used to configure a custom instruct dataset directly from the yaml config
    as an alternative to :class:`~torchtune.datasets.SFTDataset`, as it is made to be config friendly.

    The dataset should follow this format:

    .. code-block:: text

        |  input          |  output          |
        |-----------------|------------------|
        | "user prompt"   | "model response" |

    If your column names are different, you can use the ``column_map`` parameter to change
    the expected column names. For example, if your dataset has columns ``"question"`` and
    ``"answer"`` you can use::

        column_map = {"input": "question", "output": "answer"}

    Masking of the prompt during training is controlled by the ``train_on_input`` flag, which is
    set to ``False`` by default
    - If ``train_on_input`` is True, the prompt is used during training and
    contributes to the loss.
    - If ``train_on_input`` is False, the prompt is masked out (tokens replaced with -100)

    Examples:

    ::

        my_dataset.json
        [
            {
                "question": "What time is it in London?",
                "answer": "It is 10:00 AM in London.",
            },
            {
                ...
            },
            ...,
        ]

    ::

        >>> from torchtune.datasets import instruct_dataset
        >>> dataset = instruct_dataset(
        ...     tokenizer=tokenizer,
        ...     source="json",
        ...     data_files="my_dataset.json",
        ...     column_map={
        ...         "input": "question",
        ...         "output": "answer",
        ...     },
        ...     train_on_input=False,
        ...     packed=False,
        ...     split="train",
        ... )
        >>> tokens = dataset[0]["tokens"]
        >>> tokenizer.decode(tokens)
        "What time is it in London?It is 10:00 AM in London."

    This can also be accomplished via the yaml config:

    .. code-block:: yaml

        dataset:
          _component_: torchtune.datasets.instruct_dataset
          source: json
          data_files: my_dataset.json
          column_map:
            input: question
            output: answer
          train_on_input: False
          packed: False
          split: train

    Returns:
        Union[SFTDataset, PackedDataset]: the configured :class:`~torchtune.datasets.SFTDataset`
            or :class:`~torchtune.datasets.PackedDataset` if ``packed=True``

    Raises:
        ValueError: If ``packed=True`` and ``tokenizer.max_seq_len`` is not set.
    """
    message_transform = InputOutputToMessages(
        train_on_input=train_on_input,
        column_map=column_map,
        new_system_prompt=new_system_prompt,
    )

    # ds = SFTDataset(
    #     source=source,
    #     message_transform=message_transform,
    #     model_transform=tokenizer,
    #     filter_fn=filter_fn,
    #     split=split,
    #     **load_dataset_kwargs,
    # )
    ds = TextInstructionMPDataset(
        tokenizer=tokenizer,
        message_transform=message_transform,
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

    if packed:
        if tokenizer.max_seq_len is None:
            raise ValueError(
                "PackedDataset requires a max_seq_len to be set on the tokenizer."
            )
        return PackedDataset(ds, max_seq_len=tokenizer.max_seq_len)
    return ds