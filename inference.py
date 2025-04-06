import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torchtune

MODEL_PATH = "/home/hice1/athalanki3/scratch/LLM4StructGen/model"

ADAPTER_PATH = "/home/hice1/athalanki3/scratch/LLM4StructGen/instruct/lora-LLAMA3-cif-/epoch_2"
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH,device_map="auto",torch_dtype=torch.float16)
print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
print("Loading Adapter...")
# adapter = torchtune.Adapter.FromCheckpoint(ADAPTER_PATH)

# adapter.apply(model)

# model.load_adapter(ADAPTER_PATH)
print("Done loading everything and applying adapter.")
input_text = """I have a material and its band gap value. A band gap is the distance \
between the valence band of electrons and the conduction band, \
representing the minimum energy that is required to excite an electron to the conduction band. \
The material is represented by the lattice lengths, lattice angles, followed by \
the atomic species and their fractional coordinates in the unit cell. 

Material:
<material_cif>

Band gap:
<band_gap>

Please propose a modification to the material that results in a band gap of around 1.4 eV. \
You can choose one of the four following modifications:
1. exchange: exchange two atoms in the material
2. substitute: substitute one atom in the material with another
3. remove: remove an atom from the material
4. add: add an atom to the material

Your output should be a python dictionary of the following the format: {Modification: [$TYPE, $ATOM_1, $ATOM_2]}. Here are the requirements:
1. $TYPE should be the modification type; one of "exchange", "substitute", "remove", "add"
2. $ATOM should be the selected atom to be modified. For "exchange" and "substitute", two $ATOM placeholders are needed. For "remove" and "add", one $ATOM placeholder is needed.
3. $ATOM should be the element name with its index. For example: Na1.
4. For "add", $ATOM index does not need to be specified.
5. For "subsitute", $ATOM_1 needs to be indexed while $ATOM_2 does not need to be indexed.
"""
inputs = tokenizer(input_text,return_tensors="pt").to(model.device)

#Might be missing some masking or whatever.

with torch.inference_mode():
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

decoded_output = tokenizer.decode(outputs[0],skip_special_tokens=True)
print("Output: ",decoded_output)