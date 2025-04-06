Z_MATRIX_GENERATION_PROMPT_HEADER = """
Below is a description of a bulk material where each atom is described by its \
element type and three attributes: 1. distance to the previous atom, 2. angle \
to the previous two atoms, 3. dihedral angle to the previous three atoms. The \
first three Fm atoms are dummies that help define the rest of the material. \
<attributes> Generate a description of the lengths and angles of the lattice \
vectors and the three dummy Fm atoms, followed by the element type and the \
three attributes for each atom within the lattice:
"""

CARTESIAN_GENERATION_PROMPT_HEADER = """
Below is a description of a bulk material. <attributes> Generate a description \
of the lengths and angles of the lattice vectors and then the element type and \
coordinates for each atom within the lattice:
"""

DISTANCE_MATRIX_GENERATION_PROMPT_HEADER = """
Below is a description of a bulk material where each atom is described by its \
element type and distances to the preceding atoms. <attributes> \
Generate a description of the lengths and angles of the lattice vectors, \
followed by the element type and distances for each atom within the lattice, \
ensuring that each atom solely references distances to preceding atoms, \
resembling the lower triangular portion of a distance matrix:
"""

SLICES_GENERATION_PROMPT_HEADER = """
Below is a description of a bulk material. <attributes> Generate a SLICES string, \
which is a text-based representation of a crystal material:
"""

CIF_GENERATION_PROMPT_HEADER = """I have a material and its band gap value. A band gap is the distance \
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

```python
<dictionary_output>
```
"""

CIF_INSTRUCT_PROMPT_HEADER_INPUT = """I have a material and its band gap value. A band gap is the distance \
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
CIF_INSTRUCT_PROMPT_HEADER_OUTPUT = """

```python
<dictionary_output>
```
""" 