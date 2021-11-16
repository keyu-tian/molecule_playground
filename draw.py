from rdkit.Chem import Draw
from rdkit import Chem

smis = [
    'COC1=C(C=CC(=C1)NS(=O)(=O)C)C2=CN=CN3C2=CC=C3',
    'C1=CC2=C(C(=C1)C3=CN=CN4C3=CC=C4)ON=C2C5=CC=C(C=C5)F',
    'COC(=O)C1=CC2=CC=CN2C=N1',
    'C1=C2C=C(N=CN2C(=C1)Cl)C(=O)O',
]
mols = []
for smi in smis:
    mol = Chem.MolFromSmiles(smi)
    mols.append(mol)
img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(200, 200), legends=['' for x in mols])

import matplotlib.pyplot as plt
plt.imshow(img)
plt.show()
