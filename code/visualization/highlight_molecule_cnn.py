from rdkit import Chem
#from rdkit.Chem.Draw import IPythonConsole
#from IPython.display import SVG
import time
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib


def moltosvg(mol,molSize=(1000,1000),kekulize=False):
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    return svg.replace('svg:','')


def moltosvg_highlight(mol, atom_list, atom_predictions, \
    molSize=(1000,1000),kekulize=False):

    min_pred = np.amin(atom_predictions)
    max_pred = np.amax(atom_predictions)


    norm = matplotlib.colors.Normalize(vmin=min_pred,vmax=max_pred)
    cmap = cm.get_cmap('plasma')


    plt_colors = cm.ScalarMappable(norm=norm,cmap=cmap)

    atom_colors = {}
    for i,atom in enumerate(atom_list):
        color_rgba = plt_colors.to_rgba(atom_predictions[i])
        atom_rgb = color_rgba #(color_rgba[0],color_rgba[1],color_rgba[2])
        atom_colors[atom] = atom_rgb

    rdDepictor.Compute2DCoords(mol)

    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
    #drawer.DrawMolecule(mc)
    drawer.DrawMolecule(mol,highlightAtoms=atom_list,highlightBonds=[],
        highlightAtomColors=atom_colors)

    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    # It seems that the svg renderer used doesn't quite hit the spec.
    # Here are some fixes to make it work in the notebook, although I think
    # the underlying issue needs to be resolved at the generation step
    return svg.replace('svg:','')



INPUT = open('../../data/example_output/solubility_predictions_cnn.csv', 'r')
for line in INPUT:
    line = line.rstrip()
    line_list = line.split(',')
    smiles = line_list[0]
    molecule_prediction = line_list[1]
    atoms_list = line_list[2].split(':')
    atom_predictions = line_list[3].split(':')

    #convert these to floats and ints, respectively
    atom_predictions = [float(x) for x in atom_predictions]
    atom_list = [int(x) for x in atoms_list]


    mol = Chem.MolFromSmiles(smiles)
    mol_svg = moltosvg_highlight(mol, atom_list, atom_predictions)

    #get rid of SMILES characters taht will mess up saving the file
    smiles_out = smiles.replace('/', '')
    smiles_out = smiles_out.replace('\/','')

    OUTPUT = open('cnn_molecules/'+smiles_out+'_pred.svg', 'w')
    OUTPUT.write(mol_svg)
    OUTPUT.close()

INPUT.close()
