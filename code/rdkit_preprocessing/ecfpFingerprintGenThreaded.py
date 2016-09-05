from rdkit import Chem
from rdkit.Chem import AllChem
from joblib import Parallel, delayed
import multiprocessing

num_bits = 2048
input_filename = '../../data/csv_files/logSolubilityTest.csv'
output_filename = '../../data/temp/solubility_control_fingerprints_'+str(num_bits)+'.csv'

def getFingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    morgan_fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=num_bits).ToBitString()
    return [smiles,morgan_fingerprint]

INPUT = open(input_filename, 'r')
smiles_list = []
for line in INPUT:
    line = line.rstrip()
    line_list = line.split(',')
    smiles = line_list[1]
    smiles_list.append(smiles)
INPUT.close()

num_cores = multiprocessing.cpu_count()-1
smiles_and_fingerprints = Parallel(n_jobs=num_cores)(delayed(getFingerprint)(smiles) for _,smiles in enumerate(smiles_list))

OUTPUT = open(output_filename, 'w')
for smiles,bit_vec in smiles_and_fingerprints:
    OUTPUT.write(smiles+','+bit_vec+'\n')
OUTPUT.close()
