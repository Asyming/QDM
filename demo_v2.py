import torch
import numpy as np
from args import config_parser
from dataset import QDrugDataset

def main():
    """
    This script demonstrates the difference in data representation for the same molecule
    between the 'DrugQDM' and 'DrugQDM_v2' model types.
    """
    # --- Setup arguments from parser ---
    parser = config_parser()
    args = parser.parse_args()
    
    # --- Hardcode some arguments for a consistent demonstration ---
    args.dataset = 'qm9'
    args.main_qbits = 8  # Provides a vector length of 256, which fits max_atoms=9 (9*9=81) or even more.
    args.max_atoms = 9   # As per the logic in dataset.py for qm9
    
    # Set numpy print options for better readability of the vectors
    np.set_printoptions(linewidth=120, precision=4, suppress=True)

    # --- Method 1: Original 'DrugQDM' data construction ---
    print("="*60)
    print("1. Generating data using the method for 'DrugQDM'")
    print("="*60)
    args.model_type = 'DrugQDM'
    dataset_v1 = QDrugDataset(args, load_from_cache=True, file_path='data/')
    sample_v1 = dataset_v1[3000]
    vector_v1 = sample_v1['x']
    smi = sample_v1['smi']
    
    print(f"Molecule SMILES: {smi}")
    print(f"Vector shape for 'DrugQDM': {vector_v1.shape}")
    print(f"Number of atoms (stored in last element): {vector_v1[-1]}")
    print("\nFull feature vector for 'DrugQDM':")
    print(vector_v1)
    
    # Calculate and print sum of squares
    sum_sq_v1 = np.sum(vector_v1[:-1]**2)
    print(f"\nSum of squares for 'DrugQDM' feature vector: {sum_sq_v1:.6f}")
    
    # --- Method 2: New 'DrugQDM_v2' data construction ---
    print("\n" + "="*60)
    print("2. Generating data using the method for 'DrugQDM_v2'")
    print("="*60)
    args.model_type = 'DrugQDM_v2'
    dataset_v2 = QDrugDataset(args, load_from_cache=True, file_path='data/')
    sample_v2 = dataset_v2[3000]
    vector_v2 = sample_v2['x']
    
    # Sanity check to ensure we are comparing the same molecule
    assert smi == sample_v2['smi'], "Mismatch in molecules between the two datasets!"

    print(f"Molecule SMILES: {smi}")
    print(f"Vector shape for 'DrugQDM_v2': {vector_v2.shape}")
    print(f"Number of atoms (stored in last element): {vector_v2[-1]}")
    print("\nFull feature vector for 'DrugQDM_v2' (flattened adjacency matrix):")
    print(vector_v2)
    
    # Calculate and print sum of squares
    sum_sq_v2 = np.sum(vector_v2[:-1]**2)
    print(f"\nSum of squares for 'DrugQDM_v2' feature vector: {sum_sq_v2:.6f}")
    
    print("\n" + "="*60)
    print("Demonstration complete.")
    print("="*60)


if __name__ == '__main__':
    main() 