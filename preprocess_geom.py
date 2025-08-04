import msgpack
import os
import numpy as np
import argparse
import pickle
from tqdm import tqdm

def move_point_cloud(points):
    """Translates a point cloud so that its centroid is at the origin."""
    if points.shape[0] == 0:
        return points
    centroid = np.mean(points, axis=0)
    translated_points = points - centroid
    return translated_points

def rotate_point_cloud(points):
    """Rotates a point cloud to a canonical orientation."""
    if points.shape[0] == 0:
        return points

    # Use the first point to define the rotation.
    # This is a deterministic way to align the molecule.
    first_point = points[0]
    x, y, z = first_point

    # Avoid division by zero if the point is on an axis
    if np.isclose(x, 0) and np.isclose(z, 0):
        return points # Already aligned along y-axis or at origin

    # Rotation around y-axis to bring the point to the y-z plane
    cos_angle_1 = z / np.sqrt(x**2 + z**2) if not np.isclose(x**2 + z**2, 0) else 1.0
    sin_angle_1 = -x / np.sqrt(x**2 + z**2) if not np.isclose(x**2 + z**2, 0) else 0.0
    
    # Rotation around x-axis to bring the point to the z-axis
    new_y = y
    new_z = np.sqrt(x**2 + z**2)
    cos_angle_2 = new_z / np.sqrt(new_y**2 + new_z**2) if not np.isclose(new_y**2 + new_z**2, 0) else 1.0
    sin_angle_2 = new_y / np.sqrt(new_y**2 + new_z**2) if not np.isclose(new_y**2 + new_z**2, 0) else 0.0

    # Combined rotation matrix
    rotation_matrix = np.array([
        [cos_angle_1, 0, sin_angle_1],
        [sin_angle_1 * sin_angle_2, cos_angle_2, -cos_angle_1 * sin_angle_2],
        [-sin_angle_1 * cos_angle_2, sin_angle_2, cos_angle_1 * cos_angle_2]
    ])

    rotated_point_cloud = np.dot(points, rotation_matrix.T)
    return rotated_point_cloud

def standardize_positions(positions):
    """Applies centering and rotation to a molecule's coordinates."""
    pos = move_point_cloud(positions)
    pos = rotate_point_cloud(pos)
    return pos


def extract_and_process_conformers(args):
    """
    Extracts conformers from msgpack, filters, standardizes, and saves them.
    """
    drugs_file = os.path.join(args.data_dir, args.data_file)
    output_file = f'geom_drugs_processed_C{args.conformations}_h_{args.remove_h}.pkl'
    save_file_path = os.path.join(args.data_dir, output_file)
    
    # Atomic numbers of rare elements to be filtered out
    forbidden_atoms = {5, 13, 14, 33, 80, 83}

    unpacker = msgpack.Unpacker(open(drugs_file, "rb"), raw=False)

    processed_molecules = []
    total_mols_in_file = 0
    filtered_mols_count = 0
    total_conformers_saved = 0

    print(f"Starting processing of {drugs_file}...")
    for drugs_1k in tqdm(unpacker, desc="Unpacking msgpack chunks"):
        for smiles, all_info in drugs_1k.items():
            total_mols_in_file += 1

            conformers = all_info.get('conformers', [])
            if not conformers:
                continue

            # Check for forbidden atoms using the first conformer
            first_conf_xyz = np.array(conformers[0]['xyz'])
            atomic_nums = first_conf_xyz[:, 0].astype(int)
            
            if any(atom_num in forbidden_atoms for atom_num in atomic_nums):
                filtered_mols_count += 1
                continue  # Skip this molecule entirely

            # Get energies and keep only the lowest N conformers
            all_energies = np.array([conf['totalenergy'] for conf in conformers])
            # Handle cases with fewer conformers than requested
            num_to_keep = min(len(all_energies), args.conformations)
            lowest_energy_indices = np.argsort(all_energies)[:num_to_keep]

            for idx in lowest_energy_indices:
                conformer = conformers[idx]
                
                coords_with_atom_num = np.array(conformer['xyz']).astype(float)
                
                positions = coords_with_atom_num[:, 1:]
                current_atomic_nums = coords_with_atom_num[:, 0].astype(int)

                if args.remove_h:
                    mask = (current_atomic_nums != 1)
                    positions = positions[mask]
                    current_atomic_nums = current_atomic_nums[mask]
                
                if positions.shape[0] == 0:
                    continue

                # Standardize: center and rotate the coordinates
                standardized_pos = standardize_positions(positions.copy())

                if np.isnan(standardized_pos).any():
                    print(f"Warning: NaN detected in standardized positions for SMILES {smiles}. Skipping conformer.")
                    continue

                processed_molecules.append({
                    'position': standardized_pos.astype(np.float32),
                    'atom_type': current_atomic_nums, # Save raw atomic numbers
                    'smi': smiles
                })
                total_conformers_saved += 1
    
    # --- Final Report ---
    print("\n--- Preprocessing Complete ---")
    print(f"Total unique molecules in source file: {total_mols_in_file}")
    print(f"Molecules filtered out (containing B, Al, Si, As, Hg, Bi): {filtered_mols_count}")
    print(f"Molecules kept: {total_mols_in_file - filtered_mols_count}")
    print(f"Total conformers processed and saved: {total_conformers_saved}")
    
    # Save the processed data as a pickle file
    print(f"\nSaving data to {save_file_path}...")
    with open(save_file_path, 'wb') as f:
        pickle.dump(processed_molecules, f)
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--conformations", type=int, default=30,
                        help="Max number of conformations to keep for each molecule.")
    parser.add_argument("--remove_h", action='store_true', 
                        help="Remove hydrogens from the dataset.")
    parser.add_argument("--data_dir", type=str, default='data/',
                        help="Directory containing the data files.")
    parser.add_argument("--data_file", type=str, default="drugs_crude.msgpack",
                        help="Name of the msgpack file.")
    # parser.add_argument("--output_file", type=str, default="geom_drugs_processed.pkl",
    #                     help="Name of the output pickle file.")
    args = parser.parse_args()
    extract_and_process_conformers(args)