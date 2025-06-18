import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.utils.data import DataLoader
import numpy as np
import random
import math
import pickle
import math
import os
from args import config_parser
from scipy.spatial.distance import cdist

class QDrugDataset(torch.utils.data.Dataset):
    def __init__(self, args, load_from_cache=False, file_path='data/'):
        self.args = args
        self.atom_dict = {'C': 0, 'N': 1, 'O': 2, 'F': 3, 'S': 4, 'Cl': 5, 'Br': 6, 'I': 7, 'P': 8}
        self.atomic_num_to_type = {6:0, 7:1, 8:2, 9:3}

        if args.dataset == 'qm9':
            self.atom_types = 4

        if load_from_cache:
            self.raw_data = pickle.load(open(file_path + 'raw_information.pkl', 'rb'))
        else:
            if args.dataset == 'qm9':
                mols = Chem.SDMolSupplier(os.path.join(file_path, 'gdb9.sdf'), removeHs=True, sanitize=True)
                molecular_information = []
                for i, mol in enumerate(mols):
                    try:
                        n_atoms = mol.GetNumAtoms()
                        if n_atoms > 9:
                            #print(f"{i} {n_atoms} {Chem.MolToSmiles(mol)}")
                            continue
                        pos = mols.GetItemText(i).split('\n')[4:4+n_atoms]
                        pos = np.array([[float(x) for x in line.split()[:3]] for line in pos])
                        pos = self.determine_position(pos)
                        if np.isnan(pos).any():
                            continue
                        atom_type = np.array([self.atomic_num_to_type[atom.GetAtomicNum()] for atom in mol.GetAtoms()]).reshape(-1, 1)
                        molecular_information.append({'position': pos, 'atom_type': atom_type, 'smi': Chem.MolToSmiles(mol)})
                    except:
                        continue

                self.raw_data = molecular_information
                pickle.dump(molecular_information, open(file_path + 'raw_information.pkl', 'wb'))
                
        position = []
        for item in self.raw_data:
            if np.isnan(item['position']).any():
                print('fuck')
            position.append(item['position'])
        position = np.concatenate(position, axis=0)
        print(f'position.shape: {position.shape}')
        print(f'mean_position: {np.mean(position, axis=0)}')
        min_val = np.min(position, axis=0)
        print(f'min_val: {min_val}')
        max_val = np.max(position, axis=0)
        print(f'max_val: {max_val}')
        diff_minmax = np.max(max_val-min_val)
        print(f'diff_minmax: {diff_minmax}')
        
        if self.args.model_type in ['DrugQAE', 'DrugQDM']:
            self.dataset = np.zeros((len(self.raw_data), 2**self.args.main_qbits + 1))
            self.info = []
            max_len = 0
            for i,item in enumerate(self.raw_data):
                position = (item['position'] - min_val) / diff_minmax
                atom_type = np.eye(self.atom_types)[item['atom_type'].astype(int)].squeeze(1)
                aux_vec = []
                for j in range(len(position)):
                    aux_vec.append(math.sqrt(3 - (position[j][0] ** 2 + position[j][1] ** 2 + position[j][2] ** 2)))
                aux_vec = np.array(aux_vec)
                tmp = np.concatenate((position, atom_type), axis=1).flatten()
                self.dataset[i][:len(tmp)] = tmp
                self.dataset[i][len(tmp):len(position)+len(tmp)] = aux_vec
                self.dataset[i] = self.dataset[i] / (2*math.sqrt(len(position)))
                self.dataset[i][-1] = len(position)
                if len(position) > max_len:
                    max_len = len(position)
                self.info.append({'x': self.dataset[i], 'smi': item['smi']})

        elif self.args.model_type == 'DrugQDM_v2':
            max_feat_len = self.args.max_atoms * (7 + self.args.max_atoms - 1) + self.args.max_atoms
            assert max_feat_len <= 2**self.args.main_qbits
            self.dataset = np.zeros((len(self.raw_data), 2**self.args.main_qbits + 1))
            self.info = []
            
            for i, item in enumerate(self.raw_data):
                n_atoms = len(item['position'])
                if n_atoms == 0 or n_atoms > self.args.max_atoms:
                    continue
                position = (item['position'] - min_val) / diff_minmax
                atom_type = np.eye(self.atom_types)[item['atom_type'].astype(int)].squeeze(1)
                real_dist_matrix = cdist(position, position)
                full_dist_matrix = np.zeros((self.args.max_atoms, self.args.max_atoms))
                full_dist_matrix[:n_atoms, :n_atoms] = real_dist_matrix 
                node_features = np.concatenate((position, atom_type), axis=1)

                molecule_feature_list = []
                for j in range(n_atoms):
                    dist_row = full_dist_matrix[j, :]
                    node_feat = node_features[j, :]
                    dist_part1 = dist_row[:j]
                    dist_part2 = dist_row[j+1:]
                    combined_row = np.concatenate([dist_part1, node_feat, dist_part2])
                    molecule_feature_list.append(combined_row)
                
                if molecule_feature_list:
                    flat_features = np.concatenate(molecule_feature_list)
                else:
                    flat_features = np.array([])
                
                aux_vec_components = np.sqrt(3 - np.sum(position**2, axis=1))
                aux_vec = np.sqrt(2 * n_atoms + 1) * aux_vec_components
                corr_val = np.sqrt(2) * np.sqrt(np.sum(min_val**2)) * n_atoms / diff_minmax
                aux_vec = np.concatenate([aux_vec, corr_val.reshape(-1)])
                final_features = np.concatenate([flat_features, aux_vec])
                
                vec = np.zeros(2**self.args.main_qbits + 1)
                current_total_len = len(final_features)
                vec[:current_total_len] = final_features[:current_total_len]
                norm = np.sqrt(2 * n_atoms * (3 * n_atoms + 2))
                vec[:-1] = vec[:-1] / norm
                vec[-1] = n_atoms
                self.dataset[i] = vec
                self.info.append({'x': self.dataset[i], 'smi': item['smi']})

        self.diff_minmax = diff_minmax
        self.min_val = min_val

    def determine_position(self, x):
        
        def move_point_cloud(points):
            centroid = np.mean(points, axis=0) 
            translated_points = points - centroid  
            return translated_points

        def rotate_point_cloud(points):

            first_point = points[0]

            x, y, z = first_point
            sin_angle_1 = -x / np.sqrt(x ** 2 + z ** 2)
            cos_angle_1 = z / np.sqrt(x ** 2 + z ** 2)

            sin_angle_2 = y / np.sqrt(x ** 2 + y ** 2 + z ** 2)
            cos_angle_2 = np.sqrt(x ** 2 + z ** 2) / np.sqrt(x ** 2 + y ** 2 + z ** 2)

            rotation_matrix = np.array([[cos_angle_1, 0, sin_angle_1],
                                        [sin_angle_2 * sin_angle_1, cos_angle_2, -sin_angle_2 * cos_angle_1],
                                        [-cos_angle_2 * sin_angle_1, sin_angle_2, cos_angle_2 * cos_angle_1]])

            rotated_point_cloud = np.dot(rotation_matrix, points.T).T
            return rotated_point_cloud
        
        x = move_point_cloud(x)
        x = rotate_point_cloud(x)
        return x
            
    def __getitem__(self, index):
        return self.info[index]

    def __len__(self):
        return len(self.dataset)