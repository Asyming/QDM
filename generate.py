import torch
from models.model import *
from models.Cmodel import *
import numpy as np
import sys
from args import config_parser
import math
from utils.eval_validity import *
from utils.eval_property import *
torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def reconstruct_func(x, args):
    for i in range(args.max_atoms+1):
        if x[7*i+3:7*i+7].sum() <= args.threshold:
            break
    num_vertices = i

    if num_vertices == 0 or num_vertices ==1:
        return None, None

    x = x.detach().cpu().numpy()
    min_val = np.array([-3.92718444, -4.54352093, -5.10957203])
    diff_minmax = 10.36
    atomic_type_dict = {0:6, 1:7, 2:8, 3:9, 4:16, 5:17, 6:36, 7:53, 8:54}
    atom_type = np.zeros((num_vertices), dtype=int)
    position = np.zeros((num_vertices, 3))
    
    for i in range(num_vertices):
        atom_type[i] = np.argmax(x[i*7+3:i*7+7])
        atom_type[i] = atomic_type_dict[atom_type[i]]
        if args.model_type == 'DrugQAE':
            position[i] = x[i*7:i*7+3] * 2 * math.sqrt(len(position)) * diff_minmax + min_val
        elif args.model_type == 'DrugQDM':
            position[i] = x[i*7:i*7+3] *  diff_minmax + min_val

    return atom_type, position

# def reconstruct_func_qdm(x, args):
#     for i in range(args.max_atoms):
#         if x[7*i+3:7*i+7].sum() <= args.threshold:
#             break
#     num_vertices = i

#     if num_vertices == 0 or num_vertices ==1:
#         return None, None

#     x = x.detach().cpu().numpy()
#     min_val = np.array([-3.92718444, -4.54352093, -5.10957203])
#     diff_minmax = 10.36
#     atomic_type_dict = {0:6, 1:7, 2:8, 3:9, 4:16, 5:17, 6:36, 7:53, 8:54}
#     atom_type = np.zeros((num_vertices), dtype=int)
#     position = np.zeros((num_vertices, 3))
    
#     for i in range(num_vertices):
#         atom_type[i] = np.argmax(x[i*7+3:i*7+7])
#         atom_type[i] = atomic_type_dict[atom_type[i]]
#         position[i] = x[i*7:i*7+3] * diff_minmax + min_val

#     return atom_type, position
    
def random_generation(model, args, generate_num = None, f = None, debug = False):
    generate_num = args.generate_num if generate_num is None else generate_num
    atom_type, positions = [], []
    for i in range(generate_num):
        sys.stdout.write(f'\rgenerating {i+1}/{generate_num}')
        sys.stdout.flush()
        z_sample = np.abs(np.random.randn(1, 2**(args.n_qbits - args.bottleneck_qbits)))
        z_sample = torch.tensor(z_sample).to(torch_device)
        z_sample /= (z_sample**2).sum()**0.5
        x_decoded = model.generate(z_sample)
        atom_type_one, position_one = reconstruct_func(x_decoded[0], args)
        if atom_type_one is not None:
            atom_type.append(atom_type_one)
            positions.append(position_one)

    print(f"generated {len(atom_type)} molecules")

    drug_dict = {'number':atom_type, 'position': positions}
    if debug and len(drug_dict['number']) > 0:
        print(drug_dict['number'][0])
        print(drug_dict['position'][0])
    print(f"\ncalculating metric...")
    valid_ratio, unique_ratio, novel_ratio, bond_mmds, _, _ = calculate_metric(drug_dict, debug)
    print(f"validity: {valid_ratio}, unique:{unique_ratio}, novelty:{novel_ratio}")
    if f is not None:
        f.write(f"validity: {valid_ratio}, unique:{unique_ratio}, novelty:{novel_ratio}\n")
    if valid_ratio == 0:
        return print('no valid molecules')
    if not debug:
        for key, value in bond_mmds.items():
            print(f" bond mmd {key}: {value}")
            if f is not None:
                f.write(f" bond mmd {key}: {value}\n")

def diffusion_generation(model, args, generate_num = None, f = None, debug = False):
    """Generate molecules using the diffusion model"""
    generate_num = args.generate_num if generate_num is None else generate_num
    atom_type, positions = [], []
    print(f"Starting DDIM sampling with {args.ddim_steps} steps...")
    samples = model.sample_ddim(generate_num, ddim_steps=args.ddim_steps, ddim_eta=args.ddim_eta)
    for i in range(generate_num):
        sys.stdout.write(f'\rprocessing {i+1}/{generate_num}')
        sys.stdout.flush()
        
        x_decoded = samples[i:i+1]
        atom_type_one, position_one = reconstruct_func(x_decoded[0], args)
        if atom_type_one is not None:
            atom_type.append(atom_type_one)
            positions.append(position_one)
            
    print(f"generated {len(atom_type)} molecules")
    drug_dict = {'number':atom_type, 'position': positions}
    if debug and len(drug_dict['number']) > 0:
        print(drug_dict['number'][0])
        print(drug_dict['position'][0])
    print(f"\ncalculating metric...")
    valid_ratio, unique_ratio, novel_ratio, bond_mmds, _, _ = calculate_metric(drug_dict, debug)
    print(f"validity: {valid_ratio}, unique:{unique_ratio}, novelty:{novel_ratio}")
    if f is not None:
        f.write(f"validity: {valid_ratio}, unique:{unique_ratio}, novelty:{novel_ratio}\n")
    if valid_ratio == 0:
        return print('no valid molecules')
    if not debug:
        for key, value in bond_mmds.items():
            print(f" bond mmd {key}: {value}")
            if f is not None:
                f.write(f" bond mmd {key}: {value}\n")

if __name__ == '__main__':
    setup_seed(0)
    args = config_parser().parse_args()
    
    if args.model_type == 'DrugQAE':
        model = DrugQAE(args, n_qbits=args.n_qbits, n_blocks=args.n_blocks).to(torch_device)
        exp_name = '20.0'
        raw_params_dict = torch.load(f'save/qm9_results/DrugQAE_qubits7_blocks10_kappa{exp_name}_lr0.001_useMLP_False_threshold0.2_maxAtoms9/model_10.pth')
        if 'qdev.states' in raw_params_dict:
                raw_params_dict['qdev.states'] = torch.zeros((1,2,2,2,2,2,2,2), dtype=torch.complex64)
        model.load_state_dict(raw_params_dict)
        model.eval()
        random_generation(model, args, debug=True)
    elif args.model_type == 'DrugQDM':
        exp_name = ['122.0','123.0']
        lr = ['0.0001','0.0001']
        i = args.exp_id
        model = DrugQDM(args, n_qbits=args.n_qbits, n_blocks=args.n_blocks).to(torch_device)
        for model_id in range(0, 100, 9):
            try:
                raw_params_dict = torch.load(f'save/qm9_results/DrugQDM_qubits7_blocks{args.n_blocks}_kappa{exp_name[i]}_lr{lr[i]}_useMLP_False_threshold0.0_maxAtoms9/model_{model_id}.pth')
                print(f'loading model {model_id}')
            except:
                print(f'model {model_id} not found')
                continue
            if 'qdev.states' in raw_params_dict:
                raw_params_dict['qdev.states'] = torch.zeros((1,2,2,2,2,2,2,2), dtype=torch.complex64)
                model.qdev.states = torch.zeros((1,2,2,2,2,2,2,2), dtype=torch.complex64)
            model.load_state_dict(raw_params_dict)
            model.eval()
            diffusion_generation(model, args, debug=True)