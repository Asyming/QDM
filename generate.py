import torch
from models.model import DrugQAE
from models.model_op import DrugQDM, DrugQDM_v2
from models.Cmodel import *
import numpy as np
import sys
from args import config_parser
import math
from utils.eval_validity import *
from utils.eval_property import *
import gc
torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def reconstruct_func(x, args):
    if args.model_type == 'DrugQAE' or args.model_type == 'DrugQDM':
        for i in range(args.max_atoms + 1):
            if i >= args.max_atoms:
                num_vertices = args.max_atoms
                break
            if x[7*i+3:7*i+7].sum() <= args.threshold:
                break
        num_vertices = i

        if num_vertices == 0 or num_vertices == 1:
            return None, None

        x = x.detach().cpu().numpy()
        min_val = np.array([-3.92718444, -4.54352093, -5.10957203])
        diff_minmax = 10.36
        atomic_type_dict = {0:6, 1:7, 2:8, 3:9}
        atom_type = np.zeros((num_vertices), dtype=int)
        position = np.zeros((num_vertices, 3))
        
        for i in range(num_vertices):
            atom_type[i] = np.argmax(x[i*7+3:i*7+7])
            atom_type[i] = atomic_type_dict[atom_type[i]]
            # The scaling factor is an empirical choice from the original repo
            # to attempt to reverse the normalization.
            norm_factor = 2 * math.sqrt(num_vertices)
            position[i] = x[i*7:i*7+3] * norm_factor * diff_minmax + min_val
        
        return atom_type, position

    elif args.model_type == 'DrugQDM_v2':
        block_size = args.max_atoms + 6
        for i in range(args.max_atoms + 1):
            if i >= args.max_atoms:
                num_vertices = args.max_atoms
                break
            # Calculate where the 7D node feature starts for atom i
            node_feat_abs_start = i * block_size + i
            # Get the one-hot part
            one_hot_part = x[node_feat_abs_start + 3 : node_feat_abs_start + 7]
            # Use the same thresholding logic as the V1 models
            if one_hot_part.sum() <= args.threshold:
                break
        num_vertices = i
        
        if num_vertices == 0 or num_vertices == 1:
            return None, None

        x = x.detach().cpu().numpy()
        min_val = np.array([-3.92718444, -4.54352093, -5.10957203])
        diff_minmax = 10.36
        atomic_type_dict = {0:6, 1:7, 2:8, 3:9}
        atom_type = np.zeros((num_vertices), dtype=int)
        position = np.zeros((num_vertices, 3))

        for i in range(num_vertices):
            node_feat_abs_start = i * block_size + i
            node_feat = x[node_feat_abs_start : node_feat_abs_start + 7]
            
            current_pos = node_feat[:3]
            current_type_logits = node_feat[3:7]

            atom_type[i] = np.argmax(current_type_logits)
            atom_type[i] = atomic_type_dict[atom_type[i]]

            # Denormalization based on the formula used in dataset.py for v2
            norm = 2 * num_vertices #np.sqrt(2 * num_vertices * (3 * num_vertices + 2))
            position[i] = current_pos * norm * diff_minmax + min_val
            
        return atom_type, position

def random_generation(model, args, generate_num = None, f = None, debug = False):
    generate_num = args.generate_num if generate_num is None else generate_num
    atom_type, positions = [], []
    
    np.random.seed(0)
    
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
    setup_seed(42)
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    args = config_parser().parse_args()
    
    if args.model_type == 'DrugQAE':
        model = DrugQAE(args, n_qbits=args.n_qbits, n_blocks=args.n_blocks).to(torch_device)
        exp_name = '0.1'
        for i in range(0, 1):
            # raw_params_dict = torch.load(f'save/qm9_results/DrugQAE_qubits7_blocks10_kappa{exp_name}_lr0.01_useMLP_False_threshold0.2025_maxAtoms10/model_{i}.pth')
            raw_params_dict = torch.load(f'/data3/lihan/projects/zisen/ageaware/experiments/qm/checkpoint/model.pth')
            if 'qdev.states' in raw_params_dict:
                raw_params_dict['qdev.states'] = torch.zeros((1,2,2,2,2,2,2,2), dtype=torch.complex64)
            model.load_state_dict(raw_params_dict)
            model.eval()
            random_generation(model, args, debug=True)
    elif args.model_type == 'DrugQDM':
        exp_name = ['128.0','124.0']
        lr = ['0.001','0.0005']
        i = args.exp_id
        model = DrugQDM(args, n_qbits=args.n_qbits, n_blocks=args.n_blocks).to(torch_device)
        for model_id in [70]:
            try:
                raw_params_dict = torch.load(f'save/qm9_results/DrugQDM_qubits8_blocks{args.n_blocks}_kappa{exp_name[i]}_lr{lr[i]}_useMLP_False_threshold0.0_maxAtoms10/model_{model_id}.pth')
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

    elif args.model_type == 'DrugQDM_v2':
        exp_name = ['127.0','128.0','129.0']
        lr = ['0.0001','0.0005','0.0005']
        n_blocks = [5,5,1]
        i = args.exp_id
        model = DrugQDM_v2(args, n_qbits=args.n_qbits, n_blocks=args.n_blocks).to(torch_device)
        for model_id in range(4,5):
            try:
                raw_params_dict = torch.load(f'save/qm9_results/DrugQDM_v2_qubits9_blocks{n_blocks[i]}_kappa{exp_name[i]}_lr{lr[i]}_useMLP_False_threshold0.0_maxAtoms9/model_{model_id}.pth')
                print(f'loading model {model_id}')
            except:
                print(f'model {model_id} not found')
                continue
            if 'qdev_up.states' in raw_params_dict:
                raw_params_dict['qdev_up.states'] = torch.zeros((1,2,2,2,2,2,2,2,2,2,2), dtype=torch.complex64)
                model.qdev_up.states = torch.zeros((1,2,2,2,2,2,2,2,2,2,2), dtype=torch.complex64)
            if 'qdev1.states' in raw_params_dict:
                raw_params_dict['qdev1.states'] = torch.zeros((1,2,2,2,2,2,2,2,2,2), dtype=torch.complex64)
                model.qdev1.states = torch.zeros((1,2,2,2,2,2,2,2,2,2), dtype=torch.complex64)
            if 'qdev2.states' in raw_params_dict:
                raw_params_dict['qdev2.states'] = torch.zeros((1,2,2,2,2,2,2,2,2), dtype=torch.complex64)
                model.qdev2.states = torch.zeros((1,2,2,2,2,2,2,2,2), dtype=torch.complex64)
            if 'qdev3.states' in raw_params_dict:
                raw_params_dict['qdev3.states'] = torch.zeros((1,2,2,2,2,2,2,2), dtype=torch.complex64)
                model.qdev3.states = torch.zeros((1,2,2,2,2,2,2,2), dtype=torch.complex64)
            model.load_state_dict(raw_params_dict)
            model.eval()
            diffusion_generation(model, args, debug=True)