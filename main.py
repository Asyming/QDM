import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import QDrugDataset
from models.model import DrugQAE
from models.model_op import DrugQDM, PiQDM
from sklearn.metrics import f1_score
import numpy as np
from args import config_parser
from generate import random_generation, diffusion_generation
import random
import warnings
warnings.filterwarnings('ignore')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, args, reduction='mean'):
        super(MultiClassFocalLossWithAlpha, self).__init__()
        self.alpha = torch.tensor(args.atom_weight).to(torch_device)
        self.gamma = args.focal_gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]
        pt = torch.gather(pred, dim=1, index=target.view(-1, 1))
        pt = pt.view(-1)
        ce_loss = -torch.log(pt)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss

torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'torch_device: {torch_device}')

def loss_func_qm9(x, y, num_vertices, focal_loss):
    data_vertices = x[:7*num_vertices].reshape(num_vertices, 7)
    recon_vertices = y[:7*num_vertices].reshape(num_vertices, 7)

    data_atom_type = torch.argmax(data_vertices[:, 3:7], dim=-1)
    recon_atom_type = recon_vertices[:, 3:7]
    constrained_loss = F.mse_loss((recon_atom_type ** 2).sum(dim=1), torch.tensor([0.25 / num_vertices] * num_vertices).to(torch_device))
    recon_atom_type = recon_atom_type ** 2 * num_vertices * 2
    recon_atom_type = recon_atom_type / recon_atom_type.sum(dim=1).unsqueeze(1)
    

    atom_type_loss = focal_loss(recon_atom_type, data_atom_type)
    
    data_xyz = data_vertices[:, :3]
    recon_xyz = recon_vertices[:, :3]
    data_aux = x[7*num_vertices:8*num_vertices]

    recon_aux = y[7*num_vertices:8*num_vertices]
    recon_remain = y[8*num_vertices:]
    xyz_norm_val = torch.linalg.norm(recon_xyz-data_xyz, axis=-1).mean()

    data_atom_type = torch.argmax(data_vertices[:, 3:7], dim=-1)
    recon_atom_type = torch.argmax(recon_vertices[:, 3:7], dim=-1)

    accuracy = (data_atom_type == recon_atom_type).float().mean()
    return xyz_norm_val + torch.abs(recon_aux-data_aux).mean() + torch.abs(recon_remain).mean()+ constrained_loss*100 + atom_type_loss, \
        xyz_norm_val, torch.abs(recon_aux-data_aux).mean(), torch.abs(recon_remain).mean(), constrained_loss, atom_type_loss, accuracy

def fidelity_loss(x, y):
    return 1 - (torch.dot(x, y)) ** 2

def loss_func_qm9_qdm(pred_x0, true_x0, model, num_vertices, focal_loss):
    pred_vertices = pred_x0[:7*num_vertices].reshape(num_vertices, 7)
    true_vertices = true_x0[:7*num_vertices].reshape(num_vertices, 7)
    # print(f"pred_vertices: {pred_vertices}")
    # print(f"true_vertices: {true_vertices}")
    pred_xyz = pred_vertices[:, :3]
    true_xyz = true_vertices[:, :3]
    xyz_loss = F.mse_loss(pred_xyz, true_xyz)
    
    true_atom_type = torch.argmax(true_vertices[:, 3:7], dim=-1)
    pred_atom_type_logits = pred_vertices[:, 3:7]
    
    constrained_loss = F.mse_loss(
        (pred_atom_type_logits ** 2).sum(dim=1), 
        torch.tensor([0.25 / num_vertices] * num_vertices).to(torch_device)
    )
    
    pred_atom_type_normalized = pred_atom_type_logits ** 2 * num_vertices * 2
    pred_atom_type_normalized = pred_atom_type_normalized / (pred_atom_type_normalized.sum(dim=1).unsqueeze(1))
    
    atom_type_loss = focal_loss(pred_atom_type_normalized, true_atom_type)
    
    pred_aux = pred_x0[7*num_vertices:8*num_vertices]
    true_aux = true_x0[7*num_vertices:8*num_vertices]
    aux_loss = F.mse_loss(pred_aux, true_aux)
    
    pred_remain = pred_x0[8*num_vertices:]
    true_remain = true_x0[8*num_vertices:]
    remain_loss = F.l1_loss(pred_remain, true_remain)
    
    pred_atom_type_class = torch.argmax(pred_atom_type_logits, dim=-1)
    accuracy = (true_atom_type == pred_atom_type_class).float().mean()
    
    total_loss = (100 * xyz_loss + aux_loss + remain_loss) + 100 * constrained_loss +  atom_type_loss
    
    return total_loss, 100 * xyz_loss, aux_loss, remain_loss, 100 * constrained_loss, atom_type_loss, accuracy

def _parse_v2_output(x, num_vertices, args):
    """Helper function to parse the flattened output of PiQDM."""
    if num_vertices == 0:
        return None
    
    norm_factor = torch.sqrt(2 * num_vertices * (3 * num_vertices + 2))
    
    num_node_features = 3 + args.atom_types
    block_size = num_node_features + args.max_atoms - 1
        
    all_xyz, all_one_hot, all_distances = [], [], []

    for i in range(num_vertices):
        node_feat_abs_start = i * block_size + i
        
        xyz = x[node_feat_abs_start : node_feat_abs_start + 3]
        one_hot = x[node_feat_abs_start + 3 : node_feat_abs_start + num_node_features]
        
        dist_part1 = x[i * block_size : node_feat_abs_start]
        dist_part2 = x[node_feat_abs_start + num_node_features : (i + 1) * block_size]
        distances = torch.cat([dist_part1, dist_part2])

        all_xyz.append(xyz)
        all_one_hot.append(one_hot)
        all_distances.append(distances)

    all_xyz = torch.stack(all_xyz)
    all_one_hot = torch.stack(all_one_hot) * norm_factor
    all_distances = torch.stack(all_distances)

    main_feat_len = num_vertices * block_size
    aux_feat_len = num_vertices + 1
    
    aux_vec = x[main_feat_len : main_feat_len + aux_feat_len]
    remain_vec = x[main_feat_len + aux_feat_len:]

    return all_xyz, all_one_hot, all_distances, aux_vec, remain_vec

def _permute_v2_vector(x, num_vertices, args, shift):
    """Applies a circular permutation to the active atom blocks of a v2 feature vector."""
    permuted_x = torch.zeros_like(x)
    for i in range(x.shape[0]):
        current_x = x[i]
        block_size = 3 + args.atom_types + args.max_atoms - 1
        part_1 = num_vertices[i] * block_size
        part_2 = num_vertices[i]
        main_features = current_x[:part_1].reshape(num_vertices[i], block_size)
        rolled_main_features = torch.roll(main_features, shifts=shift, dims=0).flatten()
        aux_vec = current_x[part_1:part_1+part_2]
        rolled_aux_vec = torch.roll(aux_vec, shifts=shift)
        permuted_x[i] = torch.cat([rolled_main_features, rolled_aux_vec, current_x[part_1+part_2:]])
    return permuted_x

def loss_func_qm9_qdm_v2(pred_x0, true_x0, num_vertices, args, cls_loss_fn):
    """
    Calculates the reconstruction, focal, and prior losses.
    Permutation loss is handled separately in the training loop.
    """
    assert torch.abs(torch.sum(pred_x0**2) - 1.0) < 1e-4, f'{torch.sum(pred_x0**2)}'
    assert torch.abs(torch.sum(true_x0**2) - 1.0) < 1e-4, f'{torch.sum(true_x0**2)}'
    parsed_pred = _parse_v2_output(pred_x0, num_vertices, args)
    parsed_true = _parse_v2_output(true_x0, num_vertices, args)

    if parsed_pred is None or parsed_true is None:
        zero = torch.tensor(0.0, device=pred_x0.device)
        return zero, zero, zero, torch.tensor(0.0) # recon, focal, prior, acc

    pred_xyz, pred_one_hot, pred_distances, pred_aux, pred_remain = parsed_pred
    true_xyz, true_one_hot, _, true_aux, true_remain = parsed_true

    # 1. Reconstruction Loss (xyz, aux, remain)
    xyz_loss = 100 * F.mse_loss(pred_xyz, true_xyz)
    aux_loss = F.mse_loss(pred_aux, true_aux)
    remain_loss = F.l1_loss(pred_remain, true_remain)
    recon_loss = 100 * (xyz_loss + aux_loss + remain_loss)

    # 2. CE Loss for Classification
    # The model's output for atom types (`pred_one_hot`) has a large and variable scale.
    # Using F.softmax is the most numerically stable way to convert these logits
    # into probabilities for the focal loss, preventing gradient explosion.
    pred_one_hot_probs = F.softmax(pred_one_hot, dim=-1)
    true_atom_types = torch.argmax(true_one_hot, dim=-1)
    
    # Increased the weight of the classification loss to help it compete with other loss terms.
    cls_loss =  10 * cls_loss_fn(pred_one_hot_probs, true_atom_types)

    # For evaluation, argmax on the original logits is still correct.
    f1 = f1_score(
        true_atom_types.cpu().numpy(),
        torch.argmax(pred_one_hot, dim=-1).cpu().numpy(),
        average='macro'
    )


    # 3. Prior Losses
    # The one-hot prior might need re-evaluation later, but let's first stabilize cls_loss.
    # This prior pushes the L1 norm of the scaled logits to be 1.
    one_hot_norm = torch.sum(pred_one_hot, dim=-1)
    one_hot_prior_loss = 0.1 * F.mse_loss(one_hot_norm, torch.full_like(one_hot_norm, 1.0 / args.atom_types))

    # Geometry consistency prior
    recalculated_dist_matrix = torch.cdist(pred_xyz.unsqueeze(0), pred_xyz.unsqueeze(0)).squeeze(0)
    true_recalculated_dists_parts = []
    for i in range(num_vertices):
        dist_row = recalculated_dist_matrix[i]
        true_recalculated_dists_parts.append(torch.cat([dist_row[:i], dist_row[i+1:]]))
    true_recalculated_dists = torch.stack(true_recalculated_dists_parts)
    pred_distances_for_active_atoms = pred_distances[:num_vertices, :(num_vertices-1)]
    geom_consistency_loss = 1000 * F.mse_loss(pred_distances_for_active_atoms, true_recalculated_dists)

    prior_loss = 1 * (one_hot_prior_loss + geom_consistency_loss)

    loss = recon_loss + cls_loss + prior_loss
    
    return loss, xyz_loss, aux_loss, remain_loss, recon_loss, cls_loss, one_hot_prior_loss, geom_consistency_loss, prior_loss, f1 #accuracy

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'total parameters: {total_params:,}')
    return total_params

def main():
    setup_seed(0)

    parser = config_parser()
    args = parser.parse_args()
    
    if args.dataset == 'qm9':
        args.atom_types = 4
        # atom_counts calculated from QM9 dataset
        # atom_counts = torch.tensor([842155., 139551., 186959., 3311.])
        # # weights = 1. / atom_counts
        # # weights = weights / weights.sum()
        # args.atom_weight = weights.tolist()
        args.atom_weight = [1.05, 0.10, 0.10, 0.75] # C, N, O, F
        print(f'Using QM9 dataset with {args.atom_types} atom types.')
    elif args.dataset == 'geom_drugs':
        args.atom_types = 9
        # Using uniform weights for GEOM-Drugs as a starting point
        # args.atom_weight = [1.0 / args.atom_types] * args.atom_types
        args.atom_weight = [0.04, 0.02, 0.02, 0.05, 0.20, 0.02, 0.05, 0.10, 0.50] # C, N, O, F, P, S, Cl, Br, I
        print(f'Using GEOM-Drugs dataset with {args.atom_types} atom types.')
    
    dataset = QDrugDataset(args, True)

    min_val, diff_minmax = dataset.min_val, dataset.diff_minmax

    generator = torch.Generator()
    generator.manual_seed(0)
    
    train_dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=torch.utils.data.RandomSampler(dataset, generator=generator),
    )

    # Select model based on model_type
    if args.model_type == 'DrugQAE':
        save_folder = f'./save/{args.dataset}_results/{args.model_type}/{args.model_type}_qubits{args.n_qubits}_blocks{args.n_blocks}_kappa{args.kappa}_lr{args.lr}_useMLP_{args.use_MLP}_threshold{args.threshold}_maxAtoms{args.max_atoms}/'
        model = DrugQAE(args, n_qbits=args.n_qbits, n_blocks=args.n_blocks, bottleneck_qbits=args.bottleneck_qbits).to(torch_device)
        count_parameters(model)
    elif args.model_type == 'DrugQDM':
        save_folder = f'./save/{args.dataset}_results/{args.model_type}/{args.model_type}_exp{args.exp_id}_qubits{args.n_qubits}_blocks{args.n_blocks}_lr{args.lr}_useMLP_{args.use_MLP}_threshold{args.threshold}_maxAtoms{args.max_atoms}/'
        model = DrugQDM(args, n_qubits=args.n_qbits, n_blocks=args.n_blocks).to(torch_device)
        count_parameters(model)
    elif args.model_type == 'PiQDM':
        save_folder = f'./save/{args.dataset}_results/{args.model_type}/{args.model_type}_exp{args.exp_id}_qubits{args.n_qubits}_blocks{args.n_blocks}_lr{args.lr}_useMLP_{args.use_MLP}_threshold{args.threshold}_timesteps{args.timesteps}_maxAtoms{args.max_atoms}/'
        model = PiQDM(args, n_qubits=args.n_qubits, n_blocks=args.n_blocks).to(torch_device)
        count_parameters(model)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    print(f'n_qubits: {args.n_qubits}, n_blocks: {args.n_blocks}')
    model.to(torch_device)
    #cls_loss_fn = nn.CrossEntropyLoss()
    cls_loss_fn = MultiClassFocalLossWithAlpha(args)
    #cls_loss_fn = nn.NLLLoss(weight=weights.to(torch_device))
    focal_loss = MultiClassFocalLossWithAlpha(args)

    losses = []
    epochs = args.num_epochs
    if args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, 
                                   momentum=0.9)

    for epoch in range(epochs):
        f = open(save_folder + f"res.txt", 'a')
        running_loss = 0.0
        batches = 0
        
        model.train()

        if args.dataset == 'qm9':
            if args.model_type == 'DrugQAE':
                loss_func = loss_func_qm9
            elif args.model_type == 'DrugQDM':
                loss_func = loss_func_qm9_qdm
            elif args.model_type == 'PiQDM':
                loss_func = loss_func_qm9_qdm_v2
        
        for batch_idx, batch_dict in enumerate(train_dl):
            if args.model_type == 'DrugQAE':
                x_whole, smi = batch_dict['x'], batch_dict['smi']
                x = x_whole[:, :-1].float().to(torch_device)
                num_vertices = x_whole[:, -1].to(torch_device).int()
                measure, reconstruct = model(x)

                reconstruct = reconstruct.to(torch_device)

                loss, xyz_norm_val, atom_type_loss, aux_loss, remain_loss, constrained_loss, acc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                for i in range(x.shape[0]):
                    loss_, xyz_norm_val_, aux_loss_, \
                        remain_loss_, constrained_loss_, atom_type_loss_,\
                            acc_ = loss_func(x[i], reconstruct[i], num_vertices[i], focal_loss)
                    loss += loss_
                    xyz_norm_val += xyz_norm_val_
                    aux_loss += aux_loss_
                    remain_loss += remain_loss_
                    constrained_loss += constrained_loss_
                    atom_type_loss += atom_type_loss_
                    acc += acc_
                
                sys.stdout.write(
                    f"\r{batches + 1} / {len(train_dl)}, "
                    f"loss:{(loss / x.shape[0]): .6f}, "
                    f"{(xyz_norm_val / x.shape[0]): .6f}, "
                    f"{(aux_loss / x.shape[0]): .6f}, "
                    f"{(remain_loss / x.shape[0]): .6f}, "
                    f"{(constrained_loss / x.shape[0]): .6f}, "
                    f"{(atom_type_loss / x.shape[0]): .6f}, "
                    f"Acc:{acc / x.shape[0]: .6f}"
                )
            
            elif args.model_type == 'DrugQDM':
                x_whole, smi = batch_dict['x'], batch_dict['smi']
                x = x_whole[:, :-1].float().to(torch_device)
                num_vertices = x_whole[:, -1].to(torch_device).int()
                # x, smi = batch_dict['x'], batch_dict['smi']
                batch_size = x.shape[0]
                
                t_indices = torch.randint(0, args.timesteps, (batch_size,)).to(torch_device)
                x_noisy, _ = model.q_sample(x, t_indices)
                predicted_x0 = model(x_noisy, t_indices)
                
                loss, xyz_loss, aux_loss, remain_loss, constrained_loss, atom_type_loss, acc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                
                for i in range(batch_size):
                    loss_, xyz_loss_, aux_loss_, remain_loss_, constrained_loss_, atom_type_loss_, acc_ = loss_func_qm9_qdm(
                        predicted_x0[i], x[i], model, num_vertices[i], focal_loss)
                    loss += loss_
                    xyz_loss += xyz_loss_
                    aux_loss += aux_loss_
                    remain_loss += remain_loss_
                    constrained_loss += constrained_loss_
                    atom_type_loss += atom_type_loss_
                    acc += acc_
                
                sys.stdout.write(
                    f"\r{batches + 1} / {len(train_dl)}, "
                    f"loss:{loss / batch_size:.6f}, "
                    f"xyz:{xyz_loss / batch_size:.6f}, "
                    f"aux:{aux_loss / batch_size:.6f}, "
                    f"remain:{remain_loss / batch_size:.6f}, "
                    f"constrained:{constrained_loss / batch_size:.6f}, "
                    f"atom:{atom_type_loss / batch_size:.6f}, "
                    f"Acc:{acc / batch_size:.6f}"
                )
            
            elif args.model_type == 'PiQDM':
                x_whole, smi = batch_dict['x'], batch_dict['smi']
                x = x_whole[:, :-1].float().to(torch_device)
                # print(x[0])
                num_vertices = x_whole[:, -1].to(torch_device).int()
                batch_size = x.shape[0]
                
                t_indices = torch.randint(0, args.timesteps, (batch_size,)).to(torch_device)
                x_noisy, _ = model.q_sample(x, t_indices)
                predicted_x0 = model(x_noisy, t_indices)
                # test perm loss in qm9
                if args.dataset == 'qm9':
                    perm_loss = torch.tensor(0.0, device=x.device)
                    avg_perm_loss_ = 0.0
                    # # Perform n-1 permutations for n atoms
                    # for shift in range(1, num_vertices.max()):
                    #     permuted_noisy_x0 = _permute_v2_vector(x_noisy, num_vertices, args, shift)
                    #     perm_pred_x0 = model(permuted_noisy_x0, t_indices)
                    #     perm_loss += F.mse_loss(perm_pred_x0, x)
                    # avg_perm_loss_ = 1000 * perm_loss / (num_vertices.max() - 1)
                
                loss, xyz_loss, aux_loss, remain_loss, recon_loss, cls_loss, one_hot_prior_loss, geom_consistency_loss, prior_loss, f1 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

                for i in range(batch_size):
                    loss_, xyz_loss_, aux_loss_, remain_loss_, recon_loss_, cls_loss_, one_hot_prior_loss_, geom_consistency_loss_, prior_loss_, f1_ = loss_func_qm9_qdm_v2(
                        predicted_x0[i], x[i], num_vertices[i], args, cls_loss_fn
                    )

                    
                    loss += loss_
                    xyz_loss += xyz_loss_
                    aux_loss += aux_loss_
                    remain_loss += remain_loss_
                    recon_loss += recon_loss_
                    cls_loss += cls_loss_
                    one_hot_prior_loss += one_hot_prior_loss_
                    geom_consistency_loss += geom_consistency_loss_
                    prior_loss += prior_loss_
                    perm_loss += avg_perm_loss_
                    f1 += f1_
                
                sys.stdout.write(
                    f"\r{batches + 1} / {len(train_dl)}, "
                    f"loss:{(loss/batch_size):.4f} | "
                    f"xyz:{(xyz_loss / batch_size):.4f}, "
                    f"aux:{(aux_loss / batch_size):.4f}, "
                    f"remain:{(remain_loss / batch_size):.4f}, "
                    f"recon:{(recon_loss / batch_size):.4f}, "
                    f"cls:{(cls_loss / batch_size):.4f}, "
                    f"oh:{(one_hot_prior_loss / batch_size):.4f}, "
                    f"geom:{(geom_consistency_loss / batch_size):.4f}, "
                    f"prior:{(prior_loss / batch_size):.4f}, "
                    f"perm:{(perm_loss / batch_size):.4f}, "
                    f"f1:{(f1 / batch_size):.4f}"
                )
            
            sys.stdout.flush()

            loss /= batch_size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batches += 1
            
            if (batch_idx + 1) % args.eval_interval == 0:
                sys.stdout.write('\n')
                print(f"Epoch {epoch}, Batch {batch_idx+1}: Generating molecules for evaluation...")
                
                model.eval()
                if args.model_type == 'DrugQAE':
                    random_generation(model, args, generate_num=5, f=f, debug=True)
                elif args.model_type == 'DrugQDM':
                    diffusion_generation(model, args, generate_num=5, f=f, debug=True)
                elif args.model_type == 'PiQDM':
                    diffusion_generation(model, args, generate_num=5, f=f, debug=True)
                    
                model.train()

        losses.append(running_loss / batches)
        print(f"\nEpoch {epoch} loss: {losses[-1]}")
        f.write(f"Epoch {epoch} loss: {losses[-1]}\n")
        torch.save(model.state_dict(), save_folder + f"model_{epoch}.pth")
        model.eval()
        if args.model_type == 'DrugQAE':
            setup_seed(0)
            random_generation(model, args, f=f, debug=True)
        elif args.model_type == 'DrugQDM':
            setup_seed(0)
            diffusion_generation(model, args, f=f, debug=True)
        elif args.model_type == 'PiQDM':
            setup_seed(0)
            diffusion_generation(model, args, f=f, debug=True)

if __name__ == '__main__':
    main()