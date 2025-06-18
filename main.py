import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import QDrugDataset
from models.model import DrugQAE
from models.model_op import DrugQDM, DrugQDM_v2
import numpy as np
from args import config_parser
from generate import random_generation, diffusion_generation
import random

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
        """
        :param alpha: list of weight coefficients for each class, where in a 3-class problem,
        the weight for class 0 is 0.2, class 1 is 0.3, and class 2 is 0.5.
        :param gamma: coefficient for hard example mining
        :param reduction:
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        self.alpha = torch.tensor(args.atom_weight).to(torch_device)
        self.gamma = args.focal_gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]  # for each sample in the current batch, assign weight to each class, shape=(bs), one-dimensional vector
        pt = torch.gather(pred, dim=1, index=target.view(-1, 1))  # extract the log_softmax value at the position of class label for each sample, shape=(bs, 1)
        pt = pt.view(-1)  # reduce dimension, shape=(bs)
        ce_loss = -torch.log(pt)  # take the negative of the log_softmax, which is the cross entropy loss
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # calculate focal loss according to the formula and obtain the loss value for each sample, shape=(bs)
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
    """Helper function to parse the flattened output of DrugQDM_v2."""
    if num_vertices == 0:
        return None
    
    norm_factor = torch.sqrt(2 * num_vertices * (3 * num_vertices + 2))
        
    block_size = args.max_atoms + 6  # 7 features + max_atoms-1 distances
    
    all_xyz, all_one_hot, all_distances = [], [], []

    for i in range(num_vertices):
        node_feat_abs_start = i * block_size + i
        
        xyz = x[node_feat_abs_start : node_feat_abs_start + 3]
        one_hot = x[node_feat_abs_start + 3 : node_feat_abs_start + 7]
        
        dist_part1 = x[i * block_size : node_feat_abs_start]
        dist_part2 = x[node_feat_abs_start + 7 : (i + 1) * block_size]
        distances = torch.cat([dist_part1, dist_part2])

        all_xyz.append(xyz)
        all_one_hot.append(one_hot)
        all_distances.append(distances)

    all_xyz = torch.stack(all_xyz)
    all_one_hot = torch.stack(all_one_hot) * norm_factor
    all_distances = torch.stack(all_distances)

    main_feat_len = num_vertices * block_size
    aux_feat_len = num_vertices
    
    aux_vec = x[main_feat_len : main_feat_len + aux_feat_len]
    remain_vec = x[main_feat_len + aux_feat_len:]

    return all_xyz, all_one_hot, all_distances, aux_vec, remain_vec

def _permute_v2_vector(x, num_vertices, args, shift):
    """Applies a circular permutation to the active atom blocks of a v2 feature vector."""
    if num_vertices <= 1 or shift == 0:
        return x

    block_size = args.max_atoms + 6
    
    # Extract the main feature part corresponding to active atoms
    active_main_features_len = num_vertices * block_size
    active_main_features = x[:active_main_features_len]
    
    # Reshape into (num_vertices, block_size)
    reshaped_active_main = active_main_features.view(num_vertices, block_size)
    
    # Roll only the active atom blocks
    permuted_reshaped_main = torch.roll(reshaped_active_main, shifts=shift, dims=0)
    
    # Flatten back
    permuted_main_flat = permuted_reshaped_main.flatten()

    # Reassemble: Take the newly permuted active block and append the rest of the original vector
    # The "rest" includes the inactive atom blocks and the aux/remain vectors.
    original_rest = x[active_main_features_len:]
    permuted_x = torch.cat([permuted_main_flat, original_rest])
    
    return permuted_x

def loss_func_qm9_qdm_v2(pred_x0, true_x0, num_vertices, args, focal_loss):
    """
    Calculates the reconstruction, focal, and prior losses.
    Permutation loss is handled separately in the training loop.
    """
    parsed_pred = _parse_v2_output(pred_x0, num_vertices, args)
    parsed_true = _parse_v2_output(true_x0, num_vertices, args)

    if parsed_pred is None or parsed_true is None:
        zero = torch.tensor(0.0, device=pred_x0.device)
        return zero, zero, zero, torch.tensor(0.0) # recon, focal, prior, acc

    pred_xyz, pred_one_hot, pred_distances, pred_aux, pred_remain = parsed_pred
    true_xyz, true_one_hot, _, true_aux, true_remain = parsed_true

    # 1. Reconstruction Loss (xyz, aux, remain)
    xyz_loss = 1000 * F.mse_loss(pred_xyz, true_xyz)
    aux_loss = F.mse_loss(pred_aux, true_aux)
    remain_loss = F.l1_loss(pred_remain, true_remain)
    recon_loss = 10 * (xyz_loss + aux_loss + remain_loss)

    # 2. Focal Loss for Classification
    true_atom_types = torch.argmax(true_one_hot, dim=-1)
    focal_loss_val = 100 * focal_loss(pred_one_hot, true_atom_types)
    accuracy = (torch.argmax(pred_one_hot, dim=-1) == true_atom_types).float().mean()

    # 3. Prior Losses
    # One-hot prior
    one_hot_norm = torch.mean(pred_one_hot, dim=-1)
    one_hot_prior_loss = 10 * F.mse_loss(one_hot_norm, torch.full_like(one_hot_norm, 0.25))

    # Geometry consistency prior
    recalculated_dist_matrix = torch.cdist(pred_xyz.unsqueeze(0), pred_xyz.unsqueeze(0)).squeeze(0)
    true_recalculated_dists_parts = []
    for i in range(num_vertices):
        dist_row = recalculated_dist_matrix[i]
        true_recalculated_dists_parts.append(torch.cat([dist_row[:i], dist_row[i+1:]]))
    true_recalculated_dists = torch.stack(true_recalculated_dists_parts)
    
    # We compare against the predicted distances for the actual atoms
    pred_distances_for_active_atoms = pred_distances[:num_vertices, :(num_vertices-1)]
    geom_consistency_loss = 1000 * F.mse_loss(pred_distances_for_active_atoms, true_recalculated_dists)
    
    prior_loss = one_hot_prior_loss + geom_consistency_loss
    
    return recon_loss, focal_loss_val, prior_loss, accuracy

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'total parameters: {total_params:,}')
    return total_params

def main():
    setup_seed(0)

    parser = config_parser()
    args = parser.parse_args()
    if args.dataset == 'qm9':
        args.atom_weight = [0.3, 0.8, 0.5, 1.2]

    save_folder = f'./save/{args.dataset}_results/{args.model_type}_qubits{args.n_qbits}_blocks{args.n_blocks}_kappa{args.kappa}_lr{args.lr}_useMLP_{args.use_MLP}_threshold{args.threshold}_maxAtoms{args.max_atoms}/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
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
        model = DrugQAE(args, n_qbits=args.n_qbits, n_blocks=args.n_blocks, bottleneck_qbits=args.bottleneck_qbits).to(torch_device)
        count_parameters(model)
    elif args.model_type == 'DrugQDM':
        model = DrugQDM(args, n_qbits=args.n_qbits, n_blocks=args.n_blocks).to(torch_device)
        count_parameters(model)
    elif args.model_type == 'DrugQDM_v2':
        model = DrugQDM_v2(args, n_qbits=args.n_qbits, n_blocks=args.n_blocks).to(torch_device)
        count_parameters(model)

    print(f'n_qbits: {args.n_qbits}, n_blocks: {args.n_blocks}')
    model.to(torch_device)
    focal_loss = MultiClassFocalLossWithAlpha(args)

    losses = []
    epochs = args.num_epochs
    if args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, 
                                   momentum=0.9)

    for epoch in range(epochs):
        torch.save(model.state_dict(), save_folder + f"model_{epoch}.pth")
        f = open(save_folder + f"res.txt", 'a')
        running_loss = 0.0
        batches = 0
        
        model.train()

        if args.dataset == 'qm9':
            if args.model_type == 'DrugQAE':
                loss_func = loss_func_qm9
            elif args.model_type == 'DrugQDM':
                loss_func = loss_func_qm9_qdm
            elif args.model_type == 'DrugQDM_v2':
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
                print(x_noisy.shape)
                print(t_indices.shape)
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
            
            elif args.model_type == 'DrugQDM_v2':
                x_whole, smi = batch_dict['x'], batch_dict['smi']
                x = x_whole[:, :-1].float().to(torch_device)
                num_vertices = x_whole[:, -1].to(torch_device).int()
                batch_size = x.shape[0]
                
                t_indices = torch.randint(0, args.timesteps, (batch_size,)).to(torch_device)
                x_noisy, _ = model.q_sample(x, t_indices)
                predicted_x0 = model(x_noisy, t_indices)
                
                # --- Loss Calculation ---
                total_loss_batch = torch.tensor(0.0, device=x.device)
                recon_loss_acc, focal_loss_acc, prior_loss_acc, perm_loss_acc, acc_acc = 0.0, 0.0, 0.0, 0.0, 0.0

                for i in range(batch_size):
                    # 1, 3, 4: Calculate Recon, Focal, and Prior losses
                    recon_loss, focal_loss_val, prior_loss, acc_ = loss_func_qm9_qdm_v2(
                        predicted_x0[i], x[i], num_vertices[i], args, focal_loss
                    )
                    
                    # 2: Calculate Permutation Loss
                    perm_loss = torch.tensor(0.0, device=x.device)
                    current_num_atoms = num_vertices[i].item()
                    if current_num_atoms > 10:
                        # Perform n-1 permutations for n atoms
                        for shift in range(1, current_num_atoms):
                            # Permute the NOISY vector directly. No gradients needed for this op.
                            permuted_noisy_x0 = _permute_v2_vector(x_noisy[i], current_num_atoms, args, shift)
                            
                            # Forward pass with the permuted noisy vector
                            perm_pred_x0 = model(permuted_noisy_x0.unsqueeze(0), t_indices[i].unsqueeze(0)).squeeze(0)
                            
                            # The prediction should match the ORIGINAL, UNPERMUTED ground truth
                            perm_loss += F.mse_loss(perm_pred_x0, x[i])
                        
                        avg_perm_loss = perm_loss / (current_num_atoms - 1)
                    else:
                        avg_perm_loss = torch.tensor(0.0, device=x.device)

                    # --- Combine all losses for this sample ---
                    sample_loss = (recon_loss +
                                   focal_loss_val +
                                   prior_loss +
                                   avg_perm_loss)
                    
                    total_loss_batch += sample_loss
                    
                    # Accumulate for logging
                    recon_loss_acc += recon_loss.item()
                    focal_loss_acc += focal_loss_val.item()
                    prior_loss_acc += prior_loss.item()
                    perm_loss_acc += avg_perm_loss.item()
                    acc_acc += acc_.item()

                # --- Averaging and Backpropagation ---
                loss = total_loss_batch / batch_size
                
                sys.stdout.write(
                    f"\r{batches + 1} / {len(train_dl)}, "
                    f"loss:{loss.item():.4f} | "
                    f"recon:{(recon_loss_acc / batch_size):.4f}, "
                    f"focal:{(focal_loss_acc / batch_size):.4f}, "
                    f"prior:{(prior_loss_acc / batch_size):.4f}, "
                    f"perm:{(perm_loss_acc / batch_size):.4f}, "
                    f"Acc:{(acc_acc / batch_size):.4f}"
                )
            
            sys.stdout.flush()

            if args.model_type == 'DrugQAE':
                loss /= x.shape[0]
            elif args.model_type == 'DrugQDM':
                loss /= batch_size
            elif args.model_type == 'DrugQDM_v2':
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
                elif args.model_type == 'DrugQDM_v2':
                    diffusion_generation(model, args, generate_num=5, f=f, debug=True)
                    
                model.train()

        losses.append(running_loss / batches)
        print(f"\nEpoch {epoch} loss: {losses[-1]}")
        f.write(f"Epoch {epoch} loss: {losses[-1]}\n")
        model.eval()
        print("generate complete dataset...")
        if args.model_type == 'DrugQAE':
            setup_seed(0)
            random_generation(model, args, f=f, debug=True)
        elif args.model_type == 'DrugQDM':
            setup_seed(0)
            diffusion_generation(model, args, f=f, debug=True)
        elif args.model_type == 'DrugQDM_v2':
            setup_seed(0)
            diffusion_generation(model, args, f=f, debug=True)

if __name__ == '__main__':
    main()