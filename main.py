import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import QDrugDataset
from models.model import DrugQAE, DrugQDM
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

def loss_func_qm9_qdm(x, predicted_noise, true_noise, num_vertices, focal_loss):
    #data_vertices = x[:7*num_vertices].reshape(num_vertices, 7)
    pred_noise_vertices = predicted_noise[:7*num_vertices].reshape(num_vertices, 7)
    true_noise_vertices = true_noise[:7*num_vertices].reshape(num_vertices, 7)
    
    pred_noise_xyz = pred_noise_vertices[:, :3]
    true_noise_xyz = true_noise_vertices[:, :3]
    
    pred_noise_atom = pred_noise_vertices[:, 3:7]
    true_noise_atom = true_noise_vertices[:, 3:7]
    
    atom_num_pred = torch.sqrt(2.0 * predicted_noise[-1]**2 / (1.0 - predicted_noise[-1]**2))
    atom_num_true = true_noise[-1] #num_vertices.float() / torch.sqrt(2.0 + num_vertices.float()**2)

    xyz_loss = F.mse_loss(pred_noise_xyz, true_noise_xyz)
    atom_loss = F.mse_loss(pred_noise_atom, true_noise_atom)
    atom_num_loss = F.l1_loss(atom_num_pred, atom_num_true)

    # aux_noise_pred = predicted_noise[7*num_vertices:8*num_vertices]
    # aux_noise_true = true_noise[7*num_vertices:8*num_vertices]
    # aux_loss = F.l1_loss(aux_noise_pred, aux_noise_true) # 107, 109, 114
    # # aux_loss = F.mse_loss(aux_noise_pred, aux_noise_true) # 106, 108, 111, 112, 113
    
    # Loss for remaining noise
    # remain_noise_pred = predicted_noise[7*num_vertices:-1]
    # remain_noise_true = true_noise[7*num_vertices:-1]
    # remain_loss = F.l1_loss(remain_noise_pred, remain_noise_true) # 107, 109, 114
    # remain_loss = F.mse_loss(remain_noise_pred, remain_noise_true) # 106, 108, 111, 112, 113

    # Calculate noise direction accuracy (using cosine similarity)
    cos_sim = F.cosine_similarity(
        predicted_noise.unsqueeze(0), 
        true_noise.unsqueeze(0), 
        dim=1
    )
    noise_direction_acc = ((cos_sim + 1) / 2).item()  # Convert to 0-1 range scalar

    # Total loss
    total_loss = xyz_loss + atom_loss + atom_num_loss * 0.001

    return total_loss, xyz_loss, atom_loss, atom_num_loss * 0.001, noise_direction_acc

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

    train_dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=torch.utils.data.RandomSampler(dataset),
    )

    # Select model based on model_type
    if args.model_type == 'DrugQAE':
        model = DrugQAE(args, n_qbits=args.n_qbits, n_blocks=args.n_blocks, 
                        bottleneck_qbits=args.bottleneck_qbits).to(torch_device)
    elif args.model_type == 'DrugQDM':
        model = DrugQDM(args, n_qbits=args.n_qbits, n_blocks=args.n_blocks).to(torch_device)

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
                x, smi = batch_dict['x'], batch_dict['smi']
                x = x.float().to(torch_device)
                num_vertices = x[:, -1].clone().int()
                assert torch.all((num_vertices >= 1) & (num_vertices <= args.max_atoms))
                batch_size = x.shape[0]
                t = torch.rand(batch_size).to(torch_device)
                x_noisy, true_noise = model.q_sample(x, t)
                predicted_noise = model(x_noisy, t)

                loss, xyz_loss, atom_loss, atom_num_loss, noise_acc = 0.0, 0.0, 0.0, 0.0, 0.0
                
                for i in range(batch_size):
                    loss_, xyz_loss_, atom_loss_, atom_num_loss_, noise_acc_ = loss_func(
                        x[i], predicted_noise[i], true_noise[i], num_vertices[i], focal_loss)
                    loss += loss_
                    xyz_loss += xyz_loss_
                    atom_loss += atom_loss_
                    atom_num_loss += atom_num_loss_
                    noise_acc += noise_acc_
                
                sys.stdout.write(
                    f"\r{batches + 1} / {len(train_dl)}, "
                    f"loss:{loss / batch_size:.6f}, "
                    f"xyz:{xyz_loss / batch_size:.6f}, "
                    f"atom:{atom_loss / batch_size:.6f}, "
                    f"atom_num:{atom_num_loss / batch_size:.6f}, "
                    f"NoiseAcc:{noise_acc / batch_size:.6f}"
                )
            
            sys.stdout.flush()

            if args.model_type == 'DrugQAE':
                loss /= x.shape[0]
            elif args.model_type == 'DrugQDM':
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

if __name__ == '__main__':
    main()