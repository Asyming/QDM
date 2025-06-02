import argparse


def config_parser():
    parser = argparse.ArgumentParser(description='Qdrug')
    parser.add_argument("--model_type", type=str, default='DrugQAE', choices=['DrugQAE', 'DrugQDM'])
    # training options
    parser.add_argument("--device", type=str, default='cuda', help='device')
    parser.add_argument("--n_qbits", type=int, default=8, help='number of qubits in the QAE/QDM')
    parser.add_argument("--main_qbits", type=int, default=7, help='number of main qubits in the QDM') # 7 for QM9
    parser.add_argument("--num_epochs", type=int, default=1000, help='number of epochs to run')
    parser.add_argument("--lr", type=float, default=0.0005, help='initial learning rate')
    parser.add_argument("--batch_size", type=int, default=512, help='batch size')
    parser.add_argument("--generate_num", type=int, default=1000, help='generate num')
    parser.add_argument("--eval_interval", type=int, default=10, help='eval interval')
    parser.add_argument("--n_blocks", type=int, default=10, help='no of blocks in the QAE')
    parser.add_argument("--optim", type=str, default='adam', help='optimizer to use')
    parser.add_argument("--weight_decay", type=float, default=0.0001, help='weight decay')
    parser.add_argument("--focal_gamma", type=float, default=2, help='focal gamma')
    parser.add_argument("--atom_weight", type=list, default=[], help='atom weight')
    parser.add_argument("--use_MLP", action='store_true')
    parser.add_argument("--input_dim", type=int, default=73, help='input dim') # qm9 : 9 * (3+4+1), zinc : x * (3+9+1)
    parser.add_argument("--use_sigmoid", type=bool, default=False, help='use sigmoid')
    parser.add_argument("--dataset", type=str, default='qm9', help='dataset')
    parser.add_argument("--kappa", type=float, default=20, help='save dir')
    parser.add_argument("--bottleneck_qbits", type=int, default=2, help='save dir')
    parser.add_argument("--hidden_dim", type=int, default=512, help='save dir')
    parser.add_argument("--postprecess", type=int, default=0, help='save dir')
    parser.add_argument("--threshold", type=float, default=0.2, help='save dir')
    parser.add_argument("--max_atoms", type=int, default=9, help='max_atoms')
    parser.add_argument("--atom_dict", type=dict, default={0:6, 1:7, 2:8, 3:9, 4:16, 5:17, 6:36, 7:53, 8:54}, help='atom_dict')
    parser.add_argument("--num_condition", type=int, default=0)
    parser.add_argument("--property_index", type=int, default=0)

    parser.add_argument("--timesteps", type=int, default=500, help='diffusion timesteps')
    parser.add_argument("--time_emb_dim", type=int, default=32, help='time embedding dimension')
    # parser.add_argument("--beta_start", type=float, default=1e-4, help='start value for beta schedule')
    # parser.add_argument("--beta_end", type=float, default=0.02, help='end value for beta schedule')
    parser.add_argument("--cosine_s", type=float, default=0.008, help='cosine schedule offset parameter')
    
    parser.add_argument("--ddim_steps", type=int, default=30, help='Number of DDIM sampling steps')
    parser.add_argument("--ddim_eta", type=float, default=0.0, help='DDIM eta parameter (0 for deterministic sampling)')

    parser.add_argument("--exp_id", type=int, default=0)

    
    return parser