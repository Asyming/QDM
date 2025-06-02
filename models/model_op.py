import torchquantum as tq
import torchquantum.functional as tqf
import torch
import torch.nn as nn
import math
import numpy as np
import torch.distributions as dist

class DrugQDM(tq.QuantumModule):
    def __init__(self, args, n_qbits=10, n_blocks=1, use_sigmoid=False):
        super().__init__()
        self.args = args
        self.device = args.device
        self.encoder = tq.AmplitudeEncoder()
        self.use_sigmoid = use_sigmoid
        self.n_blocks = n_blocks
        self.n_qbits = args.n_qbits
        self.main_qbits = args.main_qbits
        self.post_mlp = nn.Sequential(
            nn.Linear(2**args.main_qbits, 2**args.main_qbits)
        )
        
        steps = torch.arange(args.timesteps + 1, dtype=torch.float32)
        alphas_cumprod = torch.cos(((steps / args.timesteps) + args.cosine_s) / (1 + args.cosine_s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        self.alpha_bar = alphas_cumprod[1:].to(self.device)
        betas = 1 - (self.alpha_bar[1:] / self.alpha_bar[:-1])
        beta_0 = 1 - self.alpha_bar[0]
        self.beta = torch.cat([beta_0.unsqueeze(0), betas])
        self.beta = torch.clamp(self.beta, 0.0001, 0.9999)
        self.alpha = 1 - self.beta

        self.ry_layer, self.rz1_layer, self.rz2_layer, self.crx_layer = \
            tq.QuantumModuleList(), tq.QuantumModuleList(), tq.QuantumModuleList(), tq.QuantumModuleList()
        self.ry_layer_d, self.rz1_layer_d, self.rz2_layer_d, self.crx_layer_d = \
            tq.QuantumModuleList(),tq.QuantumModuleList(),tq.QuantumModuleList(),tq.QuantumModuleList()
        
        for k in range(n_blocks+1):
            for i in range(n_qbits):
                if k == n_blocks:
                    self.rz1_layer.append(tq.RZ(trainable=True, has_params=True))
                    self.ry_layer.append(tq.RY(trainable=True, has_params=True))
                    self.rz2_layer.append(tq.RZ(trainable=True, has_params=True))
                else:
                    self.rz1_layer.append(tq.RZ(trainable=True, has_params=True))
                    self.ry_layer.append(tq.RY(trainable=True, has_params=True))
                    self.rz2_layer.append(tq.RZ(trainable=True, has_params=True))
                    self.crx_layer.append(tq.CRX(trainable=True, has_params=True))

        for k in range(n_blocks+1):
            for i in range(n_qbits):
                if k == n_blocks:
                    self.rz1_layer_d.append(tq.RZ(trainable=True, has_params=True))
                    self.ry_layer_d.append(tq.RY(trainable=True, has_params=True))
                    self.rz2_layer_d.append(tq.RZ(trainable=True, has_params=True))
                else:
                    self.rz1_layer_d.append(tq.RZ(trainable=True, has_params=True))
                    self.ry_layer_d.append(tq.RY(trainable=True, has_params=True))
                    self.rz2_layer_d.append(tq.RZ(trainable=True, has_params=True))
                    self.crx_layer_d.append(tq.CRX(trainable=True, has_params=True))
                    
        self.service = "TorchQuantum"
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.qdev = tq.QuantumDevice(n_qbits)
        
    def compute_alpha(self, t_indices):
        t_indices = torch.clamp(t_indices, 0, self.args.timesteps - 1)
        
        alpha_t = self.alpha[t_indices]
        alpha_bar_t = self.alpha_bar[t_indices]
        return alpha_t, alpha_bar_t

    def sinusoidal_time_embedding(self, timesteps, embedding_dim):
        half_dim = embedding_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half_dim, dtype=torch.float32) / half_dim)
        freqs = freqs.to(timesteps.device)
        args = timesteps[:, None].float() * freqs[None, :]
        embeddings = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if embedding_dim % 2 == 1:
            embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)
        return embeddings

    def add_condition_embedding(self, x, t_indices):
        bsz = x.shape[0]
        available_dim = 2**self.n_qbits - 2**self.args.main_qbits
        time_emb_dim = min(available_dim, 2**self.args.time_emb_dim)
        
        t_embed = self.sinusoidal_time_embedding(t_indices, time_emb_dim)
        t_embed = t_embed / torch.norm(t_embed, dim=1, keepdim=True)
        
        results = []
        
        for i in range(bsz):
            current_x = x[i:i+1].clone()
            current_x = torch.cat([
                current_x,
                t_embed[i:i+1], 
                torch.zeros(1, 2**self.n_qbits - 2**self.args.main_qbits - time_emb_dim).to(self.device)
            ], dim=1)
            results.append(current_x)
        
        results = torch.cat(results, dim=0)
        return results

    def q_sample(self, x, t_indices, noise=None):
        if noise is None:
            noise = torch.randn_like(x)
        _, alpha_bar_t = self.compute_alpha(t_indices)
        alpha_bar_t = alpha_bar_t.unsqueeze(-1)
        x_noisy = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * noise
        return x_noisy, noise
 
    def forward(self, x, t_indices):
        bsz = x.shape[0]
        self.qdev.reset_states(bsz)
        x_with_t = self.add_condition_embedding(x, t_indices)
        
        main_feat_norm = x_with_t[:, :2**self.args.main_qbits] / torch.norm(x_with_t[:, :2**self.args.main_qbits], dim=1, keepdim=True)
        time_feat = x_with_t[:, 2**self.args.main_qbits:]
        x_with_t = torch.cat([main_feat_norm, time_feat], dim=1) / math.sqrt(2.0)
        # x_with_t = x_with_t / torch.norm(x_with_t, dim=1, keepdim=True)

        self.encoder(self.qdev, x_with_t)
        for k in range(self.n_blocks+1):
            for i in range(self.n_qbits):
                self.rz1_layer[k*self.n_qbits+i](self.qdev, wires=i)
                self.ry_layer[k*self.n_qbits+i](self.qdev, wires=i)
                self.rz2_layer[k*self.n_qbits+i](self.qdev, wires=i)
            if k != self.n_blocks:
                for i in range(self.n_qbits):
                    self.crx_layer[k*self.n_qbits+i](self.qdev, wires=[i, (i+1)%self.n_qbits])

        for k in range(self.n_blocks, -1, -1):
            if k == self.n_blocks:
                for i in range(self.n_qbits):
                    self.rz2_layer_d[k*self.n_qbits+i](self.qdev, wires=i)
                    self.ry_layer_d[k*self.n_qbits+i](self.qdev, wires=i)
                    self.rz1_layer_d[k*self.n_qbits+i](self.qdev, wires=i)
            else:
                for i in range(self.n_qbits-1, -1, -1):
                    self.crx_layer_d[k*self.n_qbits+i](self.qdev, wires=[i, (i+1)%self.n_qbits])
                for i in range(self.n_qbits):
                    self.rz2_layer_d[k*self.n_qbits+i](self.qdev, wires=i)
                    self.ry_layer_d[k*self.n_qbits+i](self.qdev, wires=i)
                    self.rz1_layer_d[k*self.n_qbits+i](self.qdev, wires=i)
        
        pred_x0 = self.qdev.get_states_1d().abs()
        pred_x0 = pred_x0**2
        pred_x0 = pred_x0.reshape(-1, pow(2, self.args.main_qbits), pow(2, self.n_qbits - self.args.main_qbits)).sum(axis=-1).sqrt()
        
        return pred_x0
    
    def sample_ddim(self, batch_size, ddim_steps=None, ddim_eta=None):
        if ddim_steps is None:
            ddim_steps = self.args.ddim_steps
        if ddim_eta is None:
            ddim_eta = self.args.ddim_eta
        print(ddim_steps, ddim_eta)
        device = next(self.parameters()).device
        
        generator = torch.Generator(device=device)
        generator.manual_seed(0)
        
        x = torch.randn(batch_size, 2**self.args.main_qbits, device=device, generator=generator)/ torch.norm(torch.randn(batch_size, 2**self.args.main_qbits, device=device, generator=generator), dim=1, keepdim=True)
        
        time_indices = torch.linspace(self.args.timesteps - 1, 0, ddim_steps + 1, dtype=torch.long).to(device)
        
        for i in range(ddim_steps):
            t_curr = time_indices[i].repeat(batch_size)
            t_next = time_indices[i+1].repeat(batch_size) if i < ddim_steps-1 else torch.zeros_like(t_curr)
            
            predicted_x0 = self(x, t_curr)
            
            _, alpha_bar_curr = self.compute_alpha(t_curr)
            _, alpha_bar_next = self.compute_alpha(t_next) if i < ddim_steps-1 else (torch.ones_like(t_curr), torch.ones_like(t_curr))
            
            eps = 1e-4
            alpha_bar_curr = torch.clamp(alpha_bar_curr, min=eps, max=1.0-eps)
            alpha_bar_next = torch.clamp(alpha_bar_next, min=eps, max=1.0-eps)
            
            sigma_square_term = torch.clamp(
                (1.0 - alpha_bar_next) / (1.0 - alpha_bar_curr) * (1.0 - alpha_bar_curr / alpha_bar_next),
                min=0.0
            )
            sigma = ddim_eta * torch.sqrt(sigma_square_term)
            
            # x_{t-1} = sqrt(alpha_bar_{t-1}) * x0 + sqrt(1-alpha_bar_{t-1}-sigma^2) * predicted_noise + sigma * noise
            # predicted_noise = (x_t - sqrt(alpha_bar_t) * x0) / sqrt(1 - alpha_bar_t)
            
            sqrt_alpha_bar_curr = torch.sqrt(alpha_bar_curr).unsqueeze(-1)
            sqrt_one_minus_alpha_bar_curr = torch.sqrt(1 - alpha_bar_curr).unsqueeze(-1)
            
            predicted_noise = (x - sqrt_alpha_bar_curr * predicted_x0) / sqrt_one_minus_alpha_bar_curr
            
            sqrt_alpha_bar_next = torch.sqrt(alpha_bar_next).unsqueeze(-1)
            sqrt_one_minus_alpha_bar_next_minus_sigma2 = torch.sqrt(torch.clamp(1 - alpha_bar_next - sigma**2, min=0.0)).unsqueeze(-1)
            
            x_next = sqrt_alpha_bar_next * predicted_x0 + sqrt_one_minus_alpha_bar_next_minus_sigma2 * predicted_noise
            
            if ddim_eta > 0:
                noise = torch.randn_like(x)
                x_next = x_next + sigma.unsqueeze(-1) * noise
            
            x = x_next
            
            print(f'Step {i}: t_curr={t_curr[0].item()}, alpha_bar_curr={alpha_bar_curr[0].item():.4f}, alpha_bar_next={alpha_bar_next[0].item():.4f}')

        assert not torch.isnan(x).any()
        return x