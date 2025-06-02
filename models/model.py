import torchquantum as tq
import torchquantum.functional as tqf
import torch
import torch.nn as nn
import math
import numpy as np
import torch.distributions as dist

class DrugQAE(tq.QuantumModule):
    def __init__(self, args, n_qbits=10, n_blocks=1, bottleneck_qbits=2, use_sigmoid=False):
        super().__init__()
        self.args = args
        self.MLP_in = torch.nn.Linear(args.input_dim, 2 ** args.n_qbits)
        self.MLP_out = torch.nn.Linear(2 ** args.n_qbits, args.input_dim)
        self.encoder = tq.AmplitudeEncoder() # TODO: can change to multiphaseencoder
        self.use_sigmoid = use_sigmoid
        self.n_blocks = n_blocks
        self.n_qbits = n_qbits
        self.bottleneck_qbits = bottleneck_qbits
        self.kappa = args.kappa

        self.ry_layer, self.rz1_layer, self.rz2_layer, self.crx_layer = \
            tq.QuantumModuleList(),tq.QuantumModuleList(),tq.QuantumModuleList(),tq.QuantumModuleList()
        
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

    def digits_to_binary_approx(self, x):
        return 1 / (1 + torch.exp(-50*(x-0.5)))
    
    def sampling_vmf(self, mu, kappa):

        dims = mu.shape[-1]

        epsilon = 1e-7
        x = np.arange(-1 + epsilon, 1, epsilon)
        # print(x)
        y = kappa * x + np.log(1 - x**2) * (dims - 3) / 2
        y = np.cumsum(np.exp(y - y.max()))
        y = y / y[-1]
        rand_nums = np.interp(torch.rand(10**6), y, x)

        W = torch.Tensor(rand_nums).to(mu.device)

        idx = torch.randint(0, 10**6, (mu.size(0), 1)).to(torch.int64).to(mu.device)

        w = torch.gather(W, 0, idx.squeeze()).unsqueeze(1)

        eps = torch.randn_like(mu)
        nu = eps - torch.sum(eps * mu, dim=1, keepdim=True) * mu
        nu = torch.nn.functional.normalize(nu, p=2, dim=-1)

        return w * mu + (1 - w**2)**0.5 * nu
    
    def forward(self, x, measure=False):
        bsz = x.shape[0]
        self.qdev.reset_states(bsz)

        if self.args.use_MLP:
            x = self.MLP_in(x)

        if self.encoder:
            self.encoder(self.qdev, x)

        # encode phase
        for k in range(self.n_blocks+1):
            for i in range(self.n_qbits):
                self.rz1_layer[k*self.n_qbits+i](self.qdev, wires=i)
                self.ry_layer[k*self.n_qbits+i](self.qdev, wires=i)
                self.rz2_layer[k*self.n_qbits+i](self.qdev, wires=i)
            if k != self.n_blocks:
                for i in range(self.n_qbits):
                    self.crx_layer[k*self.n_qbits+i](self.qdev, wires=[i, (i+1)%self.n_qbits])
            
        # measure phase
        if measure:
            meas = self.measure(self.qdev)
            # decode phase
            if self.use_sigmoid:
                meas = self.digits_to_binary_approx(meas)
                self.qdev.reset_states(bsz)
                for i in range(self.n_qbits):
                    tqf.rx(self.qdev, wires=i, params=math.pi*meas[:, i])
            else:
                self.qdev.reset_states(bsz)
                for i in range(self.n_qbits):
                    tqf.ry(self.qdev, wires=i, params=meas[:, i])
        else:
            output1 = self.qdev.get_states_1d().abs()
            # 2 ^ 7 -> 2^ 5 # bottleneck操作 ae中显式压缩信息
            output1 = output1**2
            output1 = output1.reshape(-1, pow(2, self.n_qbits - self.bottleneck_qbits), pow(2, self.bottleneck_qbits)).sum(axis=-1).sqrt() # 显式meature, output1作为vMF分布的mu
            decoder_in = self.sampling_vmf(output1, self.kappa)
            self.qdev.reset_states(bsz)
            self.encoder(self.qdev, decoder_in)
        
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

        reconstruct_vector = self.qdev.get_states_1d().abs()
        if self.args.use_MLP:
            reconstruct_vector = self.MLP_out(reconstruct_vector)
                
        return output1, reconstruct_vector
    

    def generate(self, x):
        bsz = x.shape[0]
        self.qdev.reset_states(bsz)

        self.encoder(self.qdev, x)

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

        generate_vector = self.qdev.get_states_1d().abs()
                
        return generate_vector

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

    # def Unif2Gauss(self, x):
    #     """
    #     Input: x with shape (batch_size, dim), where each row is on S^{dim-1} ∩ R^dim_{>0}(unit norm, all positive)
    #     Output: standard Gaussian distribution with same shape
    #     """
    #     batch_size, dim = x.shape
    #     device = x.device
        
    #     # Method: Use the fact that if Z ~ N(0,I), then |Z|/||Z|| is uniform on positive unit sphere
    #     # So we need to: 1) sample radius from chi distribution, 2) assign random signs
        
    #     generator = torch.Generator(device=device)
    #     generator.manual_seed(0)
        
    #     # Step 1: Sample radius-squared from chi-squared distribution with 'dim' degrees of freedom
    #     # This gives us the correct radial distribution for Gaussian vectors
    #     chi2_dist = dist.Chi2(dim)
    #     radius_squared = chi2_dist.sample((batch_size,)).to(device)
    #     radius = torch.sqrt(radius_squared)
        
    #     # Step 2: Scale the unit vectors by the sampled radius
    #     scaled_vectors = x * radius.unsqueeze(-1)
        
    #     # Step 3: Randomly assign signs to each component
    #     # Each component should be positive or negative with equal probability
    #     signs = torch.randint(0, 2, (batch_size, dim), device=device, generator=generator) * 2 - 1  # -1 or +1
    #     gaussian_vectors = scaled_vectors * signs.float()
        
    #     return gaussian_vectors
 
    def forward(self, x, t_indices):
        bsz = x.shape[0]
        self.qdev.reset_states(bsz)
        x_with_t = self.add_condition_embedding(x, t_indices)
        
        main_feat_norm = x_with_t[:, :2**self.args.main_qbits] / torch.norm(x_with_t[:, :2**self.args.main_qbits], dim=1, keepdim=True)
        time_feat = x_with_t[:, 2**self.args.main_qbits:]
        x_with_t = torch.cat([main_feat_norm, time_feat], dim=1) / math.sqrt(2.0)

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
        
        # pred_noise = self.measure(self.qdev)
        pred_noise = self.qdev.get_states_1d().abs()
        pred_noise = pred_noise**2
        pred_noise = pred_noise.reshape(-1, pow(2, self.args.main_qbits), pow(2, self.n_qbits - self.args.main_qbits)).sum(axis=-1).sqrt()
        pred_noise = self.post_mlp(pred_noise)
        # pred_noise = self.Unif2Gauss(pred_noise)
        # if torch.rand(1) < 0.1:
        #     print(f'pred_noise: {pred_noise[0][0:10]}')
        return pred_noise
    
    def sample_ddim(self, batch_size, ddim_steps=None, ddim_eta=None):
        if ddim_steps is None:
            ddim_steps = self.args.ddim_steps
        if ddim_eta is None:
            ddim_eta = self.args.ddim_eta
        print(ddim_steps, ddim_eta)
        device = next(self.parameters()).device
        
        generator = torch.Generator(device=device)
        generator.manual_seed(0)
        
        x = torch.randn(batch_size, 2**self.args.main_qbits, device=device, generator=generator)
        
        time_indices = torch.linspace(self.args.timesteps - 1, 0, ddim_steps + 1, dtype=torch.long).to(device)
        
        for i in range(ddim_steps):
            t_curr = time_indices[i].repeat(batch_size)
            t_next = time_indices[i+1].repeat(batch_size) if i < ddim_steps-1 else torch.zeros_like(t_curr)
            
            predicted_noise = self(x, t_curr)
            
            _, alpha_bar_curr = self.compute_alpha(t_curr)
            _, alpha_bar_next = self.compute_alpha(t_next) if i < ddim_steps-1 else (torch.ones_like(t_curr), torch.ones_like(t_curr))
            
            eps = 1e-4
            alpha_bar_curr = torch.clamp(alpha_bar_curr, min=eps, max=1.0-eps)
            sigma_square_term = torch.clamp(
                (1.0 - alpha_bar_next) / (1.0 - alpha_bar_curr) * (1.0 - alpha_bar_curr / alpha_bar_next),
                min=0.0
            )
            sigma = ddim_eta * torch.sqrt(sigma_square_term)
            sqrt_term = torch.clamp(1 - alpha_bar_next - sigma**2, min=0.0)

            c1 = torch.sqrt(alpha_bar_next / alpha_bar_curr)
            c2 = torch.sqrt(sqrt_term)
            c3 = torch.sqrt(1 - alpha_bar_curr)
            c2 = c2 - c3 * c1
            c1 = torch.clamp(c1, min=1.0, max=2.0)
            c2 = torch.clamp(c2, min=-1.0, max=1.0)
            c3 = torch.clamp(c3, min=0.0, max=1.0)
            print(f'c1: {c1},\n c2: {c2},\n c3: {c3}')
            
            noise = torch.randn_like(x) if ddim_eta > 0 else 0
            
            x_next = c1.unsqueeze(-1) * x + c2.unsqueeze(-1) * predicted_noise
            if ddim_eta > 0:
                x_next = x_next + sigma.unsqueeze(-1) * noise
            
            x = x_next
        # x = (x_next - x_next.min(dim=1, keepdim=True).values) / (x_next.max(dim=1, keepdim=True).values - x_next.min(dim=1, keepdim=True).values)
        # x = x_next.abs() / torch.norm(x_next, dim=1, keepdim=True)

        assert not torch.isnan(x).any()
        return x