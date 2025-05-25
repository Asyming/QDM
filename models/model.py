import torchquantum as tq
import torchquantum.functional as tqf
import torch
import torch.nn as nn
import math
import numpy as np


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
        print(torch.sum(x[0][0:72]**2))

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
        self.time_embed = nn.Sequential(
            nn.Linear(1, args.time_emb_dim),
            nn.SiLU(),
            nn.Linear(args.time_emb_dim, args.time_emb_dim)
        )
        self.encoder = tq.AmplitudeEncoder()
        self.use_sigmoid = use_sigmoid
        self.n_blocks = n_blocks
        self.n_qbits = n_qbits
        self.beta = torch.linspace(args.beta_start, args.beta_end, args.timesteps).to(self.device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

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
        
    def add_time_embedding(self, x, t): # x: x_noisy, t: t
        bsz = x.shape[0]
        t_embed = self.time_embed(t.reshape(-1, 1)) # TODO: 改成用sin和cos, 同时归一化
        t_embed = t_embed / torch.norm(t_embed, dim=1, keepdim=True)
        results = []
        
        start_pos = 2**self.args.main_qbits
        embed_len = t_embed.shape[1]
        
        assert start_pos + embed_len <= 2**self.n_qbits, "Time embedding exceeds vector dimension"
        
        for i in range(bsz):
            current_x = x[i:i+1].clone()
            current_x= torch.cat([current_x, t_embed[i:i+1], torch.zeros(1, 2**self.n_qbits - start_pos - embed_len).to(self.device)], dim=1)
            results.append(current_x)
        results = torch.cat(results, dim=0)
        return results

    def compute_alpha(self, t):
        t_int = (t * (self.args.timesteps - 1)).long()
        t_int = torch.clamp(t_int, 0, self.args.timesteps - 1)
        
        alpha_t = self.alpha[t_int]
        alpha_bar_t = self.alpha_bar[t_int]
        return alpha_t, alpha_bar_t

    def q_sample(self, x, t, noise=None):
        x_main = x[:, :-1]
        if noise is None:
            noise = torch.randn_like(x_main)
        _, alpha_bar_t = self.compute_alpha(t)
        alpha_bar_t = alpha_bar_t.unsqueeze(-1)
        x_noisy = torch.sqrt(alpha_bar_t) * x_main + torch.sqrt(1 - alpha_bar_t) * noise
        noise = noise.abs() / torch.norm(noise, dim=1, keepdim=True) # TODO: 这样是否会造成异号的噪声认为是同一个?
        x_noisy = torch.cat([x_noisy, x[:, -1].unsqueeze(-1)], dim=1)
        noise = torch.cat([noise, x[:, -1].unsqueeze(-1)], dim=1)
        return x_noisy, noise
 
    def forward(self, x, t): # x: x_noisy, t: t
        bsz = x.shape[0]
        self.qdev.reset_states(bsz)
        x_with_t = self.add_time_embedding(x, t)
        # print(x_with_t[0,56:64])
        main_norm = x_with_t[:, 0:7*9] / torch.norm(x_with_t[:, 0:7*9], dim=1, keepdim=True)
        x_with_t = torch.cat([main_norm, x_with_t[:, 7*9:]], dim=1)
        x_with_t = x_with_t / torch.sqrt((2.0 + x_with_t[:, 2**self.args.main_qbits-1]**2).unsqueeze(-1))
        # norm = torch.norm(x_with_t, dim=1, keepdim=True)
        # # print(f't, norm: {t[0], norm[0]}')
        # x_with_t_norm = x_with_t / norm
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

        pred_noise = self.qdev.get_states_1d().abs()
        pred_noise = pred_noise**2
        pred_noise = pred_noise.reshape(-1, pow(2, self.args.main_qbits), pow(2, self.n_qbits - self.args.main_qbits)).sum(axis=-1).sqrt()
        # weights = pred_noise ** 2
        # samples = torch.randn_like(pred_noise)
        # gaussian_noise = (weights * samples).sum(dim=-1, keepdim=True)
        # pred_noise = (gaussian_noise - gaussian_noise.mean(dim=-1, keepdim=True)) / gaussian_noise.std(dim=-1, keepdim=True)
        # pred_noise = torch.sqrt(torch.tensor(2.0)) * torch.erfinv(2 * pred_noise - 1) # Uniform to Gaussian
        # pred_noise = pred_noise # * norm
        # print(pred_noise[0][0])
        return pred_noise
    
    def sample_ddim(self, batch_size, ddim_steps=None, ddim_eta=None):
        if ddim_steps is None:
            ddim_steps = self.args.ddim_steps
        if ddim_eta is None:
            ddim_eta = self.args.ddim_eta
        print(ddim_steps, ddim_eta)
        device = next(self.parameters()).device
        x_main = torch.randn(batch_size, 2**self.args.main_qbits-1).to(device)
        x = torch.cat([x_main, torch.randint(9, 10, (batch_size, 1)).to(device)], dim=1)# TODO: 改成按数据集的比例生成
        time_steps = torch.linspace(1, 0, ddim_steps + 1).to(device)
        # self.qdev.reset_states(batch_size)
        for i in range(ddim_steps):
            t_curr = time_steps[i] * torch.ones(batch_size).to(device)
            t_next = time_steps[i+1] * torch.ones(batch_size).to(device)
            # Predict noise
            predicted_noise = self(x, t_curr) if i == 0 else self(x_next, t_curr)
            _, alpha_bar_curr = self.compute_alpha(t_curr)
            _, alpha_bar_next = self.compute_alpha(t_next) if i < ddim_steps-1 else (torch.ones_like(t_curr), torch.ones_like(t_curr))
            
            eps = 1e-8
            alpha_bar_curr = torch.clamp(alpha_bar_curr, min=eps, max=1.0-eps)
            sigma_square_term = torch.clamp(
                (1.0 - alpha_bar_next) / (1.0 - alpha_bar_curr) * (1.0 - alpha_bar_curr / alpha_bar_next),
                min=0.0
            )
            sigma = ddim_eta * torch.sqrt(sigma_square_term)
            print(f'sigma: {sigma[0]}')
            sqrt_term = torch.clamp(1 - alpha_bar_next - sigma**2, min=0.0)
            # print(f'sigma: {sigma[0]}')

            c1 = torch.sqrt(alpha_bar_next / alpha_bar_curr)
            c2 = torch.sqrt(sqrt_term)
            c3 = torch.sqrt(1 - alpha_bar_curr)
            c2 = c2 - c3 * c1
            
            noise = torch.randn_like(x) if ddim_eta > 0 else 0
            noise = noise.abs() / torch.norm(noise, dim=1, keepdim=True) ## align with predicted_noise and true_noise
            x_next = c1.unsqueeze(-1) * x_next + c2.unsqueeze(-1) * predicted_noise if i > 0 else c1.unsqueeze(-1) * x + c2.unsqueeze(-1) * predicted_noise
            if ddim_eta > 0:
                x_next = x_next + sigma.unsqueeze(-1) * noise
        x = (x_next - x_next.min(dim=1, keepdim=True).values) / (x_next.max(dim=1, keepdim=True).values - x_next.min(dim=1, keepdim=True).values)
        # x = x_next.abs() / torch.norm(x_next, dim=1, keepdim=True)

        assert not torch.isnan(x).any()
        return x