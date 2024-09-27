import math

from dac_repo.dac.nn.quantize import *
from .importance_subnet import ImportanceSubnet
from .utils import generate_mask_smooth


class VectorQuantize(nn.Module):
    """
    Implementation of VQ similar to Karpathy's repo:
    https://github.com/karpathy/deep-vector-quantization
    Additionally uses following tricks from Improved VQGAN
    (https://arxiv.org/pdf/2110.04627.pdf):
        1. Factorized codes: Perform nearest neighbor lookup in low-dimensional space
            for improved codebook usage
        2. l2-normalized codes: Converts euclidean distance to cosine similarity which
            improves training stability
    """

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int, 
                 quantizer_loss:str = "normalized_l2"):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer_loss = quantizer_loss

        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, z, loss_per_frame=False):
        """Quantized the input tensor using a fixed codebook and returns
        the corresponding codebook vectors

        Parameters
        ----------
        z : Tensor[B x D x T]

        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        Tensor[1]
            Commitment loss to train encoder to predict vectors closer to codebook
            entries
        Tensor[1]
            Codebook loss to update the codebook
        Tensor[B x T]
            Codebook indices (quantized discrete representation of input)
        Tensor[B x D x T]
            Projected latents (continuous representation of input before quantization)
        """

        # Factorized codes (ViT-VQGAN) Project input into low-dimensional space
        z_e = self.in_proj(z)  # z_e : (B x D x T)
        z_q, indices = self.decode_latents(z_e)

        reduce_dim = [1] if loss_per_frame else [1, 2]
        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean(reduce_dim)
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean(reduce_dim)

        z_q = (
            z_e + (z_q - z_e).detach()
        )  # noop in forward pass, straight-through gradient estimator in backward pass

        z_q = self.out_proj(z_q)

        return z_q, commitment_loss, codebook_loss, indices, z_e

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight  # codebook: (N x D)

        # L2 normalize encodings and codebook (ViT-VQGAN)
        if self.quantizer_loss == "normalized_l2":
            encodings = F.normalize(encodings)
            codebook = F.normalize(codebook)
        elif self.quantizer_loss == "l2":
            # assert False
            pass
        else:
            raise ValueError(f"Invalid quantizer loss: {self.quantizer_loss}")

        # Compute euclidean distance with codebook
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)
        return z_q, indices


class VBRResidualVectorQuantize(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: int = 8,
        
        quantizer_dropout: float = 0.0,  
        full_codebook_rate: float=0.0, 
        use_framewise_masking: bool=False,
        level_min: float=1.0,
        level_max: float=48.0,
        operator_mode: str = "scaling", ## in ["scaling", "exponential", "transformed scaling"] ## Paper: scaling
        
        imp2mask_alpha: float = 1.0,
        imp2mask_func: str="logcosh", ## logcosh, square, sigmoid
    ):
        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim for _ in range(n_codebooks)]

        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        self.quantizers = nn.ModuleList(
            [
                VectorQuantize(input_dim, codebook_size, codebook_dim[i])
                for i in range(n_codebooks)
            ]
        )
        
        self.imp_subnet = ImportanceSubnet(d_feat=input_dim)
        self.quantizer_dropout = quantizer_dropout
        self.full_codebook_rate = full_codebook_rate
        self.use_framewise_masking = use_framewise_masking
        self.level_min = level_min
        self.level_max = level_max
        
        self.operator_mode = operator_mode
        self.imp2mask_alpha = imp2mask_alpha
        self.imp2mask_func = imp2mask_func
        
        
    def forward(
        self,
        z: torch.Tensor,
        n_quantizers: int = None,
        feat_enc: torch.Tensor = None,
        level: float = None
    ):
        z_q = 0
        residual = z
        bs, _, frames = z.shape # (bs, 1024, frames)
        
        commitment_loss = 0
        codebook_loss = 0
        
        codebook_indices = []
        latents = []
        
        if n_quantizers is None:
            mode = "VBR"
        else:
            mode = "CBR"
        
        if mode == "VBR":
            imp_map = self.imp_subnet(feat_enc)  ## (B, 1, T) in (0, 1)
            
            if self.training:
                if self.operator_mode == "scaling":
                    random_levels = torch.rand((bs, 1, 1)) * (self.level_max - self.level_min) + self.level_min
                    random_levels = random_levels.to(z)
                    imp_map_scaled = imp_map * random_levels
                else:
                    random_levels = torch.rand((bs, 1, 1)) * (math.log(self.level_max) - math.log(self.level_min)) + math.log(self.level_min)
                    random_levels = torch.exp(random_levels).to(z)
                    ## TODO: Implement other operator modes
                    # imp_map_scaled = apply_operator(imp_map, random_levels, self.n_codebooks, self.operator_mode)
                    raise NotImplementedError("Other operator modes are not implemented yet")
            else: ## Inference
                if level is None:
                    imp_map_scaled = imp_map * self.n_codebooks
                else:
                    if self.operator_mode == "scaling":
                        imp_map_scaled = imp_map * level
                    else:
                        raise NotImplementedError("Other operator modes are not implemented yet")
                    
            mask_imp = generate_mask_smooth(
                imp_map_scaled,
                self.n_codebooks,
                alpha=self.imp2mask_alpha,
                function=self.imp2mask_func,
            )
            
        elif mode == "CBR":
            imp_map_scaled = torch.ones((bs, 1, frames)).to(z) * (n_quantizers)
            # imp_map = torch.ones((bs, 1, frames)).to(z)
            imp_map = None
            mask_imp = None
           
        else:
            raise ValueError("Invalid mode")
        
        if self.training:
            if self.use_framewise_masking:
                n_quantizers = torch.ones((bs, 1, frames)) * self.n_codebooks + 1
                dropout = torch.randint(1, self.n_codebooks + 1, (bs, 1, frames))
            else:
                n_quantizers = torch.ones((bs, 1, 1)) * self.n_codebooks + 1
                dropout = torch.randint(1, self.n_codebooks + 1, (bs, 1, 1))
            
            n_full = int(bs * self.full_codebook_rate)
            n_dropout = int(bs * self.quantizer_dropout)
            n_imps = int(bs) - n_full - n_dropout
            ## 0~n_imps: using importance map
            ## n_imps ~ n_imps + n_dropout: using random masking
            ## n_imps + n_dropout ~ : using full codebook
            n_quantizers[n_imps:n_imps + n_dropout] = dropout[:n_dropout]
        else:
            n_quantizers = self.n_codebooks + 1

            
        if mode == 'VBR':
            if not self.training:
                n_full = n_dropout = 0
                n_imps = bs
        else:
            n_full = bs
            n_dropout = n_imps = 0
        
        for i, quantizer in enumerate(self.quantizers):
            if mode == 'CBR':
                if i >= n_quantizers:
                    break
            
            if self.use_framewise_masking:
                mask_dropout = (torch.full((bs, 1, frames), fill_value=i) < n_quantizers).to(z).float()
            else:
                mask_dropout = (torch.full((bs, 1, 1), fill_value=i) < n_quantizers).to(z).float()
                mask_dropout = mask_dropout.expand(bs, 1, frames)
            
            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(residual, loss_per_frame=True)
            
            if mask_imp is not None:
                mask = mask_imp[:, i, :].unsqueeze(1)
            else:
                mask = torch.ones((bs, 1, frames)).to(z)  ## for CBR mode
            
            if self.training:
                tempmask1 = torch.zeros((bs, 1, frames), requires_grad=False).to(z)
                tempmask2 = torch.zeros((bs, 1, frames), requires_grad=False).to(z)
                tempmask3 = torch.zeros((bs, 1, frames), requires_grad=False).to(z)
                tempmask1[:n_imps] = 1
                tempmask2[n_imps:n_imps+n_dropout] = 1
                tempmask3[n_imps+n_dropout:] = 1
                
                mask = mask * tempmask1
                mask = mask + mask_dropout * tempmask2 ## dropout 영역
                mask = mask + tempmask3 ## full codebook 영역
                assert len(mask.shape) == 3
            
            z_q = z_q + z_q_i * mask
            residual = residual - z_q_i  ## we do not have to apply mask, because once z_q_i is masked, then we don't use this residual anymore.
            
            commitment_loss += (commitment_loss_i * mask.detach()).mean()
            codebook_loss += (codebook_loss_i * mask.detach()).mean()
        
            codebook_indices.append(indices_i)
            latents.append(z_e_i)
            
        # import pdb; pdb.set_trace()
        
        codes = torch.stack(codebook_indices, dim=1)
        latents = torch.cat(latents, dim=1)
        if imp_map is not None:
            imp_map_out = imp_map[:n_imps]
        else:
            imp_map_out = None
        
        
        return z_q, codes, latents, commitment_loss, codebook_loss, imp_map_out, mask_imp
    
    
    def from_codes(self, codes: torch.Tensor):
        """
        z_q: (B, 1024, T)
        z_p: (B, 8, T)
        z_q_stack: (B, Nq, 1024, T)

        """
        z_q = 0.0
        z_p = []
        z_q_stack = []
        n_codebooks = codes.shape[1]  # codes: (B, N, T)
        for i in range(n_codebooks):
            z_p_i = self.quantizers[i].decode_code(codes[:, i, :]) # z_p_i: (B, 8, T)
            z_p.append(z_p_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)  # z_q_i: (B, 1024, T)
            z_q = z_q + z_q_i
            z_q_stack.append(z_q_i)
            
        z_q_stack = torch.stack(z_q_stack, dim=1)  # z_q_stack: (B, Nq, 1024, T)
        return z_q, torch.cat(z_p, dim=1), z_q_stack, codes
    
