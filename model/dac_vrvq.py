from dac_repo.dac.model.dac import *
from .quantizer_vbr import VBRResidualVectorQuantize


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int=64,
        strides: List[int]=[2, 4, 8, 8],
        latent_dim: int=512,
    ):
        super().__init__()
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]
        
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]
        self.block += [
            Snake1d(d_model),
            WNConv1d(d_model, latent_dim, kernel_size=3, padding=1),
        ]
        
        self.block = nn.Sequential(*self.block)
    
    def forward(self, x, return_feat=False):
        num_blocks = len(self.block)
        for i, layer in enumerate(self.block):
            x = layer(x)
            if i == num_blocks - 3 and return_feat:
                feat = x
        out = x
        if return_feat:
            return out, feat
        return out
        


class DAC_VRVQ(BaseModel, CodecMixin):
    def __init__(
        self,
        encoder_dim: int = 64, 
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: float = 0.0,  ## quantizer dropout in original paper
        sample_rate: int = 44100,
        
        full_codebook_rate: float=0.0,  ## rate of samples to use full number of codebooks
        use_framewise_dropout: bool=False, ## Apply random quantizer dropout to each frame
        level_min: float=1.0, ## minimum Scale factor
        level_max: float=48.0, ## maximum Scale factor
        operator_mode: str = "scaling", ## in ["scaling", "exponential", "transformed scaling"] ## Paper: scaling
        
        imp2mask_alpha: float = 1.0,
        imp2mask_func: str="logcosh", ## logcosh, square, sigmoid
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))

        self.latent_dim = latent_dim

        self.hop_length = np.prod(encoder_rates)
        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim)

        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        
        self.quantizer = VBRResidualVectorQuantize(
            input_dim=latent_dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
            full_codebook_rate=full_codebook_rate,
            use_framewise_masking=use_framewise_dropout,
            level_min=level_min,
            level_max=level_max,
            operator_mode=operator_mode,
            imp2mask_alpha=imp2mask_alpha,
            imp2mask_func=imp2mask_func,
        )
        
        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
        )
        self.sample_rate = sample_rate
        self.apply(init_weights)
        self.delay = self.get_delay()
        
    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))

        return audio_data
    
    
    def encode(
        self,
        audio_data: torch.Tensor,
        n_quantizers: int = None,
        level: int = None, ## Scale Factor
    ):
        """
        audio_data: (B, 1, T)
        n_quantizers: 
            - Number of quantizers to use.
            - CBR mode if not None.
            
        level:
            - Scale factor for scaling the importance map.
            - VBR mode if not None.
        
        Returns
        =======
        "z": (B, D, T)
            - Quantized continuous representation of input
            - summed
        "codes" : (B, N_q, T)
            - Codebook indices for each codebook
        "latents" : (B, N_q*D, T)
            - Projected latents (continuous representation of input before quantization)
        "vq/commitment_loss" : (1)
        "vq/codebook_loss" : (1)
        
        """
            
        z, feat = self.encoder(audio_data, return_feat=True)
        z, codes, latents, commitment_loss, codebook_loss, imp_map, mask_imp = \
            self.quantizer(
                z, n_quantizers, feat_enc=feat, level=level
            )
        return z, codes, latents, commitment_loss, codebook_loss, imp_map, mask_imp
    
    def decode(self, z: torch.Tensor):
        """
        z: (B, D, T)
            - Quantized continuous representation of input
        
        """
        return self.decoder(z)
    
    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: int = None,
        n_quantizers: int = None,
        level: int = None,
    ):
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)
        z, codes, latents, commitment_loss, codebook_loss, imp_map, mask_imp = \
            self.encode(audio_data, n_quantizers, level)
            
        x = self.decode(z)
        
        return {
            "audio": x[..., :length],
            "z": z,
            "codes": codes,
            "latents": latents,
            "vq/commitment_loss": commitment_loss,
            "vq/codebook_loss": codebook_loss,
            "imp_map": imp_map,
            "mask_imp": mask_imp, ## can be none in CBR mode
        }