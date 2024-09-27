import torch
from einops import rearrange
import math
from audiotools.data.datasets import AudioLoader
from audiotools.core import AudioSignal, util
import torchmetrics
import numpy as np


def logcosh(alpha, pmk):
    """
    For stable training, 
    we divide the calculation into two cases: pmk >= 0 and pmk < 0.
    """
    EPS = 1e-10

    mask1 = pmk >= 0
    pmk1 = pmk * mask1.detach()
    numer1 = math.exp(alpha) + torch.exp(-2*pmk1*alpha)
    denom1 = torch.exp(alpha*(-2*pmk1+1)) + 1
    mask_smooth1 = (torch.log(numer1 + EPS) - torch.log(denom1 + EPS)) / (2*alpha) + 0.5
    

    mask2 = pmk < 0
    pmk2 = pmk * mask2.detach()
    numer2 = torch.exp(alpha*(2*pmk2+1)) + 1
    denom2 = math.exp(alpha) + torch.exp(alpha*2*pmk2)
    mask_smooth2 = (torch.log(numer2 + EPS) - torch.log(denom2 + EPS)) / (2*alpha) + 0.5
    
    mask_smooth = mask_smooth1 * mask1 + mask_smooth2 * mask2
    return mask_smooth



def generate_mask_smooth(x, nq, alpha=1, function="logcosh", shift:float=None):
    device = x.device
    nqs = torch.arange(nq, dtype=torch.float).to(device) # (nq, ), [0, 1, ..., nq-1]
    nqs = rearrange(nqs, 'n -> 1 n 1')
    xmnq = x - nqs # (B, nq, T)
    
    if function=='logcosh':
        mask_smooth = logcosh(alpha, xmnq)
    elif function=='square':
        mask_smooth = torch.clamp(xmnq, 0, 1)
    elif function=='sigmoid':
        mask_smooth = torch.sigmoid(xmnq * alpha)    
    else:
        raise ValueError(f"Invalid function: {function}")
    
    mask_quant = torch.where(xmnq>=0, torch.ones_like(xmnq), torch.zeros_like(xmnq)).float()
    final_mask = mask_smooth + (mask_quant - mask_smooth).detach()
    return final_mask
    
    
class AudioLoaderException(AudioLoader):    
    def __call__(
        self,
        state,
        sample_rate: int,
        duration: float,
        loudness_cutoff: float = -40,
        num_channels: int = 1,
        offset: float = None,
        source_idx: int = None,
        item_idx: int = None,
        global_idx: int = None,
    ):
        if source_idx is not None and item_idx is not None:
            try:
                audio_info = self.audio_lists[source_idx][item_idx]
            except:
                audio_info = {"path": "none"}
        elif global_idx is not None:
            source_idx, item_idx = self.audio_indices[
                global_idx % len(self.audio_indices)
            ]
            audio_info = self.audio_lists[source_idx][item_idx]
        else:
            audio_info, source_idx, item_idx = util.choose_from_list_of_lists(
                state, self.audio_lists, p=self.weights
            )

        path = audio_info["path"]
        signal = AudioSignal.zeros(duration, sample_rate, num_channels)

        if path != "none":
            if offset is None:
                ## Add this code
                # print('Path:', path)
                try:
                    signal = AudioSignal.salient_excerpt(
                        path,
                        duration=duration,
                        state=state,
                        loudness_cutoff=loudness_cutoff,
                    )
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    signal = AudioSignal.zeros(duration, sample_rate, num_channels)
                    print(f"Silence loaded: Signal duration: {signal.duration}")
                    # import pdb; pdb.set_trace()
                    
            else:
                signal = AudioSignal(
                    path,
                    offset=offset,
                    duration=duration,
                )

        if num_channels == 1:
            signal = signal.to_mono()
        signal = signal.resample(sample_rate)

        if signal.duration < duration:
            signal = signal.zero_pad_to(int(duration * sample_rate))

        for k, v in audio_info.items():
            signal.metadata[k] = v

        item = {
            "signal": signal,
            "source_idx": source_idx,
            "item_idx": item_idx,
            "source": str(self.sources[source_idx]),
            "path": str(path),
        }
        if self.transform is not None:
            item["transform_args"] = self.transform.instantiate(state, signal=signal)
        # print("### PATH:", path)
        return item
    




    
def cal_loss(recons, signal, state, loss_fn="mel"):
    # assert loss_fn in ["mel", "stft", "waveform", "SDR", "SI-SDR", "L1", "DAC-SISDR", 
    #                    "ViSQOL", "ViSQOL-speech"]
    if loss_fn == "mel":
        return state.mel_loss(recons, signal).item()
    elif loss_fn == "stft":
        return state.stft_loss(recons, signal).item()
    elif loss_fn == "waveform":
        return state.waveform_loss(recons, signal).item()
    elif loss_fn == "SDR":
        recons = recons.audio_data
        signal = signal.audio_data
        if recons.abs().max() == 0 or signal.abs().max() == 0:
            return np.nan  
        # result = torchmetrics.functional.signal_to_distortion_ratio(recons, signal)
        result = torchmetrics.functional.signal_distortion_ratio(recons, signal)
        result = result.mean().item()
        return result
    elif loss_fn == "SI-SDR":
        recons = recons.audio_data
        signal = signal.audio_data
        # return torchmetrics.functional.si_sdr(recons, signal).item()
        result = torchmetrics.functional.scale_invariant_signal_distortion_ratio(recons, signal)
        result = result.mean().item()
        return result
    elif loss_fn == "L1":
        recons = recons.audio_data
        signal = signal.audio_data
        result = torchmetrics.functional.mean_absolute_error(recons, signal)
        result = result.item()
        return result
    elif loss_fn == "SI-SNR":
        recons = recons.audio_data
        signal = signal.audio_data
        result = torchmetrics.functional.scale_invariant_signal_noise_ratio(recons, signal)
        result = result.mean().item()
        return result
    elif loss_fn == "SNR":
        recons = recons.audio_data
        signal = signal.audio_data
        result = torchmetrics.functional.signal_noise_ratio(recons, signal)
        result = result.mean().item()
        return result
    elif loss_fn == "DAC-SISDR":
        return state.dac_sisdr_loss(signal, recons).item()
    # elif loss_fn == "ViSQOL":
    #     ## resample to 48k
    #     result = metrics.quality.visqol(signal, recons)
    #     if isinstance(result, torch.Tensor):
    #         result = result.item()
    #     return result
    # elif loss_fn == "ViSQOL-speech":
    #     ## resample to 16k
    #     result = metrics.quality.visqol(signal, recons, "speech").item()
    #     if isinstance(result, torch.Tensor):
    #         result = result.item()
    #     return result
    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")
    


def cal_bpf_from_mask(mask, bits_per_codebook):
    """
    mask: (B, Nq, Frames)
    bits_per_codebook: (Nq, )
    """
    bits_per_codebook = torch.tensor(bits_per_codebook, device=mask.device) ## (Nq, )
    bits_per_codebook = rearrange(bits_per_codebook, 'nq -> 1 nq 1')
    mask_bits = mask * bits_per_codebook
    bpf = torch.sum(mask_bits) / (mask.shape[0] * mask.shape[2])
    return bpf.item()