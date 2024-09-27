import argbind
import os
import sys
import warnings
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
from einops import rearrange

from audiotools import AudioSignal
from audiotools import ml
# from audiotools import metrics
from audiotools.core import util
from audiotools.data import transforms
from audiotools.data.datasets import AudioDataset
from audiotools.data.datasets import AudioLoader
# from temp_utils import MyAudioLoader as AudioLoader
from audiotools.data.datasets import ConcatDataset
from audiotools.ml.decorators import timer

import math

sys.path.append(str(Path(__file__).parent.parent))
import dac_repo.dac as dac
from model.utils import cal_loss, cal_bpf_from_mask, generate_mask_smooth
from model.utils import AudioLoaderException as AudioLoader
from model.dac_vrvq import DAC_VRVQ


import json
import matplotlib.pyplot as plt
from matplotlib import gridspec
import librosa
# import torchmetrics
import time

# from sweetdebug import sweetdebug; sweetdebug()

warnings.filterwarnings("ignore", category=UserWarning)
torch.backends.cudnn.benchmark = bool(int(os.getenv("CUDNN_BENCHMARK", 1)))
Accelerator = argbind.bind(ml.Accelerator, without_prefix=True)

DAC = argbind.bind(DAC_VRVQ)

AudioDataset = argbind.bind(AudioDataset, 'test')
AudioLoader = argbind.bind(AudioLoader, 'test')

# filter_fn = lambda fn: hasattr(fn, "transform") and fn.__qualname__ not in [
#     "BaseTransform",
#     "Compose",
#     "Choose",
# ]
# tfm = argbind.bind_module(transforms, 'test', filter_fn=filter_fn)

## Loss
filter_fn = lambda fn: hasattr(fn, "forward") and "Loss" in fn.__name__
losses = argbind.bind_module(dac.nn.loss, filter_fn=filter_fn)

def get_infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


# @argbind.bind('test')
# def build_transform(
#     augment_prob: float = 1.0,
#     preprocess: list = ["Identity"],
#     augment: list = ["Identity"],
#     postprocess: list = ["Identity"],
# ):
#     to_tfm = lambda l: [getattr(tfm, x)() for x in l]
#     ##!! No normalization or augment
#     preprocess = augment = postprocess = ["Identity"] 
    
#     preprocess = transforms.Compose(*to_tfm(preprocess), name="preprocess")
#     augment = transforms.Compose(*to_tfm(augment), name="augment", prob=augment_prob)
#     postprocess = transforms.Compose(*to_tfm(postprocess), name="postprocess")
#     transform = transforms.Compose(preprocess, augment, postprocess)
#     return transform

@argbind.bind("test")
def build_dataset(
    sample_rate: int,
    folders: dict = None,
    data_type: str = 'full', ## 'full', 'speech', 'music', 'general
):
    datasets = []
    for k, v in folders.items():
        if data_type=='full':
            pass
            print('### Using full data')
        elif data_type=='speech':
            if 'speech' not in k:
                continue
            print('### Using speech data')
        elif data_type=='music':
            if 'music' not in k:
                continue
            print('### Using music data')
        elif data_type=='general':
            if 'general' not in k:
                continue
            print('### Using general data')
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        loader = AudioLoader(sources=v)
        dataset = AudioDataset(loader, sample_rate, transform=None,)
        datasets.append(dataset)
    
    dataset = ConcatDataset(datasets)
    return dataset

@dataclass
class State:
    generator: DAC
    stft_loss: losses.MultiScaleSTFTLoss
    mel_loss: losses.MelSpectrogramLoss
    waveform_loss: losses.L1Loss
    dac_sisdr_loss: losses.SISDRLoss ##(references, estimates)
    
    test_data: AudioDataset
    tag:str
    data_type: str
    batch_size: int = None

@argbind.bind(without_prefix=True)
def load(
    args,
    accel: ml.Accelerator,
    save_path: str,
    resume: bool = True,
    tag: str = "latest",
    load_weights: bool = False,
    load_pretrained: bool = False,
    data_type: str = 'full',
):
    generator, g_extra = None, {}
    resume = True
    load_weights = False  ## True=> args.load 세팅값을 안불러옴. why?
    load_pretrained = False
    # tag = "best"

    kwargs = {
        "folder": f"{save_path}/{tag}",
        "map_location": "cpu",
        # "package": not load_weights  ## False => args.load 세팅값을 안불러옴. why?
        "package": False,
    }
    print(f"Resuming from {str(Path('.').absolute())}/{kwargs['folder']}")
    if (Path(kwargs["folder"]) / "dac_vrvq").exists():
        print('### Loading from folder: ',  (Path(kwargs["folder"]) / "dac_vrvq"))
        generator, g_extra = DAC.load_from_folder(strict=True, **kwargs)
        # import pdb; pdb.set_trace()
    import pdb; pdb.set_trace()
        
    generator = DAC() if generator is None else generator
    generator = accel.prepare_model(generator)
    sample_rate = accel.unwrap(generator).sample_rate
    with argbind.scope(args, "test"):
        test_data = build_dataset(sample_rate, data_type=data_type)
        
    waveform_loss = losses.L1Loss()
    stft_loss = losses.MultiScaleSTFTLoss()
    mel_loss = losses.MelSpectrogramLoss()
    dac_sisdr_loss = losses.SISDRLoss()
    
    return State(
        generator=generator,
        stft_loss=stft_loss,
        mel_loss=mel_loss,
        waveform_loss=waveform_loss,
        dac_sisdr_loss=dac_sisdr_loss,
        test_data=test_data,
        tag=tag,
        data_type=data_type
    )



@timer()
@torch.no_grad()
def test_loop(batch, state, accel, save_figs=None, no_visqol=True):
    st = time.time()
    state.generator.eval()
    batch = util.prepare_batch(batch, accel.device)

    signal = batch["signal"].clone()
    n_batch = signal.shape[0]
    audio_gt = signal.audio_data
    sr = signal.sample_rate
    out_fixed = state.generator(audio_gt,
                                sr,
                                n_quantizers=state.generator.n_codebooks+1,)                           
    decoder = state.generator.decode
    quantizer = state.generator.quantizer
    codebook_size_list = state.generator.quantizer.codebook_size
    if isinstance(codebook_size_list, int):
        codebook_size_list = [codebook_size_list] * len(quantizer.quantizers)
    assert len(codebook_size_list) == len(quantizer.quantizers)
    bits_per_codebook = [math.log2(codebook_size) for codebook_size in codebook_size_list]
    # out: audio, z, codes, latents, vq/commitment_loss, vq/codebook_loss
    
    ## Calculate many kinds of audio waveform
    codes_fixed = out_fixed["codes"]
    n_codebooks = len(quantizer.quantizers)
    bs = codes_fixed.shape[0]
    device = codes_fixed.device
    
    level_list = [4, 8, 12, 14, 16, 18, 20, 24, 28, 32]
    # level_list = [4,  8, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 48, 52, 56, 60]
    # level_list = [8, 12, 16, 20, 24, 28, 32, 40, 48, 56]
    # level_list = [] ## save fig만 하고 싶을때
    level_list = [round(level, 2) for level in level_list]
    level_list_figs=[6, 8, 14, 20, 26]
    # level_list_figs = [0.1, 0.5, 1, 4, 8] ## exp2 / straight2
    
    if save_figs:
        os.makedirs(save_figs, exist_ok=True)
        save_idx = 0
        save_path_png = f'{save_figs}/mask_spec_{save_idx}.png'
        while os.path.exists(save_path_png):
            save_idx += 1
            save_path_png = f'{save_figs}/mask_spec_{save_idx}.png'    
            
        for nb in range(1):
            # fig, axes = plt.subplots(len(level_list_figs)+2, 1, figsize=(8, 20))            
            fig = plt.figure(figsize=(2, 4))
            gs = gridspec.GridSpec(nrows=len(level_list_figs)+1,
                                   ncols=1,
                                   height_ratios=[1 for _ in range(len(level_list_figs))]+[2],
            )
                
            for jj, level in enumerate(level_list_figs):
                out_lev = state.generator(audio_gt,
                                          sr,
                                          level=level)
                msk = out_lev["mask_imp"][nb]
                
                bpf = cal_bpf_from_mask(msk.unsqueeze(0), bits_per_codebook)
                recon = out_lev["audio"]
                recon = AudioSignal(recon, sr)
            
                loss_sisdr = cal_loss(recon, signal, state, "SI-SDR")
                msk = msk.cpu().numpy()

                ax = plt.subplot(gs[jj])
                ax.imshow(msk, cmap='viridis', interpolation='none', aspect='auto')
                ax.set_yticks(np.arange(0, n_codebooks))
                ax.invert_yaxis()
                bps = (bpf+3) * 44100 / 512
                kbps = bps/1000
                ax.set_title(f'{loss_sisdr:.2f} || {kbps:.2f}kbps')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.set_yticks([])
                if jj == len(level_list_figs) -1 :
                    # ax.get_xaxis().set_visible(True)
                    ax.set_xticks([])
                else:
                    ax.set_xticks([])
                    
            ax = plt.subplot(gs[-1])
            recon_full = out_fixed["audio"]
            recon_full = AudioSignal(recon_full, sr)
            recon_ad = recon_full[nb].cpu() ## ?? (1, 1, x ,x) -> (1, 1, x, x)
            ref_re = recon_ad.magnitude.max()
            logmag_re = recon_ad.log_magnitude(ref_value=ref_re)
            logmag_re = logmag_re.numpy()[0][0]
            librosa.display.specshow(
                logmag_re,
                x_axis='time',
                sr=signal.sample_rate,
                # ax=axes[-1],
                ax=ax
            )
            loss_sisdr = loss_sisdr = cal_loss(recon_full, signal, state, "SI-SDR")
            kbps = 44100/512*80/1000
            ax.set_title(f'{loss_sisdr:.2f} || {kbps:.2f}kbps')
            ax.set_yticks([])
            
            save_path_wav = f'{save_figs}/audio_{save_idx}.wav'
            recon = out_lev["audio"]
            recon = AudioSignal(recon, sr)
            ad = signal[nb].cpu()
            ad.write(save_path_wav)
            recon[nb].cpu().write(save_path_wav.replace('.wav', '_recons.wav'))
        
            
            # ### PLOT spec of input audio
            # ref = ad.magnitude.max()
            # logmag = ad.log_magnitude(ref_value=ref)
            # # logmag = logmag.numpy()[0].mean(axis=0)
            # logmag = logmag.numpy()[0]
            # librosa.display.specshow(
            #     logmag,
            #     x_axis='time',
            #     # y_axis='linear',
            #     sr=signal.sample_rate,
            #     # ax=axes[-1]
            #     ax=ax
            # )
            
            plt.tight_layout()
            plt.subplots_adjust(hspace=1)
            plt.savefig(save_path_png, bbox_inches='tight', pad_inches=0)
            plt.close()
            
    _, _, z_q_stack_fixed, _ = quantizer.from_codes(codes_fixed) 
    ## => z_q_stack_fixed: (B, Nq, 1024, T)
    
    output = {}
    
    loss_fn_list = ['mel', 'waveform', 'DAC-SISDR', 'stft', 'SNR', 'SI-SNR', 'SDR', 'SI-SDR']
    
    # if not no_visqol:
    #     loss_fn_list += ['ViSQOL']
    #     if state.data_type == 'speech':
    #         loss_fn_list.append('ViSQOL-speech')
    if not no_visqol:
        raise NotImplementedError 
        if state.data_type == 'speech':
            loss_fn_list.append('ViSQOL-speech')
        else:
            loss_fn_list.append('ViSQOL')
    
    """ CBR Mode Evaluation"""
    for i in range(n_codebooks):
        z_q_sum = torch.sum(z_q_stack_fixed[:, :i+1], dim=1)
        recons = decoder(z_q_sum)
        recons = recons[..., :signal.audio_data.shape[-1]]
        recons = AudioSignal(recons, signal.sample_rate)
        for loss_fn in loss_fn_list:
            st = time.time()
            loss = cal_loss(recons, signal, state, loss_fn)
            bpf = np.sum(bits_per_codebook[:i+1]).item()
            output[f'{loss_fn}/fixed_{i}'] = loss
            output[f'bpf/fixed_{i}'] = bpf
            # print('loss fn:', loss_fn, '|| fixed level:', i, '|| loss:', loss, '|| bpf:', bpf, '|| time:', round(time.time() - st, 2))
    
    ## Variable codebook:
    #### simple thresholding / sampling
    """ VBR Mode Evaluation"""
    audio_gt_proc = state.generator.preprocess(audio_gt, sr)
    out_enc, feat_enc = state.generator.encoder(audio_gt_proc, return_feat=True) # feat_enc: (B, 1024, T)
    imp_map = state.generator.quantizer.imp_subnet(feat_enc) # imp_map: (B, 1, T)
    for level in level_list:
        imp_map_scaled = imp_map * level
        mask_map = generate_mask_smooth(
            x=imp_map_scaled,
            nq=n_codebooks,
            alpha=state.generator.quantizer.imp2mask_alpha,
            function=state.generator.quantizer.imp2mask_func,
        ) # (B, Nq, T)
        bpf = cal_bpf_from_mask(mask_map, bits_per_codebook)
        ### z_q_stack_fixed: (B, Nq, 1024, T)
        z_q_masked = z_q_stack_fixed * rearrange(mask_map, 'b nq t -> b nq 1 t')
        z_q_masked_sum = torch.sum(z_q_masked, dim=1)
        recons = decoder(z_q_masked_sum)
        recons = recons[..., :signal.audio_data.shape[-1]]
        recons = AudioSignal(recons, signal.sample_rate)

        for loss_fn in loss_fn_list:
            output[f'{loss_fn}/var_{level:1f}'] = cal_loss(recons, signal, state, loss_fn)
            output[f'bpf/var_{level:1f}'] = bpf+math.ceil(math.log2(n_codebooks))
            ## math.ceil(math.log2(n_codebooks)): transmission cost

    return output
    
    
@argbind.bind(without_prefix=True)
def eval(
    args,
    accel: ml.Accelerator,
    seed: int = 0,
    save_path: str = "ckpt",
    batch_size_test: int = 1,
    num_workers: int = 8,
    save_result_dir: str = None,
    no_visqol: bool = True,
):
    batch_size = batch_size_test
    print('Batch size:', batch_size)
    util.seed(seed)
    state = load(args, accel, save_path)
    test_loader = accel.prepare_dataloader(
        state.test_data,
        start_idx = 0,
        num_workers = num_workers,
        batch_size = batch_size,
        collate_fn = state.test_data.collate,
        persistent_workers = True if num_workers > 0 else False,
        shuffle=False
    )
    assert save_result_dir is not None
    state.batch_size = batch_size
    
    name = os.path.basename(save_path).split('.')[0]
    save_figs_path = os.path.join(save_result_dir, 'figs', name)
    print("## Test Set Length:", len(test_loader))
    save_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    data_iter = iter(test_loader)
    count = -1
    t_infer = None

    while True:
        count += 1
        if count >= len(test_loader):
            break
        st = time.time()
        try:
            batch = next(data_iter)
        except:
            print("Error in batch=next(data_iter)")
            continue
        
        if count in save_indices:
            save_figs = save_figs_path
            print('save_figs: ', count, save_figs_path)
        else:
            save_figs = None
            # assert False
        
        try:
            if count == 0:
                output = test_loop(batch, state, accel, save_figs=save_figs, no_visqol=no_visqol)
                total_output = {k: [v] for k, v in output.items()}
            else:
                output = test_loop(batch, state, accel, save_figs=save_figs, no_visqol=no_visqol)
                total_output = {k: v + [output[k]] for k, v in total_output.items()}
            if t_infer is None:
                t_infer = time.time() - st
            else:
                rr = 0.4
                t_infer = rr * (time.time() - st) + (1-rr) * t_infer
            # t_infer = round(time.time() - st, 2)
            eta = round(t_infer * (len(test_loader) - count), 2)
            eta_h = round(eta/3600, 2)
            print(f"Count: {count}/{len(test_loader)} || Infer time: {t_infer} || ETA(h): {eta_h}")
            if count % 100==0:
                print('total_output:', {k: np.nanmean(v) for k, v in total_output.items()})
        except Exception as e:
            print("Inference error: ", e)
            if count == 0:
                assert False, "First sample error"
            
    total_output_mean = {k: np.nanmean(v) for k, v in total_output.items()}
    
    os.makedirs(save_result_dir, exist_ok=True)
    with open(Path(save_result_dir) / f'{name}_{state.tag}_{len(test_loader)}_dtype{state.data_type}.json', 'w') as f:
        json.dump(total_output_mean, f, indent=4)


if __name__=='__main__':
    args = argbind.parse_args()
    args['args.debug'] = int(os.getenv("LOCAL_RANK", 0)) == 0
    # print(args)
    # gpu_num = args["args.unknown"][0]
    # os.environ['CUDA_VISIBLE_DEVICES']=f"{gpu_num}"
    with argbind.scope(args):
        with Accelerator() as accel:
            if accel.local_rank != 0:
                sys.tracebacklimit = 0
            eval(args, accel)
    