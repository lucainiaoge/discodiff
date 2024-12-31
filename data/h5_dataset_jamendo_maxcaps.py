import sys
import argparse
from pathlib import Path
import os
import time
import math
import random
import numpy as np
import librosa

import h5py
import json
import audio_metadata
from tinytag import TinyTag

import torch
import torchaudio
from torch.utils.data import Dataset
from datasets import load_dataset

from transformers import AutoTokenizer

from datetime import datetime, timedelta
from typing import Optional, Tuple, Union, List

try:
    from utils import convert_audio, int16_to_float32, float32_to_int16
except:
    from ..utils import convert_audio, int16_to_float32, float32_to_int16

SPECTROGRAM_NORMALIZATION_FACTOR = 100
CLAP_DIM = 512
KEY_LABELS = ['A major', 'Bb major', 'B major', 'C major', 'Db major',
              'D major', 'Eb major', 'E major', 'F major', 'F# major',
              'G major', 'Ab major', 'A minor', 'Bb minor', 'B minor',
              'C minor', 'C# minor', 'D minor', 'D# minor', 'E minor',
              'F minor', 'F# minor', 'G minor', 'G# minor']

def get_true_audio_name(audioname):
    # return audioname.split("_")[0]
    return audioname

'''
    Read audio files and convert them into dac and/or encodec tokens,
    Supports CLAP model: if specified, and text json is provided, will also extract audio and text CLAP features,
    Additional supports: key and tempo estimation by madmom; spectrogram and chroma

    This is only a parent class. DO NOT USE IT DIRECTLY

    Parameters:
    
    - dac_model: should be an instance obtained through the following code
        import dac
        dac_model_path = dac.utils.download(model_type="44khz")
        dac_model = dac.DAC.load(dac_model_path).to(device)
        
    - encodec_model: should be an instance obtained through the following code
        from audiocraft.models import CompressionModel
        encodec_model = CompressionModel.get_pretrained('facebook/encodec_32khz').to(device)

    - clap_model: should be an instance obtained through the following code
        clap_model = ClapModel.from_pretrained("laion/larger_clap_music").to(device)
        processor = ClapProcessor.from_pretrained("laion/larger_clap_music")
        clap_model = [clap_model, processor]

    - min_sec: minimum length (in sec) of the audio file
    - min_size: minimum size (in KB) of the audio file
    - start_silence_sec: read audio start time (assume there is silence before this time)
    - sample_rate: the sr reading audio file for spectrogram calculation
    - chunk_dur_sec to mel_hop_size: parameters calculating mel-spectrogram

    - exts: audio file extensions supported
    - percent_start_end: the percentage of files to (partially) read in the audio_folder, this is defined for parallel running
    - no_audio_chunk: if set to True, no audio chunk features will be extracted/updated when saving h5, but only the meta features like text
'''
class DacEncodecClapDatasetJamendocaps(Dataset):
    def __init__(
        self,
        hf_dataset,
        dac_model = None,
        encodec_model = None,
        clap_model = None,
        
        min_sec: float = 10.0,
        min_size: float = 300.0,  # in KB
        start_silence_sec: float = 0,
        sample_rate: int = 48000,
        chunk_dur_sec: float = 60,
        mel_freq_bins: int = 128,
        mel_window_size: int = 4096,
        mel_hop_size: int = 2048,
        
        exts: Union[List[str], Tuple[str]] = ['mp3', 'wav'],
        percent_start_end: Tuple[Optional[int]] = (None, None),
        no_audio_chunk: bool = False,
    ):
        super().__init__()
        if 'train' in hf_dataset:
            self.hf_dataset = hf_dataset['train']
        else:
            self.hf_dataset = hf_dataset
        self.dac_model = dac_model
        self.encodec_model = encodec_model
        self.clap_model = clap_model

        # identify the models to use
        if dac_model is not None:
            self.sample_rate_dac = dac_model.sample_rate
        if encodec_model is not None:
            self.encodec_device = next(encodec_model.parameters()).device
            self.sample_rate_encodec = encodec_model.sample_rate
        if clap_model is not None:
            self.clap_model, self.clap_processor = clap_model[0], clap_model[1]
            self.clap_device = next(self.clap_model.parameters()).device
            self.sample_rate_clap = 48000

        self.use_encodec = False if self.encodec_model is None else True
        self.use_dac = False if self.dac_model is None else True
        self.use_clap = False if self.clap_model is None else True
        
        self.no_audio_chunk = no_audio_chunk
        if not self.no_audio_chunk and not self.use_dac and not self.use_encodec and not self.use_clap:
            print("Neither DAC or Encodec is given. Switched into no_audio_chunk mode")
            self.no_audio_chunk = True
        
        # set parameters
        self.sample_rate = sample_rate
        self.min_sec = min_sec

        self.mel_freq_bins = mel_freq_bins
        self.mel_window_size = mel_window_size
        self.mel_hop_size = mel_hop_size
        
        self.start_silence_sec = start_silence_sec
        self.chunk_dur_sec = chunk_dur_sec
        
        self.recompute_feature = False # enable it if feature need updating
    
    def __len__(self):
        if hasattr(self, "audio_ids"):
            return len(self.audio_ids)
        else:
            return len(self.hf_dataset)

    def get_rvq_latent_from_wav(self, wav: torch.Tensor, sample_rate: Optional[int] = None):
        if sample_rate is None:
            sample_rate = self.sample_rate
        if self.use_dac:
            wav_dac = convert_audio(
                wav, sample_rate, self.sample_rate_dac, 1
            ).unsqueeze(0).to(self.dac_model.device) # [1, 1, n_samples]
        if self.use_encodec:
            wav_encodec = convert_audio(
                wav, sample_rate, self.sample_rate_encodec, self.encodec_model.channels
            ).unsqueeze(0).to(self.encodec_device)

        with torch.no_grad():
            if self.use_dac:
                _, dac_codes, dac_latents, _, _ = self.dac_model.encode(wav_dac) # [1, n_codebooks = 9, n_frames], [1, 8x9, n_frames]
            else:
                dac_codes, dac_latents = [None], [None]
            if self.use_encodec:
                encodec_codes = self.encodec_model.encode(wav_encodec)[0]
                encodec_latents = self.encodec_model.model.quantizer.decode(encodec_codes).reshape(-1, encodec_codes.shape[-1])
            else:
                encodec_codes, encodec_latents = [None], None

        return dac_codes[0], dac_latents[0], encodec_codes[0], encodec_latents
        # (9, n_frames) (int), (72, n_frames) (float), (4, n_frames) (int), (4*128, n_frames) (float)

    @torch.no_grad()
    def get_rvq_latents_clap_from_wav(self, wav: torch.Tensor, sample_rate: Optional[int] = None):
        wav_clap = convert_audio(wav, sample_rate, self.sample_rate_clap, 1).squeeze(0) # (n_samples,) -> (1, n_samples')

        dac_rvq, dac_latents, encodec_rvq, encodec_latents = self.get_rvq_latent_from_wav(wav, sample_rate)
        if self.use_clap:
            # clap_emb = self.clap_model.get_audio_embedding_from_data(x=wav_clap, use_tensor=True)
            clap_inputs = self.clap_processor(audios=wav_clap, sampling_rate=self.sample_rate_clap, return_tensors="pt").to(self.clap_device)
            clap_emb = self.clap_model.get_audio_features(**clap_inputs) # (1, 512)
        else:
            clap_emb = None

        return dac_rvq, dac_latents, encodec_rvq, encodec_latents, clap_emb
    
    def get_wav_through_audio_id(self, audio_id: int, audio_obj = None):
        if audio_obj is None:
            audio_obj = self.hf_dataset[audio_id]['audio']
        
        wav = torch.tensor(audio_obj['array']).to(torch.float32).unsqueeze(0)
        print(wav.shape)
        sr = int(audio_obj['sampling_rate'])
        return wav, sr
        
    @torch.no_grad()
    def __getitem__(self, index: int):
        wav, sample_rate = self.get_wav_through_audio_id(index)
        return self.get_rvq_latents_clap_from_wav(wav, sample_rate)

    def get_text_feat_meta_data_dict(self, audio_name: str):
        pass
        
    '''
        Save a single audio file to audio_name.hdf5, which is organized as such

        'metadata':
            'text' (if provided): encoded text str
            'text_clap' (if provided): int numpy array (encoded from float array) in size (512,)
            'salmonn_text' (if provided): encoded text str
            'salmonn_text_clap' (if provided): int numpy array (encoded from float array) in size (512,)
            ...(other text features to add)
            'chatgpt_texts' (if provided): list of encoded text str
            'chatgpt_text_clap' (if provided): int numpy array (encoded from float array) in size (N_chatgpt_results, 512)
            'madamom_key' (if provided): int
            'madamom_tempo' (if provided): int

        '0' (chunk number and so on):
            'dac_rvq' (if provided): int numpy array in shape (9, L)
            'dac_frame_len' (if provided): int L
            'dac_latents' (if provided): float numpy array in shape (72, L)
            'encodec_rvq' (if provided): int numpy array in shape (K, L')
            'encodec_frame_len' (if provided): int L
            'encodec_latents' (if provided): float numpy array in shape (K*D, L')
            'audio_clap' (if provided): int numpy array (encoded from float array) in size (512,)
            'spectrogram': int numpy array (encoded from float array) in size (num_freq_bins, n_frames)
            'chroma': int numpy array (encoded from float array) in size (12, n_frames)

        '1' (same format as 0):
        ... (all the way to the last chunk)
    '''
    @torch.no_grad()
    def save_audio_text_to_h5_single(
        self, 
        audio_id_parsed: int, 
        target_dir: Union[str, os.PathLike], 
        skip_existing: bool = False,
        skip_existing_strong: bool = False,
        skip_recent: bool = False,
    ):
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            print("Target dir", target_dir, "is created!")

        audio_id = self.audio_ids[audio_id_parsed]
        audio_obj = self.hf_dataset[audio_id]['audio']
        filename = audio_obj['path']
        basename = os.path.splitext(filename)[0]
        out_h5_filename = f"{basename}.hdf5"
        out_h5_path = os.path.join(target_dir, out_h5_filename)
        if os.path.exists(out_h5_path) and skip_existing_strong:
            print(f"{out_h5_path} exists, skipped.")
            return

        if os.path.exists(out_h5_path) and skip_recent:
            modified_datetime = datetime.strptime(time.ctime(os.path.getmtime(out_h5_path)), "%c")
            current_datetime = datetime.now()
            timediff = current_datetime - modified_datetime
            if timediff < timedelta(days=2):
                print(f"{out_h5_path} exists and is recently modified, skipped.")
                return

        print("\n ===================== \n")

        print(audio_id_parsed, ": Parsing", basename)

        # save text feat if child class has text_feat_metadata defined (read from json for example)
        if hasattr(self, 'text_feat_metadata') and self.clap_model is not None:
            print("------saving text feat meta data------")
            with h5py.File(out_h5_path, 'a') as f:
                audio_name = get_true_audio_name(basename)
                data_dict = self.get_text_feat_meta_data_dict(audio_name)

                if "metadata" not in f:
                    grp = f.create_group("metadata")
                else:
                    grp = f["metadata"]

                for attr in data_dict:
                    if attr not in grp:
                        grp.create_dataset(attr, data=data_dict[attr])
                        print("archived", attr, "with type", type(data_dict[attr]), "into hdf5 file")
                    else:
                        del grp[attr]
                        grp.create_dataset(attr, data=data_dict[attr])
                        print("updated", attr, "with type", type(data_dict[attr]), "into hdf5 file")
                print("\n")

        if self.no_audio_chunk:
            return

        wav_all, sample_rate = self.get_wav_through_audio_id(audio_id, audio_obj = audio_obj)
        chunk_len = int(self.chunk_dur_sec * sample_rate)
        num_chunks = int(wav_all.shape[-1] / chunk_len)
        chunk_starts = np.arange(num_chunks) * chunk_len
        for relative_index in range(num_chunks):
            chunkname = f'{relative_index}'
            print(audio_id_parsed, ": Parsing", basename, "chunk num", relative_index)

            # save audio
            print("------saving audio data------")
            with h5py.File(out_h5_path, 'a') as f:
                if chunkname in f and skip_existing:
                    print("chunk already exists and skipped")
                    print("\n")
                    continue
                    
                this_start_time = chunk_starts[relative_index]
                wav = wav_all[:, this_start_time:this_start_time+chunk_len]
                if (chunkname not in f) or self.recompute_feature:
                    print("Computing features...")
                    wav_spec = convert_audio(
                        wav, sample_rate, self.sample_rate, 1
                    ).squeeze().numpy()  # [n_samples]
    
                    powergram = librosa.stft(
                        y=wav_spec, n_fft=self.mel_window_size,
                        hop_length=self.mel_hop_size, window='hann', center=True,
                        pad_mode='constant'
                    )
                    powergram = np.abs(powergram)**2
                    spectrogram = librosa.feature.melspectrogram(
                        S=powergram, sr=self.sample_rate, n_mels=self.mel_freq_bins
                    )
                    chromagram = torch.tensor(librosa.feature.chroma_stft(S=powergram, sr=self.sample_rate))
                    spectrogram_dB = librosa.power_to_db(spectrogram)
                    spectrogram_dB = torch.tensor(spectrogram_dB / SPECTROGRAM_NORMALIZATION_FACTOR)
                    
                    dac_rvq, dac_latents, encodec_rvq, encodec_latents, audio_clap = self.get_rvq_latents_clap_from_wav(wav, sample_rate)
                    
                else: # in case processing an h5 file where features are already computed
                    print("Found existing features, no feature computation is done")
                    spectrogram_dB = torch.tensor(np.array(f[chunkname]['spectrogram'])) if 'spectrogram' in f[chunkname] else None
                    chromagram = torch.tensor(np.array(f[chunkname]['chroma'])) if 'chroma' in f[chunkname] else None
                    
                    dac_rvq = torch.tensor(np.array(f[chunkname]['dac_rvq'])) if 'dac_rvq' in f[chunkname] else None
                    dac_latents = torch.tensor(np.array(f[chunkname]['dac_latents'])) if 'dac_latents' in f[chunkname] else None
                    encodec_rvq = torch.tensor(np.array(f[chunkname]['encodec_rvq'])) if 'encodec_rvq' in f[chunkname] else None
                    encodec_latents =torch.tensor( np.array(f[chunkname]['encodec_latents'])) if 'encodec_latents' in f[chunkname] else None
                    audio_clap = torch.tensor(int16_to_float32(np.array(f[chunkname]['audio_clap']))).unsqueeze(0) if 'audio_clap' in f[chunkname] else None
                    
                
                print(f"Got chunk number {relative_index}")
                data_dict = {}
                if spectrogram_dB is not None:
                    data_dict['spectrogram'] = spectrogram_dB.cpu().numpy().astype(np.float32)
                    print("Got mel spectrogram with shape", spectrogram_dB.shape)
                if chromagram is not None:
                    data_dict['chroma'] = chromagram.cpu().numpy().astype(np.float32)
                    print("Got chromagram with shape", chromagram.shape)
                if dac_rvq is not None:
                    data_dict['dac_rvq'] = dac_rvq.cpu().numpy().astype(int)
                    data_dict['dac_frame_len'] = dac_rvq.shape[-1]
                    print("Got dac RVQ code with shape", data_dict['dac_rvq'].shape)
                if dac_latents is not None:
                    data_dict['dac_latents'] = dac_latents.cpu().numpy().astype(np.float32)
                    print("Got dac latent with shape", data_dict['dac_latents'].shape)
                if encodec_rvq is not None:
                    data_dict['encodec_rvq'] = encodec_rvq.cpu().numpy().astype(int)
                    data_dict['encodec_frame_len'] = encodec_rvq.shape[-1]
                    print("Got encodec RVQ code with shape", data_dict['encodec_rvq'].shape)
                if encodec_latents is not None:
                    data_dict['encodec_latents'] = encodec_latents.cpu().numpy().astype(np.float32)
                    print("Got encodec latent with shape", data_dict['encodec_latents'].shape)
                if audio_clap is not None:
                    data_dict['audio_clap'] = float32_to_int16(audio_clap.cpu().numpy())[0, :]
                    print("Got audio clap with shape", data_dict['audio_clap'].shape)

                if chunkname not in f:
                    grp = f.create_group(chunkname)
                else:
                    grp = f[chunkname]

                for attr in data_dict:
                    if attr not in grp:
                        grp.create_dataset(attr, data=data_dict[attr])
                        print("archived", attr, "with type", type(data_dict[attr]), "into hdf5 file")
                    else:
                        del grp[attr]
                        grp.create_dataset(attr, data=data_dict[attr])
                        print("updated", attr, "with type", type(data_dict[attr]), "into hdf5 file")
                print("\n")

    '''
        Run save_audio_text_to_h5_single over all audio files in a folder, and save all h5 files to target_dir
    '''
    def save_audio_text_to_h5_multiple(
        self, 
        target_dir: Union[str, os.PathLike], 
        skip_existing: bool = False, 
        skip_existing_strong: bool = False, 
        file_id_sel: bool = None
    ):
        if file_id_sel is None:
            file_id_sel = range(len(self.audio_ids))
        for audio_id_parsed in file_id_sel:
            try:
                self.save_audio_text_to_h5_single(
                    audio_id_parsed, 
                    target_dir, 
                    skip_existing=skip_existing, 
                    skip_existing_strong=skip_existing_strong
                )
            except Exception as ex:
                trace = []
                tb = ex.__traceback__
                while tb is not None:
                    trace.append({
                        "filename": tb.tb_frame.f_code.co_filename,
                        "name": tb.tb_frame.f_code.co_name,
                        "lineno": tb.tb_lineno
                    })
                    tb = tb.tb_next
                print(str({
                    'type': type(ex).__name__,
                    'message': str(ex),
                    'trace': trace
                }))


'''
    A child class of DacEncodecClapDataset
    Reads text metadata feature in json, which is in format
    {
        "the-audio-name-without-slash": {
            'text': 'Human-labelled description.',
            'salmonn_text': 'Description given by salmonn.',
            'tags_text': 'Description summarized from audio tags.',
            'chatgpt_texts': ['A rephase from chatgpt', 'Another rephrase', ...],
            'madmom_key': int taken from 0 to 23 indicating the maj/min main key,
            'madmom_tempo': int indicating the tempo
        } # the components are optional, do not need to provide them all
    }
'''
class DacEncodecClapTextFeatDatasetJamendocaps(DacEncodecClapDatasetJamendocaps):
    def __init__(
        self, 
        *args,
        json_path: Optional[Union[str, os.PathLike]] = None, 
        metadata_dir: Optional[Union[str, os.PathLike]] = None, 
        caption_dir: Optional[Union[str, os.PathLike]] = None,
        valid_ids_json_path: Optional[Union[str, os.PathLike]] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.metadata_dir = metadata_dir
        self.caption_dir = caption_dir
        self.json_path = json_path
        assert json_path is not None
        if metadata_dir is not None or caption_dir is not None:
            init_json = True
            assert metadata_dir is not None and caption_dir is not None
        else:
            init_json = False

        if os.path.exists(json_path):
            with open(json_path) as f:
                self.text_feat_metadata = json.load(f)
        else:
            assert init_json
            self.text_feat_metadata = {}
        if init_json:
            self.captions_txts = os.listdir(self.caption_dir)
            for i_txt in range(len(self.captions_txts)):
                this_txt_path = os.path.join(self.caption_dir, self.captions_txts[i_txt])
                this_audioname = os.path.splitext(self.captions_txts[i_txt])[0]
                with open(this_txt_path, "r") as f:
                    this_caption = f.readline()[4:-4]
                    if "Sorry," in this_caption or "cannot generate" in this_caption:
                        continue
                    else:
                        self.text_feat_metadata[this_audioname] = {
                            "text": this_caption
                        }
            print("Valid text files:", len(self.text_feat_metadata))
    
            self.metadata_jsonls = os.listdir(self.metadata_dir)
            for i_json in range(len(self.metadata_jsonls)):
                this_jsonl_path = os.path.join(self.metadata_dir, self.metadata_jsonls[i_json])
                with open(this_jsonl_path, 'r') as this_jsonl_file:
                    this_json_list = list(this_jsonl_file)
                
                for json_str in this_json_list:
                    this_dict = json.loads(json_str)
                    this_audioname = this_dict["id"]
                    if this_audioname not in self.text_feat_metadata:
                        continue
                    else:
                        tags_text = ""
                        if "vocalinstrumental" in this_dict:
                            if len(this_dict["vocalinstrumental"]) > 0:
                                tags_text += f"{this_dict["vocalinstrumental"]};"
                        if "speed" in this_dict:
                            if len(this_dict["speed"]) > 0:
                                tags_text += f"speed: {this_dict["speed"]};"
                        if "tags" in this_dict:
                            if len(this_dict["tags"]["genres"]) > 0 or len(this_dict["vartags"]["genres"]) > 0:
                                tags_text += f"genres: {",".join(this_dict["tags"]["genres"] + this_dict["tags"]["vartags"])};"
                            if len(this_dict["tags"]["instruments"]) > 0:
                                tags_text += f"instruments: {",".join(this_dict["tags"]["instruments"])};"
                        if len(tags_text) == 0 or ("genres" not in tags_text and "instruments" not in tags_text):
                            tags_text = tags_text + self.text_feat_metadata[this_audioname]["text"]
                        self.text_feat_metadata[this_audioname]["tags_text"] = tags_text

                if i_json % 100 == 0 or i_json >= len(self.metadata_jsonls) - 1:
                    with open(json_path, "w") as jsonFile:
                        json.dump(self.text_feat_metadata, jsonFile)
                        print(f"Dumped text_feat_metadata to {json_path}, num of parsed files: {len(self.text_feat_metadata)}")

        starting_id = 0
        if valid_ids_json_path is not None and os.path.exists(valid_ids_json_path):
            with open(valid_ids_json_path) as f:
                self.audio_ids = json.load(f)
            starting_id = self.audio_ids[-1]
            print(f"Resumed i_audio {starting_id}")
            
        else:
            self.audio_ids = []
        
        self.audio_names_from_json = list(self.text_feat_metadata.keys())
        # the keys of the json should match the audio names saved in the parent class
        for i_audio in range(starting_id, len(self.hf_dataset)):
            audio_filename = self.hf_dataset[i_audio]['audio']['path']
            audio_name = os.path.splitext(audio_filename)[0]
            audio_name = get_true_audio_name(audio_name)
            if audio_name in self.audio_names_from_json:
                self.audio_ids.append(i_audio)
            else:
                print(audio_name, "is not in the json metadata")

            if valid_ids_json_path is not None and (i_audio % 100 == 0 or i_audio >= len(self.hf_dataset) - 1):
                with open(valid_ids_json_path, "w") as jsonFile:
                	json.dump(self.audio_ids, jsonFile)

        if kwargs["percent_start_end"][0] >= 0 and kwargs["percent_start_end"][1] <= 100:
            i_raw_start = int(kwargs["percent_start_end"][0] / 100 * len(self.audio_ids))
            i_raw_end = int(kwargs["percent_start_end"][1] / 100 * len(self.audio_ids))
            print(f"Starting and ending relative id: {i_raw_start}, {i_raw_end}")
            self.audio_ids = self.audio_ids[i_raw_start:i_raw_end]
            
    
    @torch.no_grad()
    def get_text_clap_emb(self, texts: Union[str, List[str]]):
        # texts can be list of strings or a single string
        assert self.clap_model is not None
        inputs_text = self.clap_processor(text=texts, return_tensors="pt").to(self.clap_device)
        text_embed = self.clap_model.get_text_features(**inputs_text) # (n_texts, 512)
        return text_embed.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def get_text_feat_meta_data_dict(self, audio_name: str):
        metadata_dict = self.text_feat_metadata[audio_name]

        print(f"Got metadata of audio {audio_name}")
        data_dict = {}
        
        if 'madmom_key' in metadata_dict:
            print("Found madmom_key feature")
            data_dict['madmom_key'] = np.array(metadata_dict['madmom_key'])
        if 'madmom_tempo' in metadata_dict:
            print("Found madmom_tempo feature")
            data_dict['madmom_tempo'] = np.array(metadata_dict['madmom_tempo'])

        if 'text' in metadata_dict:
            print("Found text description")
            data_dict['text'] = metadata_dict['text'].encode("ISO-8859-1", "ignore")
            print("Got text", metadata_dict['text'])
            # text_clap = self.clap_model.get_text_embedding([metadata_dict['text'], ""])
            text_clap = self.get_text_clap_emb(metadata_dict['text'])
            print("Got CLAP text emb with dummy shape", text_clap.shape)
            data_dict['text_clap'] = float32_to_int16(text_clap[0])

        if 'tags_text' in metadata_dict:
            print("Found tags_text description")
            data_dict['tags_text'] = metadata_dict['tags_text'].encode("ISO-8859-1", "ignore")
            print("Got tags_text", metadata_dict['tags_text'])
            # tags_text_clap = self.clap_model.get_text_embedding([metadata_dict['tags_text'], ""])
            tags_text_clap = self.get_text_clap_emb(metadata_dict['tags_text'])
            print("Got CLAP text emb with dummy shape", tags_text_clap.shape)
            data_dict['tags_text_clap'] = float32_to_int16(tags_text_clap[0])
        
        if 'salmonn_text' in metadata_dict:
            print("Found salmonn_text description")
            data_dict['salmonn_text'] = metadata_dict['salmonn_text'].encode("ISO-8859-1", "ignore")
            print("Got salmonn_text", metadata_dict['salmonn_text'])
            # salmonn_text_clap = self.clap_model.get_text_embedding([metadata_dict['salmonn_text'], ""])
            salmonn_text_clap = self.get_text_clap_emb(metadata_dict['salmonn_text'])
            print("Got CLAP text emb with dummy shape", salmonn_text_clap.shape)
            data_dict['salmonn_text_clap'] = float32_to_int16(salmonn_text_clap[0])

        if 'chatgpt_texts' in metadata_dict:
            print("Found chatgpt_texts description")
            data_dict['chatgpt_texts'] = np.array([caption.encode("ISO-8859-1", "ignore") for caption in metadata_dict['chatgpt_texts']])
            print("Got chatgpt_texts", metadata_dict['chatgpt_texts'])
            # chatgpt_texts_clap = self.clap_model.get_text_embedding(metadata_dict['chatgpt_texts'] + [""])
            chatgpt_texts_clap = self.get_text_clap_emb(metadata_dict['chatgpt_texts'])
            print("Got CLAP text emb with dummy shape", chatgpt_texts_clap.shape)
            data_dict['chatgpt_text_clap'] = float32_to_int16(chatgpt_texts_clap)

        return data_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args in generating h5 dac clap dataset.')
    parser.add_argument(
        '-json-path', type=str,
    )
    parser.add_argument(
        '--caption-dir', type=str, nargs='?',
    )
    parser.add_argument(
        '--metadata-dir', type=str, nargs='?',
    )
    parser.add_argument(
        '--valid-ids-json-path', type=str, nargs='?',
    )
    args = parser.parse_args()

    hf_dataset = load_dataset("amaai-lab/JamendoMaxCaps", data_dir="data")
    json_creator = DacEncodecClapTextFeatDatasetJamendocaps(
        hf_dataset,
        json_path = args.json_path,
        metadata_dir = args.metadata_dir, 
        caption_dir = args.caption_dir,
        valid_ids_json_path = args.valid_ids_json_path
    )