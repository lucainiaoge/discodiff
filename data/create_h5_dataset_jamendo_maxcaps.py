import argparse
import os
import torch
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir("..")
from datasets import load_dataset
try:
    from data.h5_dataset_jamendo_maxcaps import DacEncodecClapTextFeatDatasetJamendocaps
    from data.create_h5_dataset import get_dac_encodec_clap
except:
    from h5_dataset_jamendo_maxcaps import DacEncodecClapTextFeatDatasetJamendocaps
    from create_h5_dataset import get_dac_encodec_clap

def create_h5_dataset_jamendo_maxcaps(
    hf_dataset, json_path, valid_ids_json_path, target_dir, dac_model, encodec_model, clap_model,
    percent_start_end = (None, None), skip_existing = False, skip_existing_strong = False,
    chunk_dur_sec = 27, min_sec = 28,
    no_audio_chunk = False
):
    subdataset = DacEncodecClapTextFeatDatasetJamendocaps(
        hf_dataset = hf_dataset,
        dac_model = dac_model,
        encodec_model = encodec_model,
        clap_model = clap_model,
        json_path = json_path,
        valid_ids_json_path = valid_ids_json_path,
        exts = ['mp3', 'wav'],
        start_silence_sec = 0,
        chunk_dur_sec = chunk_dur_sec,
        min_sec = min_sec,
        percent_start_end = percent_start_end,
        no_audio_chunk = no_audio_chunk
    )
    print("Dataset created")
    subdataset.save_audio_text_to_h5_multiple(
        target_dir, 
        skip_existing = skip_existing,
        skip_existing_strong = skip_existing_strong
    )

def main(args):
    target_dir = args.target_dir
    json_path = args.json_path
    valid_ids_json_path = args.valid_ids_json_path

    dac_model, encodec_model, clap_model = get_dac_encodec_clap(
        use_dac = not args.no_dac,
        use_encodec = not args.no_encodec,
        use_clap = not args.no_clap,
        device = args.device
    )
    hf_dataset = load_dataset("amaai-lab/JamendoMaxCaps")
    percent_start_end = (args.percent_start, args.percent_end)
    
    create_h5_dataset_jamendo_maxcaps(
        hf_dataset, json_path, valid_ids_json_path, target_dir, dac_model, encodec_model, clap_model,
        percent_start_end = percent_start_end, skip_existing = args.skip_existing,
        chunk_dur_sec = args.chunk_dur_sec, min_sec = args.min_sec,
        no_audio_chunk=args.no_audio_chunk
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args in generating h5 dac clap dataset.')
    parser.add_argument(
        '--target-dir', type=str, default='',
        help='the directory that h5 data is saved'
    )
    parser.add_argument(
        '--json-path', type=str, nargs='?',
        help='the path that text feat metadata is saved'
    )
    parser.add_argument(
        '--valid-ids-json-path', type=str, nargs='?',
        help='the path storing the valid ids to visit in the dataset'
    )
    parser.add_argument(
        '--percent-start', type=float, nargs='?',
    )
    parser.add_argument(
        '--percent-end', type=float, nargs='?',
    )
    parser.add_argument(
        '--no-dac', type=bool, default=False,
    )
    parser.add_argument(
        '--no-encodec', type=bool, default=False,
    )
    parser.add_argument(
        '--no-clap', type=bool, default=False,
    )
    parser.add_argument(
        '--skip-existing', type=bool, default=False,
    )
    parser.add_argument(
        '--chunk-dur-sec', type=float, default=27.0,
    )
    parser.add_argument(
        '--min-sec', type=float, default=27.1,
    )
    parser.add_argument(
        '--no-audio-chunk', type=bool, default=False,
    )
    parser.add_argument(
        '--device', type=str, default=None,
    )
    args = parser.parse_args()
    main(args)
