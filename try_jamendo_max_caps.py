import json
from datasets import load_dataset

ds = load_dataset("amaai-lab/JamendoMaxCaps")
print(ds['train'][0])
print(ds.keys())


example_metadata_file = "/pfss/mlde/workspaces/mlde_wsp_ETH_Text_Music/datasets/jamendo_max_caps/2008-01-01.jsonl"
with open(example_metadata_file) as f:
    metadata = [json.loads(line) for line in f]

metadata_id_list = []
for i in range(len(metadata)):
    metadata_id_list.append(metadata[i]["id"])
print(metadata_id_list[:10])
print("num audio samples:", len(ds['train']))
print("num metadata:", len(metadata_id_list))

id_list = []
for i in range(len(ds['train'])):
    data = ds['train'][i]
    this_id = data['audio']['path'].split(".")[0]
    id_list.append(this_id)
    print(data)
    if this_id in metadata_id_list:
        print(i, this_id)

