from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import json
import pickle
import subprocess


RESULTSPATH = Path('/scratch/abdominal/data/sequences')


def store_optimization(resultspath, sequences, prot):

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')[2:]
    timestamppath = resultspath/timestamp
    timestamppath.mkdir()

    for sequence in tqdm(sequences, desc='Compressing', total=len(sequences)):
        sequence.compress()

    print('Saving...', end='')
    
    with open(timestamppath/'sequences.pkl', 'wb') as handle:
        pickle.dump(sequences, handle)

    with open(timestamppath/'prot.json', 'w') as handle: 
        json.dump(prot, handle, indent='\t')

    print('done.')


def get_githash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


def get_gitbranch() -> str:
    return str(subprocess.check_output(['git', 'branch'])).split("* ")[1].split("\\n")[0]
