"""Download the raw data for the paper's sample.

    python scripts/download_data.py

1. Open Supernova Catalog JSON files (spectra + photometry). The OSC website
   (sne.space) is offline; its data live in the astrocatalogs/sne-* GitHub
   repositories (one JSON, or JSON.gz, per supernova). Reads
   data/meta/paper_table_C1.csv (Table C.1 of Ramirez et al. 2024) and writes
   data/raw/osc/<name>.json[.gz].
2. The CSP SN II spectra of Gutierrez et al. (2017), which the OSC lacks for
   several SNe II of the sample, into data/raw/csp/Gutierrez_et_al_2017/.
Existing files are skipped.
"""
import csv
import io
import json
import os
import tarfile

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META = os.path.join(ROOT, 'data', 'meta')
RAW = os.path.join(ROOT, 'data', 'raw', 'osc')
CSP_DIR = os.path.join(ROOT, 'data', 'raw', 'csp')
CSP_URL = 'https://csp.obs.carnegiescience.edu/data/Gutierrez_et_al_2017.tar.gz'
REPOS = {'sne-pre-1990': 'master', 'sne-1990-1999': 'master', 'sne-2000-2004': 'master',
         'sne-2005-2009': 'master', 'sne-2010-2014': 'master', 'sne-2015-2019': 'master',
         'sne-2020-2024': 'main'}


def build_index():
    """Map lower-case SN name -> list of (repo, branch, path) from the repo trees."""
    idx = {}
    for repo, branch in REPOS.items():
        r = requests.get(f'https://api.github.com/repos/astrocatalogs/{repo}/git/trees/{branch}', timeout=120)
        r.raise_for_status()
        tree = r.json()
        json.dump(tree, open(os.path.join(META, f'tree_{repo}.json'), 'w'))
        for e in tree['tree']:
            p = e['path']
            if p.endswith('.json') or p.endswith('.json.gz'):
                idx.setdefault(p.rsplit('.json', 1)[0].lower(), []).append((repo, branch, p))
    return idx


def main():
    os.makedirs(RAW, exist_ok=True)
    idx = build_index()
    rows = list(csv.DictReader(open(os.path.join(META, 'paper_table_C1.csv'))))
    for row in rows:
        hits = idx.get(row['name'].lower(), [])
        if not hits:
            print('not in OSC:', row['name'])
            continue
        for k, (repo, branch, path) in enumerate(hits):
            # macOS file systems are case-insensitive: ASASSN-14lp.json and asassn-14lp.json
            # collide, so extra case-variants get a suffix (the pipeline reads the canonical one).
            ext = '.json.gz' if path.endswith('.gz') else '.json'
            out = os.path.join(RAW, row['name'] + ext if k == 0 else f"{row['name']}__dup{k}{ext}")
            if os.path.exists(out):
                continue
            url = f'https://raw.githubusercontent.com/astrocatalogs/{repo}/{branch}/{path}'
            r = requests.get(url, timeout=600)
            r.raise_for_status()
            with open(out, 'wb') as f:
                f.write(r.content)
            print(f'{row["name"]:12s} <- {repo}/{path} ({len(r.content) / 1e6:.1f} MB)')


def download_csp():
    if os.path.isdir(os.path.join(CSP_DIR, 'Gutierrez_et_al_2017')):
        return
    os.makedirs(CSP_DIR, exist_ok=True)
    r = requests.get(CSP_URL, timeout=600)
    r.raise_for_status()
    with open(os.path.join(CSP_DIR, os.path.basename(CSP_URL)), 'wb') as f:
        f.write(r.content)
    with tarfile.open(fileobj=io.BytesIO(r.content)) as tar:
        tar.extractall(CSP_DIR, filter='data')      # rejects absolute paths, '..' and links
    print(f'CSP SN II spectra <- {CSP_URL} ({len(r.content) / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
    download_csp()
