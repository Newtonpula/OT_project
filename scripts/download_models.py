"""Download the CMFGEN model spectra used for the method tests of the paper (Sect. 4, Figs. 3-5).

    python scripts/download_models.py

The paper uses the public model spectra of Dessart et al. (2014) for SNe Ia and Dessart et al. (2013)
for SNe II. Both are on Zenodo (CC-BY-4.0); the author's OCA web pages are often unreachable.
  SN Ia: Dessart, Blondin, Hillier & Khokhlov 2014, MNRAS 441, 532   doi:10.5281/zenodo.5524943
  SN II: Dessart, Hillier, Waldman & Livne 2013, MNRAS 433, 1745     doi:10.5281/zenodo.5524916
Writes data/raw/models/<record>/<archive>.tgz (md5-checked) and extracts each archive into
data/raw/models/<record>/<archive>/. Existing archives with a matching md5 are skipped.
"""
import hashlib
import os
import tarfile

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, 'data', 'raw', 'models')
RECORDS = {'5524943': 'dessart14_snia', '5524916': 'dessart13_snii'}


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    for rec, name in RECORDS.items():
        out_dir = os.path.join(RAW, name)
        os.makedirs(out_dir, exist_ok=True)
        r = requests.get(f'https://zenodo.org/api/records/{rec}', timeout=120)
        r.raise_for_status()
        for f in r.json()['files']:
            path = os.path.join(out_dir, f['key'])
            want = f['checksum'].split(':', 1)[1]
            if not (os.path.exists(path) and md5(path) == want):
                with requests.get(f['links']['self'], stream=True, timeout=600) as d:
                    d.raise_for_status()
                    with open(path, 'wb') as fo:
                        for chunk in d.iter_content(1 << 20):
                            fo.write(chunk)
                if md5(path) != want:
                    raise RuntimeError(f'md5 mismatch for {path}')
                print(f'{name}: {f["key"]} ({f["size"] / 1e6:.1f} MB)')
            stem = f['key'].removesuffix('.tgz')
            if os.path.isdir(os.path.join(out_dir, stem)):
                continue
            with tarfile.open(path) as tar:
                # the SN Ia archive has its own top-level folder <stem>/, the SN II ones do not
                nested = all(m.name.split('/')[0] == stem for m in tar.getmembers())
                tar.extractall(out_dir if nested else os.path.join(out_dir, stem), filter='data')


if __name__ == '__main__':
    main()
