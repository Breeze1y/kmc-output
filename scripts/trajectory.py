"""Read signed I–V traces and locate matching per-phase KMC state files."""
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PHASES = ('SET', 'RESET')
SWEEPS = {'IV SWEEP +': 'SET', 'IV SWEEP -': 'RESET'}
BRANCH = {'forward':'outward', 'outward':'outward', 'reverse':'return',
          'backward':'return', 'return':'return'}


@dataclass
class Data:
    root: Path
    records: dict
    states: dict
    sources: list
    combined: Path | None = None

    def traces(self):
        result = {}
        for (n,phase),rows in self.records.items():
            result.setdefault(n,{})[phase] = {
                'v':[float(r['voltage_V']) for r in rows],
                'i':[float(r['current_A']) for r in rows],
                'branch':[r['branch'] for r in rows]}
        return result

    def resistances(self):
        result = {}
        for (n,phase),rows in self.records.items():
            value = rows[-1].get('read_G_S')
            if value:
                g = float(value)
                if math.isfinite(g) and g>0:
                    result.setdefault(n,{})[phase] = 1/g
        return result


def read_groups(path, cycle=None, phase=None, *, measured=False, device=None, enforce_phase=False):
    groups = {}
    labels = set()
    with path.open(encoding='utf-8-sig',newline='') as h:
        source = csv.DictReader(h)
        fields = source.fieldnames or ()
        current = 'current_I1_A' if measured and 'current_I1_A' in fields else 'current_A'
        if 'voltage_V' not in fields or current not in fields:
            raise ValueError(f'{path} requires voltage_V and {current} in volts and signed amperes')
        for lineno,r in enumerate(source,2):
            d = (r.get('device_label') or '').strip()
            if device is not None and d != device:
                continue
            labels.add(d)
            tag = (r.get('phase') or r.get('sweep') or phase or '').strip().upper()
            if not tag:
                raise ValueError(f'{path}: provide --phase for unlabeled single-phase input')
            tag = SWEEPS.get(tag,tag)
            if tag not in PHASES:
                continue
            if enforce_phase and phase and phase != tag:
                raise ValueError(f'{path}:{lineno} phase {tag} conflicts with {phase}')
            raw_id = r.get('cycle') or cycle
            if raw_id is None:
                raise ValueError(f'{path}:{lineno} requires a cycle column or cycle_NN directory')
            n = int(raw_id)
            if cycle is not None and n!=cycle:
                raise ValueError(f'{path}:{lineno} cycle {n} conflicts with directory {cycle}')
            rows = groups.setdefault((n,tag),[])
            point = int(r.get('point') or r.get('point_index') or len(rows))
            if rows and point != int(rows[-1]['point'])+1:
                raise ValueError(f'{path}:{lineno} duplicate or out-of-order point in C{n} {tag}')
            i = r[current]
            if not all(math.isfinite(float(x)) for x in (r['voltage_V'],i)):
                raise ValueError(f'{path}:{lineno} non-finite voltage or current')
            r.update(cycle=str(n),phase=tag,point=str(point),current_A=i)
            branch = (r.get('branch') or '').strip().lower()
            if branch and branch not in BRANCH:
                raise ValueError(f'{path}:{lineno} unsupported branch {branch!r}')
            r['branch'] = BRANCH.get(branch,'')
            rows.append(r)
    if measured and len(labels)>1:
        raise ValueError('Select one measured device with --device: '+', '.join(sorted(labels)))
    if not groups:
        raise ValueError(f'{path}: no matching SET/RESET records (device={device!r})')
    for key,rows in groups.items():
        if any(not r['branch'] for r in rows):
            if any(r['branch'] for r in rows):
                raise ValueError(f'{path}: partially missing branch labels in {key}')
            v = np.array([abs(float(r['voltage_V'])) for r in rows])
            pivot = int(np.argmax(v))
            if np.any(np.diff(v[:pivot+1]) < -1e-10) or np.any(np.diff(v[pivot:]) > 1e-10):
                raise ValueError(f'{path}: complex sweep {key} requires explicit branch labels')
            for j,r in enumerate(rows):
                r['branch'] = 'outward' if j<=pivot else 'return'
        phases = [r['branch'] for r in rows]
        if 'return' in phases and 'outward' in phases[phases.index('return'):]:
            raise ValueError(f'{path}: multiple sweep reversals in {key}')
    return groups


def load_run(location, single_phase=None):
    location = Path(location).resolve()
    if not location.exists():
        raise ValueError(f'Run not found: {location}')
    root = location if location.is_dir() else location.parent
    local_phase = root.name.upper() if root.name.upper() in PHASES else single_phase
    local_cycle = re.fullmatch(r'cycle_(\d+)',root.parent.name if root.name.upper() in PHASES else root.name)
    # A combined CSV can also be specified directly, including run/iv/iv_all.csv.
    if location.is_file() and root.name=='iv' and any(root.parent.glob('cycle_*')):
        root = root.parent
    merged = location if location.is_file() else next((p for p in
        (root/'iv/iv_all.csv',root/'iv_all.csv',root/'iv.csv') if p.is_file()),None)
    records = read_groups(merged,int(local_cycle.group(1)) if local_cycle else None,local_phase,
                          enforce_phase=root.name.upper() in PHASES) if merged else {}
    sources = [merged] if merged else []
    states = {}
    seen = set()
    if local_cycle and merged and (root/'states.npz').is_file() and len(records)==1:
        n,tag=next(iter(records))
        states[n]={tag:root/'states.npz'}
    folders = sorted((int(m.group(1)),p) for p in root.glob('cycle_*')
                     if p.is_dir() and (m:=re.fullmatch(r'cycle_(\d+)',p.name)))
    if len({n for n,p in folders})!=len(folders):
        raise ValueError(f'{root}: duplicate numeric cycle directory')
    if not folders and local_cycle and not merged:
        folders=[(int(local_cycle.group(1)),root)]
    for n,p in folders:
        entries = [(phase,p/phase) for phase in PHASES if (p/phase).is_dir()]
        if not entries and (p/'iv.csv').is_file():
            entries = [(single_phase,p)]
        for phase,directory in entries:
            path = directory/'iv.csv'
            if path.is_file():
                block = read_groups(path,n,phase,enforce_phase=directory.name.upper() in PHASES)
                for key,rows in block.items():
                    if key in seen:
                        raise ValueError(f'Duplicate phase data for {key}: {path}')
                    seen.add(key)
                    if merged:
                        if key not in records:
                            raise ValueError(f'{path} has a phase missing in {merged}: {key}')
                        previous=records[key]
                        if (len(rows)!=len(previous) or any(
                            a['point']!=b['point'] or a['branch']!=b['branch'] or
                            any(float(a[k])!=float(b[k]) for k in ('voltage_V','current_A','endpoint_current_A','read_G_S')
                                if k in a and k in b and a[k] and b[k])
                            for a,b in zip(rows,previous))):
                            raise ValueError(f'Combined CSV disagrees with {path} for {key}')
                        if not previous[-1].get('read_G_S') and rows[-1].get('read_G_S'):
                            for a,b in zip(previous,rows):
                                a['read_G_S']=b.get('read_G_S','')
                    else:
                        records[key]=rows
                if not merged:
                    sources.append(path)
            state=directory/'states.npz'
            # A single states file is localizable when one phase is identified.
            known = [key for key in records if key[0]==n and (not phase or phase==key[1])]
            if state.is_file() and len(known)==1:
                states.setdefault(n,{})[known[0][1]]=state
    if not records:
        raise ValueError(f'No cycle I–V data in {root}. Supply the trajectory root or its combined CSV.')
    return Data(root,records,states,sources,merged)


def write_csv(path,data):
    fields = ['cycle','phase','point','branch','voltage_V','current_A']
    extras = dict.fromkeys(k for rows in data.records.values() for r in rows[:1] for k in r if k not in fields)
    with path.open('w',newline='',encoding='utf-8-sig') as h:
        writer=csv.DictWriter(h,fieldnames=[*fields,*extras])
        writer.writeheader()
        for key,rows in data.records.items():
            writer.writerows(rows)
