"""Read just the requested frames from a row-aligned compressed state archive."""
import csv
import json
import zipfile

import numpy as np

from trajectory import PHASES, SWEEPS


def attach_packed(data, focus, temporary):
    archive = data.root/'states.npz'
    if data.states or not data.combined or not archive.is_file():
        return
    # A flat archive has one morphology per CSV row, including any FORMING
    # rows that will not be part of a cycling report.
    ranges = {}
    count = 0
    with data.combined.open(encoding='utf-8-sig',newline='') as h:
        for index,r in enumerate(csv.DictReader(h)):
            count=index+1
            phase=(r.get('phase') or r.get('sweep') or '').strip().upper()
            phase=SWEEPS.get(phase,phase)
            if phase not in PHASES or not r.get('cycle'):
                continue
            key=(int(r['cycle']),phase)
            if key in ranges:
                if ranges[key][1]!=index:
                    raise ValueError(f'Noncontiguous packed morphology block {key}')
                ranges[key][1]=index+1
            else:
                ranges[key]=[index,index+1]
    if not ranges:
        return
    temporary.mkdir(exist_ok=True)
    with np.load(archive,allow_pickle=False) as z:
        units = z['subunits'].item() if 'subunits' in z else None
        if units is None and (data.root/'parameters.json').is_file():
            units=json.loads((data.root/'parameters.json').read_text(encoding='utf-8-sig')).get('subunits')
        key = 'Ag_units' if 'Ag_units' in z and units is not None else 'metal_id'
        if key not in z:
            raise ValueError(f'{archive}: cannot identify Ag occupancy; Ag_units requires its subunits scale')
        vn = z['VN'] if 'VN' in z else None
    with zipfile.ZipFile(archive) as pack, pack.open(key+'.npy') as stream:
        version=np.lib.format.read_magic(stream)
        if version not in ((1,0),(2,0)):
            raise ValueError(f'Unsupported packed state header version {version}')
        header=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
        shape,fortran,dtype=header(stream)
        if len(shape)!=3 or shape[0]!=count or fortran or dtype.kind not in 'bui':
            raise ValueError(f'{archive}: expected C-order integer Ag frames aligned to {count} CSV rows')
        if vn is not None and vn.shape!=shape[1:]:
            raise ValueError(f'{archive}: expected a time-independent VN lattice')
        origin=stream.tell()
        size=int(np.prod(shape[1:]))*dtype.itemsize
        for (n,phase),(start,end) in ranges.items():
            rows=data.records[n,phase]
            if n!=focus:
                start=end-1
                rows=rows[-1:]
            stream.seek(origin+start*size)
            block=np.frombuffer(stream.read((end-start)*size),dtype=dtype).reshape((-1,*shape[1:]))
            if len(block)!=len(rows):
                raise ValueError(f'{archive}: mismatched frame counts at cycle {n} {phase}')
            fields={key:block,'voltage_V':np.array([float(r['voltage_V']) for r in rows]),
                    'point':np.array([int(r['point']) for r in rows])}
            if vn is not None:
                fields['VN']=vn
            if key=='Ag_units':
                fields['subunits']=units
            if all(r.get('metal_connected','').lower() in ('true','false','1','0') for r in rows):
                fields['metal_connected']=np.array([r['metal_connected'].lower() in ('true','1') for r in rows])
            path=temporary/f'{n}_{phase}.npz'
            np.savez(path,**fields)
            data.states.setdefault(n,{})[phase]=path
