"""Render Ag morphology from phase-specific NPZ snapshots without running a solver."""
from collections import Counter
from pathlib import Path
import warnings

import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle
import numpy as np


EMPTY = '#fbfcfc'
VACANCY = '#a5b5c0'
PALE = np.array([1., .87, .66])
METAL = np.array([.55, .245, .09])
TEAL = '#006d73'


def _phase_data(path, cycle, phase, duplicated=False):
    """Map documented quantities only. A metal ID gives presence, never volume."""
    with np.load(path, allow_pickle=False) as z:
        fraction = None
        divisor = None
        if 'Ag_units' in z and 'subunits' in z:
            units = z['Ag_units']
            scale = z['subunits']
            if (scale.size != 1 or not np.isfinite(scale.item()) or
                    scale.item() != int(scale.item()) or scale.item() <= 0):
                raise ValueError(f'{path}: invalid subunits per full cell')
            divisor = int(scale.item())
            if units.dtype.kind not in 'bui' or np.any(units < 0) or np.any(units > divisor):
                raise ValueError(f'{path}: Ag units outside 0..{divisor}')
            fraction = units.astype(np.float32) / divisor
        elif 'Ag_fraction' in z or 'ag_fraction' in z:
            key = 'Ag_fraction' if 'Ag_fraction' in z else 'ag_fraction'
            fraction = z[key].astype(np.float32)
            if not np.all(np.isfinite(fraction) & (fraction >= 0) & (fraction <= 1)):
                raise ValueError(f'{path}: invalid Ag volume fraction')
        quantitative = fraction is not None
        if fraction is None and 'metal_id' in z:
            ids = z['metal_id']
            if ids.dtype.kind in 'bui' and np.all(ids >= 0):
                fraction = ids > 0
        if fraction is None:
            for key in ('Ag_mask', 'ag_mask'):
                if key in z and np.isin(z[key], [0, 1]).all():
                    fraction = z[key].astype(bool)
                    break
        if fraction is None or fraction.ndim not in (2, 3) or not fraction.size:
            return None
        if fraction.ndim == 2:
            fraction = fraction[None]
        n = len(fraction)
        index = np.arange(n)
        # A collection of phases in one file can be sliced only when labeled.
        available = {k for k in ('cycle', 'phase') if k in z and z[k].shape == (n,)}
        if duplicated and len(available) != 2:
            warnings.warn(f'{path}: repeated path without per-frame cycle and phase labels')
            return None
        if 'cycle' in available:
            index = index[z['cycle'][index].astype(str) == str(cycle)]
        if 'phase' in available:
            index = index[np.char.upper(z['phase'][index].astype(str)) == phase]
        if not len(index):
            return None
        data = dict(f=fraction[index], count=divisor, quantitative=quantitative, vn=None,
                    initial=None, v=None, connected=None, time=None, time_unit='')
        if 'VN' in z:
            vn = z['VN']
            if vn.shape == fraction.shape[1:] and np.isin(vn, [0, 1]).all():
                data['vn'] = vn.astype(bool)
            elif vn.shape == fraction.shape and np.isin(vn, [0, 1]).all():
                data['vn'] = vn[index].astype(bool)
        if 'initial_metal_id' in z and len(index) == n and z['initial_metal_id'].shape == fraction.shape[1:]:
            data['initial'] = z['initial_metal_id']
        for key, name in [('voltage_V','v'), ('metal_connected','connected'),
                          ('elapsed_units','time'), ('time_s','time')]:
            if key in z and z[key].shape == (n,):
                values = z[key][index]
                if name == 'connected' and np.isin(values,[0,1]).all():
                    data[name] = values.astype(bool)
                elif name != 'connected' and np.isfinite(values).all():
                    data[name] = values
                    if name == 'time':
                        data['time_unit'] = 's' if key == 'time_s' else 'model units'
        return data


def _vn_at(data, i):
    vn = data['vn']
    return vn[i] if vn is not None and vn.ndim == 3 else vn


def _paired(a, b):
    if (not b or a['f'].shape != b['f'].shape or a['quantitative'] != b['quantitative'] or
            a['v'] is None or b['v'] is None or not np.array_equal(a['v'],b['v'])):
        return False
    has_initial = a['initial'] is not None and b['initial'] is not None
    if len(a['f']) == 1 and not has_initial:
        return False  # One terminal image alone does not establish a common start.
    if not np.array_equal(_vn_at(a,0), _vn_at(b,0)):
        return False
    if has_initial and not np.array_equal(a['initial'], b['initial']):
        return False
    if len(a['f']) == 1:
        return True
    return np.array_equal(a['f'][0], b['f'][0])


def _colors(fraction, vn):
    pixels = np.broadcast_to(mpl.colors.to_rgb(EMPTY), (*fraction.shape,3)).copy()
    if vn is not None:
        pixels[vn] = mpl.colors.to_rgb(VACANCY)
    ag = fraction > 0
    pixels[ag] = PALE + fraction[ag,None] * (METAL-PALE)
    return pixels


def _draw(ax, fraction, vn, cell, roi=None, marker=None):
    if roi is None:
        y0,x0,y1,x1 = 0,0,*fraction.shape
    else:
        y0,x0,y1,x1 = roi
    dx,dy = (1.,1.) if cell is None else cell
    ax.imshow(_colors(fraction,vn)[y0:y1,x0:x1], interpolation='nearest',
              extent=(x0*dx,x1*dx,y1*dy,y0*dy), aspect='equal')
    ax.set_xticks([x0*dx,x1*dx]); ax.set_yticks([y0*dy,y1*dy])
    ax.tick_params(labelsize=6.5,length=2,pad=2)
    for spine in ax.spines.values():
        spine.set_color('#aeb9bf')
    if marker:
        y,x = marker
        ax.add_patch(Rectangle((x*dx,y*dy),dx,dy,fill=False,edgecolor=TEAL,lw=1.5))


def _legend(fig, cell):
    entries=[(EMPTY,'Unoccupied'),(VACANCY,'VN (if recorded)'),(tuple(PALE),'Low Ag fraction'),
             (tuple(METAL),'Full Ag / occupied in a binary mask')]
    fig.legend(handles=[Patch(facecolor=c,edgecolor='#9aa3a9',linewidth=.5,label=s) for c,s in entries],
               loc='lower center',bbox_to_anchor=(.5,.034),ncol=4,frameon=False,fontsize=8)
    unit='lattice cells' if cell is None else f'nm, each cell {cell[0]:g} × {cell[1]:g} nm (x × y)'
    fig.text(.5,.018,f'Horizontal = column, vertical = row, row zero at the top. Axes in {unit}.',
             ha='center',fontsize=8)


def _overview(label, cycle, phase, data, baseline, cell):
    f=data['f']; n=len(f)
    conn=data['connected']
    transitions=(np.flatnonzero(conn[:-1] & ~conn[1:])+1
                 if conn is not None else np.array([],int))
    cut=int(transitions[0]) if len(transitions) else None
    if baseline:
        # Work one frame at a time to avoid a second full float movie.
        scores=[np.abs(x.astype(float)-y).sum() for x,y in zip(f,baseline['f'])]
        sample=int(np.argmax(scores))
        if not scores[sample]:
            sample=cut if cut is not None else n//2
    else:
        counts=[np.count_nonzero((x>0)&(x<1)) for x in f]
        sample=int(np.argmax(counts)) if max(counts) else n//2
    stages=([(cut-1,'Before break'),(cut,'First broken sample')] if cut is not None else [(0,'First sample')])
    if sample not in [i for i,_ in stages]:
        stages.append((sample,'Paired sample' if baseline else 'Selected sample'))
    if n-1 not in [i for i,_ in stages]:
        stages.append((n-1,'Last sample'))
    stages.sort(key=lambda t:t[0])
    frames=[]
    for i,s in stages:
        frames.append((data,i,s))
        if baseline and i==sample:
            frames.append((baseline,i,'Baseline, same bias'))
    if baseline:
        field=np.abs(f[sample].astype(float)-baseline['f'][sample])
    else:
        field=np.where((f[sample]>0) & (f[sample]<1),1.,0.)
    if not field.any() and cut is not None:
        field=np.abs(f[cut].astype(float)-f[cut-1])
    if not field.any():
        field=np.abs(f[-1].astype(float)-f[0])
    zoom=bool(field.any())
    if zoom:
        y,x=map(int,np.unravel_index(np.argmax(field),field.shape))
        h,w=field.shape; roi=(max(y-3,0),max(x-3,0),min(y+4,h),min(x+4,w))
    fig=Figure(figsize=(max(6.6,len(frames)*2.35),6.65 if zoom else 4.55))
    fig.suptitle(f'{label}\nCycle {cycle} · {phase} morphology',fontsize=15,y=.97)
    cols=len(frames)
    grid=fig.add_gridspec(2 if zoom else 1,cols,left=.055,right=.978,top=.79,bottom=.19,
                          hspace=.69,wspace=.35)
    for col,(d,i,title) in enumerate(frames):
        image=d['f'][i];vn=_vn_at(d,i)
        ax=fig.add_subplot(grid[0,col]);_draw(ax,image,vn,cell)
        details=[f'frame {i+1}/{n}']
        if d['v'] is not None:
            details.append(f"{d['v'][i]:.3g} V")
        if d['time'] is not None:
            details.append(f"t={d['time'][i]:.4g} {d['time_unit']}")
        ax.set_title(title+'\n'+', '.join(details),fontsize=8.2,pad=5,wrap=True)
        if zoom:
            y0,x0,y1,x1=roi;dx,dy=(1.,1.) if cell is None else cell
            ax.add_patch(Rectangle((x0*dx,y0*dy),(x1-x0)*dx,(y1-y0)*dy,
                                   fill=False,edgecolor=TEAL,lw=.9))
            detail=fig.add_subplot(grid[1,col]); _draw(detail,image,vn,cell,roi,(y,x))
            value=image[y,x]
            if d['count']:
                amount=f"{round(float(value)*d['count'])}/{d['count']}"
            elif d['quantitative']:
                amount=f'{value:.3g}'
            else:
                amount='present' if value else 'absent'
            detail.set_title(f'Marked cell · Ag {amount}',fontsize=9,color=TEAL)
    msg=('First recorded connected-to-disconnected transition.' if cut is not None else
         'Recorded frames shown; no identified connected-to-disconnected transition.')
    if baseline:
        msg += '\nBaseline matched by initial Ag IDs or first Ag image, available VN and the exact voltage sequence.'
    fig.text(.5,.103,msg,ha='center',fontsize=8.5)
    _legend(fig,cell)
    return fig


def write_morphology(output, states, focus, cell_nm, label, baseline_states=None):
    """Write the two morphology reports for available cycle/phase snapshots."""
    if cell_nm is not None:
        cell_nm=np.asarray(cell_nm,dtype=float)
        if cell_nm.ndim==0:
            cell_nm=np.repeat(cell_nm,2)
        if cell_nm.shape!=(2,) or not np.all(np.isfinite(cell_nm) & (cell_nm>0)):
            raise ValueError('cell_nm must be a positive scalar, (dx, dy), or None')
    items=[(c,phase,Path(phases[phase])) for c,phases in sorted(states.items())
           for phase in ('SET','RESET') if phases.get(phase) and Path(phases[phase]).is_file()]
    copies=Counter(p.resolve() for _,_,p in items)
    ends=[]
    for c,phase,path in items:
        data=_phase_data(path,c,phase,copies[path.resolve()]>1)
        if data is not None:
            vn=_vn_at(data,-1)
            ends.append((c,phase,path,data['f'][-1].copy(),None if vn is None else vn.copy(),data['quantitative']))
    if not ends:
        return None
    chosen=next((e for e in ends if e[0]==focus and e[1]=='RESET'),
                next((e for e in ends if e[0]==focus),ends[0]))
    c,phase,path=chosen[:3]
    data=_phase_data(path,c,phase,copies[path.resolve()]>1)
    other=None
    bp=(baseline_states or {}).get(c,{}).get(phase)
    if bp and Path(bp).is_file():
        baseline_paths=[Path(p).resolve() for ph in baseline_states.values() for p in ph.values() if p]
        candidate=_phase_data(Path(bp),c,phase,baseline_paths.count(Path(bp).resolve())>1)
        if _paired(data,candidate):
            other=candidate
        elif candidate is not None:
            warnings.warn(f'Cycle {c} {phase}: baseline comparison omitted (initial morphology or voltage grid mismatch)')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    name='morphology_atlas.pdf'
    fonts={f.name for f in font_manager.fontManager.ttflist}
    fallback=[f for f in ('Microsoft YaHei','Noto Sans CJK SC','Arial Unicode MS') if f in fonts]
    with mpl.rc_context({'font.family':['DejaVu Sans',*fallback],'pdf.fonttype':42}), PdfPages(output/name) as pdf:
        front=_overview(label,c,phase,data,other,cell_nm)
        pdf.savefig(front,dpi=200);front.savefig(output/'morphology_overview.png',dpi=220)
        for start in range(0,len(ends),12):
            chunk=ends[start:start+12]
            cols=min(4,len(chunk));rows=(len(chunk)+cols-1)//cols
            fig=Figure(figsize=(max(7.,cols*2.5),max(4.,rows*2.23+1.3)))
            fig.suptitle(f'{label}\nFinal saved morphology · cycles {chunk[0][0]}–{chunk[-1][0]}',fontsize=14,y=.975)
            grid=fig.add_gridspec(rows,cols,left=.06,right=.97,top=.84,bottom=.16,hspace=.55,wspace=.36)
            for k,(cy,ph,_,image,vn,quantitative) in enumerate(chunk):
                ax=fig.add_subplot(grid[k//cols,k%cols]);_draw(ax,image,vn,cell_nm)
                ax.set_title(f'Cycle {cy} · {ph}\n'+('Ag fraction' if quantitative else 'Ag presence'),
                             fontsize=8.5,pad=3)
            _legend(fig,cell_nm)
            pdf.savefig(fig,dpi=200)
    return name
