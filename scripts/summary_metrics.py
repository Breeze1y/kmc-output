"""Optional RESET, endpoint resistance and Ag connectivity summaries for a saved run."""
import numpy as np


def _half_peak(v, i, outward):
    keep = np.flatnonzero(outward & (v != 0))
    if not len(keep) or not np.max(np.abs(i[keep])) > 0:
        return None
    peak = keep[np.argmax(np.abs(i[keep]))]
    after = keep[(keep > peak) & (np.abs(i[keep]) <= abs(i[peak])/2)]
    if not len(after):
        return '峰后去程未采到降至半峰的点。'
    return f'峰后去程首次采到电流幅值不高于半峰的偏压为 {v[after[0]]:.6g} V（原始采样值）。'


def _window(v, i, outward, settings):
    low, high, threshold = settings
    x = np.abs(v)
    # Include nominal boundary samples when abs(V) differs by a few float ulps.
    eps = 8*np.finfo(float).eps*max(high, np.finfo(float).tiny)
    inside = (x >= low-eps) & (x <= high+eps) & (i != 0) & outward
    width = np.diff(x)
    idx = np.flatnonzero(inside[:-1] & inside[1:] & (width > eps))
    if not idx.size:
        return '无可比较的相邻采样步。'
    drop = np.log10(abs(i[idx])) - np.log10(abs(i[idx+1]))
    delta = width[idx]*1000
    spacing = (f'{delta[0]:.6g}' if np.allclose(delta,delta[0],rtol=1e-8,atol=0)
               else f'{delta.min():.6g}–{delta.max():.6g}')
    return (f'{idx.size} 个有效步中有 {int((drop>threshold).sum())} 次下降超过 {threshold:g} dec，'
            f'|ΔV| 为 {spacing} mV。')


def _resistance_lines(data, focus):
    resistances = data.resistances()
    result = []
    for phase in ('SET','RESET','ratio'):
        if phase=='ratio':
            series = {n:r['RESET']/r['SET'] for n,r in resistances.items() if 'SET' in r and 'RESET' in r}
            label = '同圈末态读电阻比 R_RESET/R_SET'
        else:
            series = {n:r[phase] for n,r in resistances.items() if phase in r}
            label = f'{phase} 末态读电阻（Ω）'
        if series:
            vals = list(series.values())
            val = f'{series[focus]:.6g}' if focus in series else '—'
            result.append(f'| {label} | {len(vals)} | {min(vals):.6g}–{max(vals):.6g} | {np.median(vals):.6g} | {val} |')
    if not result:
        return []
    return ['', '### 末态读电阻的跨圈范围', '',
            f'| 指标 | 有效圈数 | 范围 | 中位数 | 第 {focus} 圈 |', '|---|---:|---:|---:|---:|', *result]


def _connectivity(rows, path):
    """Prefer time-resolved states, then time-resolved CSV with a consistent endpoint."""
    v = np.array([float(r['voltage_V']) for r in rows])
    values = [str(r.get('metal_connected','')).lower() for r in rows]
    csv_flags = np.isin(values,['true','1']) if all(f in ('false','true','0','1') for f in values) else None
    if path is not None:
        with np.load(path,allow_pickle=False) as state:
            if 'metal_connected' in state and 'voltage_V' in state:
                raw = np.asarray(state['metal_connected']).reshape(-1)
                voltage = np.asarray(state['voltage_V']).reshape(-1)
                points = np.asarray(state['point']).reshape(-1) if 'point' in state else None
                if len(raw)!=len(voltage) or not len(raw) or not np.isin(raw,[0,1]).all():
                    raise ValueError(f'{path}: invalid metal_connected or voltage_V samples')
                flags = raw.astype(bool)
                if points is None:
                    full = len(raw)==len(rows)
                    endpoint = full or len(raw)==1
                else:
                    expected = np.array([int(r['point']) for r in rows])
                    full = np.array_equal(points, expected)
                    endpoint = points.size==raw.size and points[-1]==expected[-1]
                if full and not np.allclose(v,voltage,rtol=1e-8,atol=1e-15):
                    raise ValueError(f'{path}: state voltage differs from the phase I–V')
                if endpoint and not np.isclose(voltage[-1],v[-1],rtol=1e-8,atol=1e-15):
                    raise ValueError(f'{path}: state voltage differs from the phase endpoint')
                if endpoint and csv_flags is not None and flags[-1]!=csv_flags[-1]:
                    raise ValueError(f'{path}: endpoint connectivity differs from the phase CSV')
                if not full:
                    if csv_flags is not None:
                        return (*_transition(rows[0]['phase'],csv_flags,v),'CSV')
                    return (bool(flags[-1]) if endpoint else None),False,None,'NPZ'
                return (*_transition(rows[0]['phase'],flags,voltage),'NPZ')
    if csv_flags is not None:
        return (*_transition(rows[0]['phase'],csv_flags,v),'CSV')
    return None,False,None,None


def _transition(phase, flags, voltage):
    desired = phase=='SET'
    crossing = np.flatnonzero((flags[:-1]!=desired) & (flags[1:]==desired))+1
    first = float(voltage[crossing[0]]) if crossing.size else None
    return bool(flags[-1]),len(flags)>1,first


def _geometry_lines(data, focus):
    result = []
    for phase, change in [('SET','由断变通'),('RESET','由通变断')]:
        endpoints, changes = {}, {}
        full_count = 0
        csv_count = 0
        for (n,p),rows in data.records.items():
            if p!=phase:
                continue
            path = data.states.get(n,{}).get(phase)
            end,full,first,origin = _connectivity(rows,path)
            if end is not None:
                endpoints[n]=end
            full_count += full
            csv_count += full and origin=='CSV'
            if first is not None:
                changes[n]=first
        if endpoints:
            result.append(f'- {phase} 末态有 {len(endpoints)} 圈记录 Ag 通路状态，'
                          f'其中 {sum(endpoints.values())} 圈贯通、{len(endpoints)-sum(endpoints.values())} 圈断开。')
        if full_count:
            text = f'- {full_count} 圈有 {phase} 全阶段通路采样，其中 {len(changes)} 圈观测到{change}'
            if changes:
                values=list(changes.values())
                text += (f'。相邻采样首次{change}时的有符号偏压范围为 {min(values):.6g} 至 {max(values):.6g} V，'
                         f'中位数 {np.median(values):.6g} V')
                if focus in changes:
                    text += f'，第 {focus} 圈为 {changes[focus]:.6g} V'
            if csv_count:
                text += f'。其中 {csv_count} 圈采用同阶段 CSV 的逐点通路标记'
            result.append(text+'。')
        if len(endpoints)>full_count:
            result.append(f'- {phase} 另有 {len(endpoints)-full_count} 圈仅能确认末态，首次通断电压未知。')
    return ['', '### Ag 通路的通断', '', *result] if result else []


def lines(data, focus, reference=None, steps=None):
    """Return Markdown lines. Reference is {label, trace}; steps is (low, high, drop_dec)."""
    result = []
    if steps is not None:
        low,high,threshold = steps
        if not (np.isfinite(steps).all() and 0<=low<high and threshold>0):
            raise ValueError('RESET steps require finite absolute bias bounds 0 <= low < high and a positive drop threshold')
    rows = data.records.get((focus,'RESET'))
    if rows:
        v = np.array([float(r['voltage_V']) for r in rows])
        i = np.array([float(r['current_A']) for r in rows])
        outward = np.array([r['branch']=='outward' for r in rows])
        half = _half_peak(v,i,outward)
        if half:
            result += ['', f'- 第 {focus} 圈 RESET {half}']
        if steps is not None:
            result += ['', f'|V| = {low:g}–{high:g} V 的 RESET 去程直接使用原始相邻采样点。'
                       '两个端点均须落窗、电流非零且 |V| 递增。重复转折电压只算一次采样位置，'
                       '不跨越零电流或回扫，也不平滑。', '',
                       f'- 第 {focus} 圈模拟，'+_window(v,i,outward,steps)]
            if reference and 'RESET' in reference['trace']:
                trace = reference['trace']['RESET']
                result.append(f'- {reference["label"]}，'+_window(np.asarray(trace['v']),np.asarray(trace['i']),
                                                             np.asarray(trace['branch'])=='outward',steps))
    return result + _resistance_lines(data,focus) + _geometry_lines(data,focus)
