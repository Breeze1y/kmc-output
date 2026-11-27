"""Write a self-contained I–V viewer. Only Python's standard library is needed."""
import json
import math
from pathlib import Path
from urllib.parse import quote


def _series(trace):
    """Preserve each contiguous sweep segment and its native sample coordinates."""
    segments = []
    for phase, values in trace.items():
        if phase not in ('SET', 'RESET'):
            raise ValueError(f'Unexpected phase {phase!r}')
        voltage, current, branches = (values[key] for key in ('v', 'i', 'branch'))
        if len(voltage) != len(current) or len(current) != len(branches):
            raise ValueError(f'{phase} voltage, current and branch lengths differ')
        segment = None
        for point, (v, i, branch) in enumerate(zip(voltage, current, branches)):
            v, i = float(v), float(i)
            if not (math.isfinite(v) and math.isfinite(i)):
                raise ValueError(f'{phase} sample {point + 1} has nonfinite voltage or current')
            if branch not in ('outward', 'return'):
                raise ValueError(f'{phase} sample {point + 1} has unknown branch {branch!r}')
            if segment is None or segment['branch'] != branch:
                segment = dict(phase=phase, branch=str(branch), first=point, v=[], i=[])
                segments.append(segment)
            segment['v'].append(v)
            segment['i'].append(i)
    return segments


def write_html(output, traces, measured, focus, label, overlays, morphology_name, reference=None):
    """Create iv_comparison.html using signed amperes, volts and per-point branches.

    Each trace maps SET/RESET to v, i and branch arrays. Measurement dict order
    is acquisition order. The optional reference has a label and a single trace.
    Each sample uses the voltage grid of its own source.
    """
    if not traces:
        raise ValueError('At least one simulated cycle is required')
    cap = int(overlays)
    if cap < 0:
        raise ValueError('overlays must be nonnegative')
    simulated = [dict(cycle=int(n), series=_series(traces[n])) for n in sorted(traces)]
    observed = [dict(cycle=int(n), series=_series(curve)) for n, curve in (measured or {}).items()]
    selected = int(focus) if focus in traces else simulated[0]['cycle']
    ref = None if reference is None else dict(label=str(reference['label']), series=_series(reference['trace']))
    payload = dict(simulated=simulated, measured=observed, reference=ref, focus=selected,
                   label=str(label), overlays=cap,
                   morphology=quote(Path(morphology_name).as_posix(), safe='/') if morphology_name else None)
    template = (Path(__file__).resolve().parents[1] / 'assets/iv_report.html').read_text(encoding='utf-8')
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(',', ':')).replace('<', r'\u003c')
    dest = Path(output)/'iv_comparison.html'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(template.replace('__IV_DATA__', encoded), encoding='utf-8')
    return dest
