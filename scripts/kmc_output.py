"""Export a local KMC trajectory as data, figures and an offline report."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from packed_states import attach_packed
from trajectory import Data, PHASES, load_run, read_groups, write_csv


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True,help='Trajectory directory or combined cycle I–V CSV')
    parser.add_argument('--output',type=Path,help='Report directory (default: kmc-output under the run)')
    parser.add_argument('--label',help='Model or run name for figures')
    parser.add_argument('--kind',choices=['continuous','conditional','unspecified'],default='unspecified')
    parser.add_argument('--focus',type=int,help='Cycle to highlight (default: highest available number)')
    parser.add_argument('--phase',choices=PHASES,help='Phase of unlabeled single-phase inputs')
    parser.add_argument('--measured',type=Path,help='Long-form measured I–V CSV')
    parser.add_argument('--device',help='Select device_label from a multi-device measurement')
    parser.add_argument('--measured-order',choices=['file','ascending','descending'],default='file')
    parser.add_argument('--overlays',type=int,default=100,help='Maximum displayed cycles per overlay panel')
    parser.add_argument('--baseline',type=Path,help='Optional reference simulation directory or combined CSV')
    parser.add_argument('--reference-cycle',type=int,help='Fixed baseline reference (default: same as focus)')
    parser.add_argument('--reset-steps',type=float,nargs=3,metavar=('MIN_ABS_V','MAX_ABS_V','DROP_DEC'),
                        help='Count adjacent outward RESET steps in an absolute-voltage window')
    parser.add_argument('--incident-note',type=Path,help='Existing Markdown diagnosis to link as an explicitly supplied source')
    parser.add_argument('--cell-nm',type=float,nargs='+',metavar='NM',help='Known square pitch, or x and y pitches; otherwise use cells')
    args=parser.parse_args()
    if args.cell_nm is not None and (len(args.cell_nm) not in (1,2) or
              any(not np.isfinite(x) or x<=0 for x in args.cell_nm)):
        parser.error('--cell-nm expects one or two positive finite pitches in nm')
    if args.overlays<1:
        parser.error('--overlays must be positive')
    if args.reference_cycle is not None and not args.baseline:
        parser.error('--reference-cycle requires --baseline')
    if args.device and not args.measured:
        parser.error('--device requires --measured')
    if args.incident_note and not args.incident_note.is_file():
        parser.error('--incident-note must identify an existing Markdown note')
    if args.reset_steps is not None:
        low, high, drop=args.reset_steps
        if not all(np.isfinite(args.reset_steps)) or not (0<=low<high and drop>0):
            parser.error('--reset-steps expects 0 <= MIN_ABS_V < MAX_ABS_V and a positive DROP_DEC')
    return args


def check_state_alignment(data):
    """Check voltage/current axes when the phase archive explicitly provides them."""
    for n,phases in data.states.items():
        for phase,path in phases.items():
            rows=data.records[n,phase]
            with np.load(path,allow_pickle=False) as state:
                if 'point' in state:
                    by_point={int(r['point']):r for r in rows}
                    rows=[by_point[int(k)] for k in state['point']]
                for field in ('voltage_V','current_A'):
                    if field in state:
                        values=np.asarray(state[field]).reshape(-1)
                        matched=rows[-1:] if len(values)==1 and 'point' not in state else rows
                        if len(values)!=len(matched) or not np.allclose(values,[float(r[field]) for r in matched],
                                                            rtol=1e-8,atol=1e-15):
                            raise ValueError(f'{path}: {field} is not aligned with C{n} {phase} I–V samples')


def summary(out,data,label,kind,focus,traces,resistance,measured,device,measured_path,
            order,overlays,morph,baseline_path,reference,steps=None,incident_note=None):
    from summary_metrics import lines as metric_lines
    total=sum(map(len,data.records.values()))
    kinds={'continuous':'连续轨迹','conditional':'分别从各初态计算的条件轨迹','unspecified':'输入轨迹'}
    lines=[f'# {label} 的 KMC 结果','',f'{kinds[kind]}包含 {len(traces)} 个圈号、{len(data.records)} 个扫描阶段和 '
           f'{total:,} 个有符号 I–V 采样点。图中突出第 {focus} 圈。','',
           '[交互式单圈及叠加 I–V](iv_comparison.html) · [独立 I–V CSV](iv_all.csv) · '
           '[静态曲线 PDF](iv_overlay.pdf) · [PNG](iv_overlay.png) · [SVG](iv_overlay.svg)']
    if morph:
        lines+=['',f'已定位的空间状态对应 {sum(len(ph) for ph in data.states.values())} 个扫描阶段。'
                f'[Ag 形貌图册]({morph}) · [形貌预览](morphology_overview.png)']
    else:
        lines+=['','本次输入没有可绘制的 Ag 空间状态。']
    lines+=['','## 逐阶段读出','',
            '| 阶段 | 圈数 | 采样点 | 有末态读电阻的圈数 | 末态读电阻中位数（Ω） |',
            '|---|---:|---:|---:|---:|']
    for phase in PHASES:
        rows=[r for (n,p),r in data.records.items() if p==phase]
        if rows:
            res=[v[phase] for v in resistance.values() if phase in v]
            val=f'{np.median(res):.6g}' if res else '—'
            lines.append(f'| {phase} | {len(rows)} | {sum(map(len,rows)):,} | {len(res)} | {val} |')
    lines+=['',f'## 第 {focus} 圈','']
    for phase,trace in traces[focus].items():
        v=np.asarray(trace['v']);i=np.asarray(trace['i']);b=np.asarray(trace['branch'])
        outward=np.flatnonzero((b=='outward')&(v!=0))
        if outward.size:
            j=outward[np.argmax(abs(i[outward]))]
            lines.append(f'- {phase} 去程的最大电流幅值为 {abs(i[j])*1e3:.6g} mA，对应施加电压 {v[j]:.6g} V。')
        else:
            lines.append(f'- {phase} 没有带非零偏压的去程采样点。')
        if reference and phase in reference['trace']:
            compare=reference['trace'][phase]
            if (np.array_equal(b,compare['branch']) and v.shape==np.shape(compare['v']) and
                    np.allclose(v,compare['v'],rtol=0,atol=1e-10)):
                base=np.abs(np.asarray(compare['i'])); use=(b=='outward')&(v!=0)&(base>0)&(i!=0)
                if use.any():
                    error=np.sqrt(np.mean((np.log10(abs(i[use]))-np.log10(base[use]))**2))
                    lines.append(f'- 与{reference["label"]} 相比，{phase} 在 {use.sum()} 个同偏压、非零电流去程点的 '
                                 f'log₁₀|I| 均方根误差为 {error:.5g} dec。')
    lines.extend(metric_lines(data,focus,reference,steps))
    lines+=['','## 数据来源','',f'- 模拟输入 `{data.combined or data.root}`。I–V 按采样顺序导出，'
            '重复的转折点和零值均保留。末态读电阻取末行正的读出电导倒数。']
    if measured:
        ids=list(measured)[:overlays]
        name=f'（{device}）' if device else ''
        method={'file':'文件出现顺序','ascending':'圈号升序','descending':'圈号降序'}[order]
        lines.append(f'- 实测输入{name} `{measured_path.resolve()}`。可选 {len(measured)} 圈，'
                     f'叠加图按{method}取 {len(ids)} 圈（首圈 {ids[0]}，末圈 {ids[-1]}）。')
    if reference:
        lines.append(f'- {reference["label"]} 来自 `{baseline_path.resolve()}`。')
    if incident_note:
        lines.append(f'- [运行问题与处理记录]({incident_note.resolve().as_uri()})（随本次导出指定的本地文档）。')
    (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run(args):
    import matplotlib
    matplotlib.use('Agg')
    from interactive import write_html
    from morphology import write_morphology
    from static_iv import write_iv_plots

    data=load_run(args.run,args.phase)
    traces=data.traces()
    focus=max(traces) if args.focus is None else args.focus
    if focus not in traces:
        raise ValueError(f'Focus cycle {focus} missing; available: {sorted(traces)}')
    output=(args.output or data.root/'kmc-output').resolve()
    label=args.label or data.root.name
    resistances=data.resistances()
    base=load_run(args.baseline,args.phase) if args.baseline else None
    reference=None
    if base:
        ref_num=focus if args.reference_cycle is None else args.reference_cycle
        base_traces=base.traces()
        if ref_num not in base_traces:
            raise ValueError(f'Baseline has no C{ref_num}; pass an available --reference-cycle')
        reference={'label':f'基线模拟 C{ref_num}','trace':base_traces[ref_num]}
    measured={}
    if args.measured:
        records=read_groups(args.measured,measured=True,device=args.device)
        measured=Data(args.measured.parent,records,{},[args.measured]).traces()
        if args.measured_order!='file':
            measured=dict(sorted(measured.items(),reverse=args.measured_order=='descending'))
    inputs=[*data.sources,*(base.sources if base else []),*([args.measured] if args.measured else [])]
    if output in (data.root,base.root if base else None) or any((output/'iv_all.csv')==p.resolve() for p in inputs):
        raise ValueError('Choose a report directory separate from the input trajectory and source CSV')
    cell=None if not args.cell_nm else (args.cell_nm[0] if len(args.cell_nm)==1 else tuple(args.cell_nm))
    output.mkdir(parents=True,exist_ok=True)
    # Generate into a temporary directory so an error does not replace an older report.
    with TemporaryDirectory(dir=output) as temp:
        work=Path(temp)
        stage=work/'report'
        stage.mkdir()
        check_state_alignment(data)
        attach_packed(data,focus,work/'simulation_states')
        paired=None
        if base and ref_num==focus:
            check_state_alignment(base)
            attach_packed(base,focus,work/'reference_states')
            paired=base.states
        write_csv(stage/'iv_all.csv',data)
        write_iv_plots(stage,traces,resistances,focus,label)
        morph=write_morphology(stage,data.states,focus,cell,label,baseline_states=paired)
        write_html(stage,traces,measured,focus,label,args.overlays,morph,reference)
        summary(stage,data,label,args.kind,focus,traces,resistances,measured,args.device,args.measured,
                args.measured_order,args.overlays,morph,args.baseline,reference,args.reset_steps,args.incident_note)
        previous_summary=output/'summary.md'
        retire_morph=(not morph and previous_summary.is_file() and
                      '[Ag 形貌图册](morphology_atlas.pdf)' in previous_summary.read_text(encoding='utf-8'))
        for artifact in stage.iterdir():
            artifact.replace(output/artifact.name)
        if retire_morph:
            for name in ('morphology_atlas.pdf','morphology_overview.png'):
                (output/name).unlink(missing_ok=True)
    print(f'{len(traces)} cycles, {sum(map(len,data.records.values()))} signed I–V points: {output}')
    print('iv_comparison.html, iv_all.csv, iv_overlay.{pdf,png,svg}, summary.md'+
          (', morphology_atlas.pdf, morphology_overview.png' if morph else ' (I–V only)'))


if __name__=='__main__':
    try:
        run(parse_args())
    except (ValueError, OSError) as e:
        raise SystemExit(f'kmc-output: {e}')
