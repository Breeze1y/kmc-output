"""Render KMC sweeps and optional endpoint resistances from in-memory traces."""
from pathlib import Path
import textwrap
import unicodedata

import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np


PHASE_COLORS = {"SET": "#126775", "RESET": "#ab4c27"}
BRANCH_STYLES = {"outward": "-", "return": "--"}


def _title_font(label):
    chars = {ord(c) for c in label if not c.isspace()}
    preferred = ("Microsoft YaHei", "Noto Sans CJK SC", "Noto Sans SC",
                 "PingFang SC", "WenQuanYi Zen Hei", "SimHei", "DejaVu Sans")
    installed = font_manager.fontManager.ttflist
    first = [f for name in preferred for f in installed if f.name == name]
    for font in first + installed:
        try:
            face = font_manager.get_font(font.fname)
            if all(face.get_char_index(c) for c in chars):
                return font_manager.FontProperties(fname=font.fname)
        except (OSError, RuntimeError):
            continue
    raise ValueError("No local font covers the model label. Install an appropriate CJK/Unicode font and retry.")


def _current_unit(peak):
    for scale, unit in ((1., "A"), (1e-3, "mA"), (1e-6, "µA"),
                        (1e-9, "nA"), (1e-12, "pA"), (1e-15, "fA")):
        if peak >= scale:
            return scale, unit
    return 1., "A"


def write_iv_plots(output, traces, resistances, focus, label):
    """Write iv_overlay.png, .pdf and .svg. Return their paths by extension.

    Each trace contains equally long v, i (signed amps) and branch arrays.
    Missing phases and missing endpoint resistances are permitted. Log axes
    display nonzero current magnitudes; a wholly zero dataset uses a linear
    axis. Branch points remain in input order, including repeated voltages.
    """
    output = Path(output)
    cycles = sorted(traces)
    if focus not in cycles or not traces[focus]:
        raise ValueError("The focus cycle must have a recorded sweep")
    data = {}
    ohms = {}
    peak = 0.
    used_branches = set()
    for cycle in cycles:
        for phase, trace in traces[cycle].items():
            if phase not in PHASE_COLORS:
                raise ValueError(f"Unknown phase {phase!r}")
            v = np.asarray(trace["v"], dtype=float)
            i = np.asarray(trace["i"], dtype=float)
            branch = np.asarray(trace["branch"])
            if (v.ndim != 1 or not v.size or i.shape != v.shape or branch.shape != v.shape
                    or not np.isfinite(v).all() or not np.isfinite(i).all()
                    or not np.isin(branch, tuple(BRANCH_STYLES)).all()):
                raise ValueError(f"Invalid voltage, current or branch in cycle {cycle} {phase}")
            data[cycle,phase] = v, np.abs(i), branch
            peak = max(peak, float(np.abs(i).max()))
            used_branches.update(branch.tolist())
            value = (resistances or {}).get(cycle, {}).get(phase)
            if value is not None and np.isfinite(value) and value > 0:
                ohms[cycle,phase] = float(value)
    factor, unit = _current_unit(peak)
    model_label = str(label).strip() or "KMC I–V"
    width = 8.45 if ohms else 6.8
    # Roughly two Latin half-widths fit in one full-width ideograph.
    double = sum(unicodedata.east_asian_width(c) in "WF" for c in model_label)
    capacity = int(width*8/(1 + double/max(len(model_label),1)))
    heading = "\n".join(textwrap.fill(line, width=capacity) for line in model_label.splitlines())
    font = _title_font(model_label)
    extra_lines = heading.count("\n")
    style = {"font.family":"sans-serif", "font.sans-serif":["DejaVu Sans", "Arial"],
             "font.size":8.5, "axes.titlesize":10, "pdf.fonttype":42,
             "svg.fonttype":"path" if double else "none",
             "axes.spines.top":False, "axes.spines.right":False, "axes.linewidth":.65,
             "axes.labelcolor":"#25323b", "text.color":"#25323b", "legend.frameon":False}
    with mpl.rc_context(style):
        fig = Figure(figsize=(width, 4.12+.23*extra_lines), layout="constrained")
        FigureCanvasAgg(fig)
        gs = fig.add_gridspec(3,2 if ohms else 1, height_ratios=(.56+.23*extra_lines, 2.65, .39),
                              width_ratios=(1.42,1) if ohms else (1,))
        title_ax = fig.add_subplot(gs[0,:])
        title_ax.set_axis_off()
        title_ax.text(0,.98,heading, transform=title_ax.transAxes, fontproperties=font,
                      fontsize=14.5, ha="left",va="top",parse_math=False)
        count_label = f"{len(cycles)} cycle{'s' if len(cycles)!=1 else ''}"
        title_ax.text(0,0,f"{count_label}   ·   Focus cycle {focus}",
                      transform=title_ax.transAxes,fontsize=9,va="bottom")
        iv = fig.add_subplot(gs[1,0])
        rax = fig.add_subplot(gs[1,1]) if ohms else None
        for ax in (iv,rax) if rax is not None else (iv,):
            ax.grid(axis="y",which="major",lw=.55,color="#dee3e6")
            ax.set_axisbelow(True)
            ax.tick_params(which="major",width=.65,length=3)
        if peak > 0:
            iv.set_yscale("log")
        for cycle in [n for n in cycles if n!=focus]+[focus]:
            highlight = cycle==focus
            for phase, color in PHASE_COLORS.items():
                if (cycle,phase) not in data:
                    continue
                v, magnitude, branch = data[cycle,phase]
                for name, linestyle in BRANCH_STYLES.items():
                    keep = branch==name
                    if peak > 0:
                        keep &= magnitude>0
                    if not keep.any():
                        continue
                    y = np.where(keep,magnitude/factor,np.nan)
                    line, = iv.plot(v,y,linestyle,color=color if highlight else "#73808b",
                                    lw=1.6 if highlight else .6,alpha=1. if highlight else .25,
                                    marker="." if keep.sum()==1 else None,
                                    zorder=4 if highlight else 2)
                    line.set_gid(f"cycle-{cycle}-{phase}-{name}")
        phases = [phase for phase in PHASE_COLORS if any(p==phase for _,p in data)]
        iv.set(xlabel="Voltage (V)",ylabel=f"|Current| ({unit})",title=" and ".join(phases)+" sweeps")
        lo = min(float(v.min()) for v,_,_ in data.values())
        hi = max(float(v.max()) for v,_,_ in data.values())
        if hi > lo:
            pad = .025*(hi-lo)
            iv.set_xlim(lo-pad,hi+pad)
        if rax is not None:
            rax.set_yscale("log")
            for phase,color in PHASE_COLORS.items():
                if not any(p==phase for _,p in ohms):
                    continue
                vals = np.array([ohms.get((n,phase), np.nan) for n in cycles])
                line, = rax.plot(cycles,vals,color=color,lw=1.2,label=phase,
                                 marker="." if np.isfinite(vals).sum()<3 else None)
                line.set_gid(f"endpoint-resistance-{phase}")
                if (focus,phase) in ohms:
                    rax.plot(focus,ohms[focus,phase],"o",ms=5.3,markerfacecolor="white",
                             markeredgecolor=color,markeredgewidth=1.4)
            if len(cycles)==1:
                rax.set_xticks(cycles)
                rax.set_xlim(focus-.5,focus+.5)
            else:
                rax.xaxis.set_major_locator(MaxNLocator(nbins=6,integer=True))
            rax.set(xlabel="Cycle",ylabel="Read resistance (Ω)",title="Phase endpoints")
            if any(phase not in traces[focus] for _,phase in ohms):
                rax.legend(loc="best",fontsize=7.5)
        handles = []
        if len(cycles)>1:
            others = len(cycles)-1
            handles.append(Line2D([],[],color="#73808b",label=f"Other {others} cycle{'s' if others!=1 else ''}"))
        for phase,color in PHASE_COLORS.items():
            if phase in traces[focus]:
                handles.append(Line2D([],[],color=color,lw=1.8,label=f"Cycle {focus} {phase}"))
        for name,linestyle in BRANCH_STYLES.items():
            if name in used_branches:
                handles.append(Line2D([],[],color="#35444c",ls=linestyle,lw=1.1,label=name.capitalize()))
        legend = fig.add_subplot(gs[2,:])
        legend.set_axis_off()
        legend.legend(handles=handles, loc="center", ncols=len(handles),
                      fontsize=8., handlelength=1.9, columnspacing=1.15)
        output.mkdir(parents=True,exist_ok=True)
        paths = {ext:output/f"iv_overlay.{ext}" for ext in ("png","pdf","svg")}
        fig.savefig(paths["png"],dpi=300,metadata={"Title":model_label})
        for ext in ("pdf","svg"):
            fig.savefig(paths[ext],metadata={"Title":model_label})
    return paths
