---
name: kmc-output
description: 从已完成的 KMC 忆阻器循环模拟导出有符号 I–V 数据、实测与模拟单圈对比、100 圈叠加图、Ag 形貌图册和中文结果总结。用于用户要求查看、更新或打包 KMC 的 SET/RESET 曲线与形貌，以及比较新模拟、实测数据或指定基线圈时。
---

# kmc-output

使用随技能提供的 `scripts/kmc_output.py` 后处理已落盘的轨迹。圈数、电压网格、扫描方向和形貌均取自输入。用当前项目已有的 Python 环境运行，依赖 Python 3.10+、NumPy 和 Matplotlib。

## 运行

1. 从当前任务选取对应的轨迹根目录或合并 I–V CSV。优先使用已经完成的模拟。检查现有运行记录，确定可用圈号、是否连续继承前一圈状态、实测数据所属器件及采集顺序。
2. 调用本技能目录下的脚本。必填参数只有 `--run`。默认写入轨迹目录下的 `kmc-output/`。可用 `--output` 指定本次报告专用目录，更新时复用同一目录。首次生成时无需提前建目录。
3. 查看导出器打印的实际圈数与采样点数，核对 `summary.md` 中的来源、焦点圈和阶段数。向用户给出 HTML、CSV、形貌图册和总结的可点击路径。如仅有 I–V，说明输入中缺少可绘制的空间状态。

最短调用示例（PowerShell，在项目目录运行，替换轨迹路径）：

```powershell
$exporter = Join-Path $HOME '.codex/skills/kmc-output/scripts/kmc_output.py'
& '.\.venv\Scripts\python.exe' -B -X utf8 $exporter --run '.\results\current'
```

有实测与旧机制基线时的示例（在示例项目 `kmc40` 下运行）：

```powershell
& '.\.venv\Scripts\python.exe' -B -X utf8 $exporter `
  --run '.\replay\reset68\continuous' --kind continuous --focus 68 `
  --measured '..\cycle_data\organized\all_iv_points.csv' `
  --device AlScN_40nm --measured-order descending --overlays 100 `
  --baseline '.\replay\cycles100' --reference-cycle 68 --cell-nm 0.5 `
  --reset-steps 0.30 1.20 0.030
```

示例中的器件、圈号、网格尺寸和实测排序仅适用于这组轨迹。新项目沿用其实际设置。使用其他 Python 环境时替换解释器路径。

## 常用选项

| 参数 | 适用场景 |
|---|---|
| `--focus N`、`--label 名称` | 突出指定的模拟圈，默认取现有最大圈号。标签可写机制或批次名。HTML 可切换其余圈。 |
| `--kind continuous` 或 `conditional` | 依据运行记录标注跨圈连续轨迹或按圈独立初态回放，来源未知时省略。 |
| `--measured CSV`、`--device 标签` | 加入实测单圈和叠加图。多器件实测表须选择器件。 |
| `--measured-order file/ascending/descending`、`--overlays N` | 实测按文件出现顺序或圈号升降序选取，模拟按圈号升序。各自最多叠加 N 圈，默认 100。所有可用圈保留在单圈选择器中。 |
| `--baseline 路径`、`--reference-cycle N` | 在单圈图加入固定的旧模拟参考，默认参考圈等于焦点圈。有可比的初始形貌和电压序列时，可并排比较同圈形貌。 |
| `--reset-steps 最小绝对电压 最大绝对电压 下降阈值dec` | 量化原始 RESET 去程的台阶数，记录电压窗口、阈值与可比较的相邻采样步。比较台阶数时检查两组的采样间距。 |
| `--incident-note 路径.md` | 把指定的运行问题记录链接到本次总结。保留原文位置和其中的证据链接。 |
| `--cell-nm X [Y]` | 使用有来源的网格间距（nm）。一个值代表方格，两个值按 x、y 排列；缺省则用格点坐标。 |
| `--phase SET/RESET` | 为缺少阶段字段的单阶段输入指定阶段。 |

## 输入与读图

- 支持 `cycle_NN/SET/`、`cycle_NN/RESET/` 下的 `iv.csv` 与可选 `states.npz`，以及 `iv/iv_all.csv`、`iv_all.csv`、`iv.csv` 合并表。平铺的 `iv.csv + states.npz` 支持与 CSV 逐行对齐的 Ag 状态。选择轨迹根目录可一并收集各圈形貌。
- 模拟表须提供 `voltage_V` 和有符号的 `current_A`，圈号及 SET/RESET 可由列或目录给出。实测表支持 `cycle`、`phase` 或 `sweep`、`point` 或 `point_index` 以及 `current_I1_A` 或 `current_A`。保留去回程与转折点的原始采样顺序。缺少 `branch` 时只对单转折扫描按绝对偏压转折点划分。FORMING、INIT_RESET 等非循环阶段不计入循环结果。
- `iv_all.csv` 是单独的模拟 I–V 表，单位 V、A。交互页支持逐圈切换、选择 SET/RESET 与扫描方向、显示有符号电流或对数幅值，并分别展示实测与模拟的叠加图。报告可离线打开。
- `iv_overlay.png/.pdf/.svg` 给出模拟曲线；有正的末态 `read_G_S` 时列出对应的读电阻。`morphology_atlas.pdf` 汇总有空间数据的阶段及焦点圈快照，`morphology_overview.png` 便于预览。Ag 子单元使用已知标度显示占比，只有金属 ID 或二值掩膜时显示占据状态。
- `summary.md` 汇总圈数、采样点、读电阻、焦点圈去程峰值与输入来源。同网格基线可计算焦点圈去程的电流幅值对数误差。解释机理时区分图中直接可见的变化与需要额外事件日志支持的推断。

## 运行异常记录

用户同时要求记录某圈异常时，读取该次运行的日志、事件表和相应阶段的已保存状态。在报告目录另存 `cycle_NN_<问题>.md`，记录最早失败的圈、阶段、采样点，触发事件与当时形貌，实际实施的调度改动，以及续算后的验证结果。链接原始证据，标明缺失信息。可用 `--incident-note` 将既有记录加入总结。例如 KMC40 原第 16 圈的旧报告在 `replay/cycles100/report/cycle_16_reset_return_route.md`，归属于当时的调度和轨迹。
