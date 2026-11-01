# 新增实验图：可独立运行的绘图包

此包重画四幅新增数据图，包含运行所需 CSV 和脚本。CSV 来自本次返修已有实验记录；脚本不生成或修改实验结果，不挑选正向结果。原投稿六幅 PDF 没有放入绘图流程，也没有裁切、转换或重画。

## 在 VS Code 中运行

1. 解压后用 VS Code 打开整个 `supplementary/plotting` 文件夹。
2. 安装 Python 3.10 或更新版本，在 VS Code 中选择该 Python 解释器。
3. 打开终端，当前目录为本文件所在的文件夹，然后执行：

```bash
python -m pip install -r requirements.txt
python run_all.py
```

macOS / Linux 如果 Python 命令是 `python3`，将以上命令中的 `python` 改为 `python3`。Windows 也可使用 `py`。已安装 numpy、pandas、matplotlib 时无需重复安装。为隔离环境，可先运行 `python -m venv .venv`，在 VS Code 选择 `.venv` 中的解释器，再安装依赖。

两份脚本也可以分别执行：

```bash
python scripts/generate_publication_plots.py
python scripts/generate_supplement_publication_plots.py
```

VS Code 的“运行和调试”提供“重画全部新增实验图”“仅重画正文两图”“仅重画补充两图”三个选项。

## 输出及对应内容

| 文件名 | 内容 | PDF 固定画布 |
|---|---|---|
| `figures/new/local_effects.pdf` | 3种拓扑×3种观测条件×2种延迟的局部信息增益，共18个均值和原有区间 | 7.2×3.65英寸 |
| `figures/new/reliability_profile.pdf` | 5种方法的CDP相对风险、绝对PAS尾部、正后悔尾部 | 7.2×3.65英寸 |
| `figures/new/guard_cost.pdf` | 18个独立确认增量对比，以及全部6种方法的决策时间与平均PAS | 7.2×3.60英寸 |
| `figures/new/parameter_sensitivity.pdf` | 25组参数的相对风险、P95和P99，各面板完整保留25格 | 7.2×2.95英寸 |

同目录还有300 dpi PNG预览和两份 `plot_audit_*.json`，记录原始数值、CSV哈希与输出哈希。PDF为真正的矢量图，字号按论文双栏宽度设计；将其放到约7.1英寸宽时，轴标签约8.7 pt。建议正文使用PDF，Word中使用PNG。源数据不动的情况下，重画仅改变排版；不同软件版本可能使PDF字节或字体子集哈希不同，这不表示数据不同。

## 方法标签

| 图中名称 | CSV 中的方法 |
|---|---|
| CDP | `CDP` |
| Direct local | `LOCAL_RANK` |
| Static rank | `STATIC_RANK` |
| Static candidate | `CF_STATIC` |
| Guarded local | `CF_LOCAL` |
| Local, no confirmation / No confirmation | `CF_LOCAL_NO_GUARD` |

`Static candidate` 和直接静态排序 `Static rank` 是不同方法。为免误读，正文或返修信图注中应保留这层对应关系。

## 读图及数据口径

- 局部信息图读取 `CF_LOCAL vs CF_STATIC` 的18个现有配对统计量；增益沿用CSV的定义，正值表示CF_LOCAL的PAS更低。数值列仅四舍五入显示，点和区间使用原始精度。
- 确认图读取 `CF_LOCAL vs CF_LOCAL_NO_GUARD` 的全部18个对比。横轴乘以1000显示，因此标有 `×10⁻³`。负均值及跨零区间完整保留。
- 森林图区间使用CSV提供的描述性图层bootstrap区间，没有重新估计或替换区间。
- 风险图(a)使用原始有害情景比例及其描述性区间；它是完整情景表现差于CDP的比例，不能解释为每次切换造成损害的因果概率。
- 风险图(b)保留P95、P99和ES95的原值；图中ES95对应CSV字段 `CVaR95_PAS`。这些是经验尾部统计量，不是三组置信区间。
- 风险图(c)保留CSV字段 `CVaR95_regret`，显示正后悔的经验尾部均值。相对CDP的风险和绝对PAS尾部是不同指标，不能相互替代。
- 决策时间按 `sum(n × mean_s) / sum(n)` 加权后换算为毫秒。平均PAS用全部 `raw_outcomes.csv` 按方法分组计算。时间轴为对数轴，六种方法全部显示。
- 参数图使用全部25个配置，三个面板分别按自己的真实最小值和最大值着色，单元格给出数值；颜色深浅不能跨指标当作同一量纲比较。此图对应阈值回放实验，与观测延迟验证是不同实验条件。

## 更换为后续实验文件

保留 `data/` 内相对目录与CSV列名，将新一批完整记录替换到对应文件后再运行。不要仅修改PNG上的数字。新的CSV须同时更新均值、区间及相关汇总，避免混用不同实验批次。脚本中包含行数、唯一性和区间有效性检查；如果增加拓扑、方法或参数格点，应同步调整行列布局和这些检查。

如果新数据的区间超出当前轴范围，应扩大横轴并保留零线；不能截去不理想的点或区间。原稿和返修信的文字、表格与数值不会由这个绘图包自动更新，需在更换数据后同步核对。

## 可调整样式

`generate_publication_plots.py` 中 `set_style()` 定义字号与通用样式，颜色常量和画布尺寸位于文件开头。补充脚本复用同一风格。输出固定画布，不使用自动紧裁切，以保证正文排版尺寸稳定。所有路径基于脚本所在文件夹解析，可整体移动到服务器或其他电脑。

此包只是重现现有结果的绘图工具，不包含服务器仿真引擎。`source_manifest.json` 保存此次交付的5个输入CSV的SHA-256。图中不利结果、非单调参数变化和较强基线的优势均保留。
