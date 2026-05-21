# AEC v2 — 声学回声消除算法

基于 WebRTC AEC 核心的两级回声消除实现，针对多通道 USB 麦克风设备（EMEET OfficeCore M0 Plus）进行了适配与优化。

---

## 算法原理

### 整体架构

```
远端信号 x(n) ───────────────────────────────────────────┐
                                                          ↓
近端信号 d(n) ──► [ Stage 1: PBFDAF 线性消除 ] ──► e(n) ──► [ Stage 2: NLP 非线性抑制 ] ──► 输出 y(n)
```

### Stage 1 — 线性回声消除（PBFDAF-NLMS）

算法：**Partitioned Block Frequency Domain Adaptive Filter**（分区块频域自适应滤波器）

将自适应滤波器分成 **12 个分区**，每分区 64 个采样，共覆盖 **768 个采样（16ms @ 48kHz）** 的回声尾。

每帧处理流程：

```
1. 远端信号 x(n) 拼接上帧 → 128 点 FFT → X_k（保存到环形缓冲 xfBuf）
2. 回声估计：ŝ = IFFT( Σ X_k · H_k )，k = 0..11（12 分区频域卷积）
3. 误差信号：e(n) = d(n) - ŝ(n)（近端减去估计回声）
4. NLMS 归一化：ef = μ · E / xPow（步长 μ=0.5，功率归一化防抖动）
5. 系数更新：H_k += IFFT( X_k* · ef ) → FFT（overlap-save 约束）
6. 发散检测：若 seSum > 20×sdSum，清零滤波器重启
```

关键参数：

| 参数 | 值 | 说明 |
|---|---|---|
| `PART_LEN` | 64 | 每帧采样数 |
| `kNormalNumPartitions` | 12 | 分区数 |
| 回声尾覆盖 | 768 采样 / 16ms | 12 × 64 / 48000 |
| `filter_step_size` (μ) | 0.5 | NLMS 步长 |
| `error_threshold` | — | 误差限幅，防止系数溢出 |

### Stage 2 — 非线性回声抑制（NLP）

算法：**基于子带相干性的非线性处理**

利用三路信号（近端 `d`、误差 `e`、远端 `x`）在每个子带计算两个相干指标：

```
cohde = |S_de|² / (S_dd · S_ee)   ← 近端与误差的相干度
cohxd = |S_xd|² / (S_xx · S_dd)   ← 远端与近端的相干度
```

**状态机判断说话状态：**

| 场景 | cohde | cohxd | 处理方式 |
|---|---|---|---|
| 只有近端说话 | ≈ 1 | ≈ 0 | `hNl = cohde`，基本不压制 |
| 双讲 | 中等 | 中等 | `hNl = min(cohde, 1-cohxd)` |
| 纯回声 | ≈ 0 | ≈ 1 | `hNl → 0`，大力压制 |

**Overdrive 非线性放大：**

```
hNl[i] = hNl[i] ^ (overdrive_scaling × overDriveCurve[i])
```

高频子带的 `overDriveCurve` 更大，回声压制量向高频倾斜，同时保留低频的自然感。

**舒适噪声注入：** 强压制后补充背景噪声，避免完全静音的人工感。

---

## 项目结构

```
aec/
├── aec_v2.h            # 公共接口、数据结构、算法注释
├── aec_v2.c            # 算法核心实现
├── ooura_fft/          # Ooura FFT 库（纯 C，无依赖）
│   ├── ooura_fft.c
│   └── ooura_fft_mips.c
├── test_main.c         # 合成信号测试（400Hz 正弦，验证消除效果）
├── test_real.c         # 离线 WAV 文件处理器
├── main_realtime.c     # 实时处理（PortAudio，录制 WAV）
├── compile.bat         # MSVC 编译脚本
└── README.md
```

---

## 编译

依赖：
- Visual Studio 2022 Build Tools（MSVC）
- [vcpkg](https://vcpkg.io) 安装 portaudio：`vcpkg install portaudio:x64-windows`

```bat
compile.bat
```

编译产物：

| 文件 | 用途 |
|---|---|
| `aec_test.exe` | 合成信号自测，打印消除前后对比 |
| `aec_wav.exe` | 离线处理多通道 WAV 文件 |
| `aec_rt.exe` | 实时处理 + 录制（PortAudio） |

---

## 使用方法

### 1. 合成信号测试

```bat
aec_test.exe
```

输出近端/远端/AEC 输出的 RMS 对比，PASS 标准：回声消除 > 6 dB。

### 2. 离线 WAV 处理

```bat
aec_wav.exe <input.wav> [near_ch=0] [ref_ch=5] [dsp_ch=6] [output=output.wav]
```

输入：多通道 WAV（支持 16/24/32-bit PCM 或 32-bit float）

输出文件：

| 文件 | 内容 |
|---|---|
| `output.wav` | AEC 处理结果 |
| `near_ch0.wav` | 提取的近端通道 |
| `ref_ch5.wav` | 提取的回采通道 |
| `dsp_ch6.wav` | 硬件 DSP 通道（对比用） |

### 3. 实时处理（PortAudio）

**列出设备：**

```bat
aec_rt.exe
```

**启动实时处理并录制：**

```bat
aec_rt.exe <input_dev> [near_ch=0] [ref_ch=5] [total_ch=7] [out_dev=-1]
```

参数说明：

| 参数 | 说明 |
|---|---|
| `input_dev` | 输入设备索引（从设备列表中选） |
| `near_ch` | 近端麦克风通道号，默认 0 |
| `ref_ch` | 回采/远端参考通道号，默认 5 |
| `total_ch` | 输入设备总通道数，默认 7 |
| `out_dev` | 输出设备索引（-1=系统默认，-2=不播放） |

**运行后按 Enter 停止**，自动保存：

| 文件 | 内容 |
|---|---|
| `rec_near.wav` | 近端麦克风（单声道） |
| `rec_ref.wav` | 远端参考/回采（单声道） |
| `rec_aec.wav` | AEC 处理输出（单声道） |
| `rec_raw.wav` | 全部通道交织原始数据 |

**EMEET OfficeCore M0 Plus 通道布局：**

```
ch 0-3 : 原始麦克风（near-end）
ch 4   : 级联通道
ch 5   : 回采通道（far-end reference）
ch 6   : 硬件 DSP 处理后通道
```

典型命令：

```bat
aec_rt.exe 1              # 设备1，默认通道配置，系统默认输出播放
aec_rt.exe 1 0 5 7 -2     # 设备1，不播放，只录制
```

---

## API 接口

```c
// 创建实例
AECV2 *aec = AECV2_Create();

// 初始化（采样率支持 8000 / 16000 / 48000，nlp_mode: 0=轻度 1=标准 2=强力）
AECV2_Init(aec, 48000, 1);

// 可选：预设回声延迟（分区索引），加速收敛
AECV2_Delay_Set(aec, 0);

// 每次处理 64 个采样
// farEnd:  远端参考信号（回采）float[-1,1]，64 点
// nearEnd: 近端麦克风信号（含回声）float[-1,1]，64 点
// output:  回声消除后输出，64 点
AECV2_Process(aec, farEnd, nearEnd, output);

// 释放
AECV2_Free(aec);
```

---

## 平台加速

代码支持多平台 SIMD 加速，通过编译宏启用：

| 宏 | 指令集 | 平台 |
|---|---|---|
| `ARCH_X86_FAMILY` | SSE2 | x86/x64 |
| `ARCH_ARM_NEON` | NEON | ARM |
| `MIPS_FPU_LE` | MIPS FPU | MIPS |
| 无 | 纯 C fallback | 通用 |

加速覆盖的热点函数：`FilterFar`、`ScaleErrorSignal`、`FilterAdaptation`、`UpdateCoherenceSpectra`、`ComputeCoherence`、`Overdrive`、`Suppress`。

当前 Windows 编译未启用任何宏，使用纯 C fallback。如需 SSE2，在 `compile.bat` 中加 `/DARCH_X86_FAMILY`（需同时提供 SSE2 实现文件）。

---

## 参考

- WebRTC AEC (`modules/audio_processing/aec/aec_core.cc`)
- Ooura FFT: http://www.kurims.kyoto-u.ac.jp/~ooura/fft.html
- S. Haykin, *Adaptive Filter Theory*, 4th ed.
