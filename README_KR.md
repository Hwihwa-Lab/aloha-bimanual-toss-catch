---
language:
- ko
- en
license: mit
library_name: lerobot
pipeline_tag: robotics
short_description: "14-DOF Bimanual Dynamic Toss & Catch Gym & LeRobot"
tags:
- robotics
- lerobot
- aloha
- bimanual
- bimanual-manipulation
- dual-arm
- 14-dof
- mujoco
- mujoco-3
- physical-ai
- embodied-ai
- reinforcement-learning
- ppo
- stable-baselines3
- gymnasium
- dynamic-manipulation
- toss-and-catch
- ballistic-interception
- compliant-control
- active-damping
- telemetry-hud
- 60fps-telemetry
- sim-to-real
- domain-randomization
- pytorch
- python
- korean
- 한국어
- model-index
- region:us
model-index:
- name: aloha-bimanual-toss-catch-model
  results:
  - task:
      type: robotics
      name: Dynamic Bimanual Ballistic Toss and Catch
    dataset:
      name: aloha-bimanual-toss-catch-telemetry
      type: simulation
    metrics:
    - name: Catch Intercept Rate
      type: success_rate
      value: 100.0
    - name: Time to Intercept
      type: time_to_success
      value: 0.59
    - name: Control Frequency
      type: frequency
      value: 50.0
    - name: Headless Throughput (Colab)
      type: throughput_fps
      value: 12500.0
    - name: Torque Smoothness (Jerk)
      type: jerk_metric
      value: 0.946
---

# 🎯 ALOHA 14자유도 양손 동역학 투척 & 캐치 // Gymnasium 환경 및 PPO 벤치마크

[![Language: English](https://img.shields.io/badge/Language-English-blue)](README.md)
[![Language: 한국어](https://img.shields.io/badge/Language-한국어-green)](README_KR.md)
[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-라이브%20데모-purple)](https://huggingface.co/spaces/hwihwalab/aloha-bimanual-toss-catch)
[![Hugging Face Model Hub](https://img.shields.io/badge/🤗%20Hugging%20Face-모델%20허브-orange)](https://huggingface.co/hwihwalab/aloha-bimanual-toss-catch-model)
[![Hugging Face Telemetry Dataset](https://img.shields.io/badge/📦%20데이터셋-60fps%20텔레메트리-green)](https://huggingface.co/datasets/hwihwalab/aloha-bimanual-toss-catch-telemetry)
[![Physics: MuJoCo 3.x](https://img.shields.io/badge/물리엔진-MuJoCo%203.x-0080FF)](https://mujoco.org)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-AlohaTossCatch--v0-00D26A)](https://gymnasium.farama.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-PPO-E65C00)](https://stable-baselines3.readthedocs.io)
[![LeRobot](https://img.shields.io/badge/LeRobot-호환%20생태계-FFD21E)](https://github.com/huggingface/lerobot)
[![Benchmark](https://img.shields.io/badge/벤치마크-100%25%20요격%20%7C%200.59초%20도달-brightgreen)](#-정량적-벤치마크-성능-비교-매트릭스)
[![GitHub](https://img.shields.io/badge/GitHub-aloha--bimanual--toss--catch-181717?logo=github)](https://github.com/Hwihwa-Lab/aloha-bimanual-toss-catch)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **"자율 듀얼암 로봇은 초고속 공중 비행체를 능동 감쇠로 완충 캐칭할 수 있는가?"**  
> 심층 강화학습(PPO)과 능동 감쇠 완충 제어를 결합하여 고속 양팔 탄도 투척 및 공중 요격을 실증하는 MuJoCo 3.x 기반 고정밀 피지컬 AI 로보틱스 벤치마크.  
> *[ 🌐 영문 문서 (English) ](README.md) | [ 🇰🇷 한국어 매뉴얼 ](README_KR.md) | [ 🎮 실시간 인터랙티브 웹 조종석 ](https://huggingface.co/spaces/hwihwalab/aloha-bimanual-toss-catch)*

> [!TIP]
> 🎮 **브라우저 즉시 체험 (설치 불필요)**: [👉 Hugging Face Spaces 실시간 데모 열기](https://huggingface.co/spaces/hwihwalab/aloha-bimanual-toss-catch)  
> 📦 **공식 모델 허브**: [🤗 hwihwalab/aloha-bimanual-toss-catch-model](https://huggingface.co/hwihwalab/aloha-bimanual-toss-catch-model) | 🐙 **GitHub 저장소**: [Hwihwa-Lab/aloha-bimanual-toss-catch](https://github.com/Hwihwa-Lab/aloha-bimanual-toss-catch) | 📊 **데이터셋 허브**: [hwihwalab/aloha-bimanual-toss-catch-telemetry](https://huggingface.co/datasets/hwihwalab/aloha-bimanual-toss-catch-telemetry)

<div align="center">
  <img src="assets/cockpit_preview.png" alt="ALOHA Toss & Catch 60fps 텔레메트리 콕핏" width="100%" style="border-radius: 8px; border: 1px solid rgba(0,240,255,0.35); box-shadow: 0 12px 40px rgba(0,0,0,0.85);" />
  <p><em>그림 1: 4단계 위상 HUD, 탄도 고도 및 정점 델타 게이지, 요격 레이더, 듀얼 손목 PiP 카메라가 결합된 실시간 60fps 텔레메트리 콕핏. (실시간 요격 영상은 우측 상단의 <strong>Video Preview</strong> 코너에서 확인 가능)</em></p>
</div>

---

## 🕹️ 실시간 인터랙티브 웹 조종석 (`run_live_cockpit.py`)

명령어를 실행하면 **전용 독립 윈도우 앱 창이 자동으로 화면에 바로 팝업**되어 즉시 시작할 수 있습니다 (수동 접속 시 `http://localhost:8080`):

```bash
python run_live_cockpit.py
```

```
[ 🚀 LAUNCH | ⏸️ PAUSE | 🔄 RESET ] │ MODE: [ MANUAL | AUTO | CHAOS RUSH ⚡ ] │ SHOT: [ 🎯 BULLET | 🌈 LOB | ↩️ HOOK | ⚡ BEAST ] [AIM YAW: L ──●── R 0.0°] │ CAM: [ 1 FRONT | 2 SIDE | 3 TOP | 4 L-WRIST | 5 R-WRIST ]
```

### ⌨️ 콕핏 단축키 및 키보드 조작 일람표 (Keyboard Shortcuts)

| 키 (Key) | 기능 (Action) | 상세 설명 (Description) |
| :---: | :--- | :--- |
| **`Space` / `Enter`** | **🚀 발사 / 일시정지** | 대기 상태에서 투척 개시 / 비행 중 물리 일시정지 |
| **`P`** | **⏸️ 일시정지 (PAUSE)** | 물리 시뮬레이션 즉각 일시정지 및 실시간 재개 |
| **`R`** | **🔄 초기화 (RESET)** | 에피소드 즉각 초기화 및 로비 런치패드 복귀 |
| **`A`** | **🤖 자동 루프 (AUTO)** | 무한 자동 평가 및 정책 벤치마킹 연속 루프 토글 |
| **`C`** | **⚡ 카오스 러시 (CHAOS)** | 매번 무작위 3D 각도로 쉴 새 없이 공이 날아오는 서바이벌 모드 |
| **`←` / `→`** | **🎯 수평 조준 (AIM YAW)** | 좌측 투척 로봇의 수평 조준각 1.0° 단위 미세 조절 ($-15^\circ \sim +15^\circ$) |
| **`1` ~ `5`** | **🎥 카메라 전환** | `1`: 프론트 개요 \| `2`: 사이드 시네마틱 \| `3`: 탑뷰 전술 \| `4`: 좌측 손목 \| `5`: 우측 손목 |
| **`S` / `M`** | **🔊 사운드 (SFX)** | 복고풍 아케이드 캐치/실패 효과음 및 음소거 토글 |
| **`F`** | **🖥️ 전체화면** | 브라우저 전체화면 콕핏 모드 전환 |
| **`?` / `ESC`** | **⌨️ 핫키 가이드** | 와이드 글래스모피즘 사이버 단축키 레퍼런스 HUD 열기 / 닫기 |

### 🎮 3대 시뮬레이션 모드 & 4대 시그니처 물리 구종

1. **3대 다이내믹 게임 모드 (Game Modes)**:
   - `MANUAL`: 사용자가 직접 조준하고 1구씩 발사하는 정밀 제어 모드.
   - `AUTO`: 무한 연속 훈련 및 정책 평가 벤치마킹 자동화 루프.
   - `CHAOS RUSH ⚡`: 매번 무작위 3D 각도(피치/요)로 쉴 새 없이 공이 날아오는 서바이벌 아케이드 모드.

2. **4대 시그니처 탄도 구종 (4 Signature Ballistic Trajectories)**:
   - 🎯 **`BULLET`**: 전진 속도 $2.33\text{ m/s}$의 수평 초고속 직구 (18스텝 만에 요격 강타).
   - 🌈 **`LOB`**: 수직 상승 $0.61\text{ m/s}$의 고공 무지개 포물선 (긴 체공 시간 후 26스텝 만에 낙하).
   - ↩️ **`HOOK`**: $-10.0^\circ$ 좌측 사선으로 급격히 휘어지는 훅 궤적.
   - ⚡ **`BEAST`**: $+10.0^\circ$ 우측 원거리 극한 와이드 샷 (HOOK과 좌우 $22\text{ cm}$ 궤적 차이).


---

## ⚡ 과제 배경: 모방 학습(Imitation Learning)의 한계를 넘어서

기존의 양손 매니퓰레이션 벤치마크(자사의 [**ALOHA 14자유도 Transfer Cube ACT**](https://huggingface.co/hwihwalab/act-aloha-sim-transfer-cube) 등)는 사람이 직접 조종한 데이터를 기반으로 하는 모방 학습(ACT / Diffusion Policy)의 **준정적(Quasi-static) 환경**입니다.

그러나 **인간의 운동 반응 지연(300ms)으로 인해 고속 동적 작업의 텔레오퍼레이션 데이터셋 제작은 물리적으로 불가능**합니다. **`AlohaTossCatch-v0`**는 순수 강화학습으로 이 한계를 돌파합니다:

```
[ 투척 팔 (왼쪽 7-DOF) ]                     [ 캐칭 팔 (오른쪽 7-DOF) ]
   견고한 파지 ➔ 정점 가속 투척                   포물선 비행 궤적 예측
   최적 릴리즈 각도 (Z = 1.25m)  ════════>    400ms 이내 초고속 요격
                                              능동 감쇠 및 완충 안착
```

1. **투척 팔 (왼쪽 7-DOF):** 공기역학 캡슐을 최적의 포물선 정점($X=0.0\text{m}, Z=1.25\text{m}$)을 향해 가속한 뒤 정밀한 타이밍에 릴리즈.
2. **캐칭 팔 (오른쪽 7-DOF):** 탄도 비행 시간을 예측하여 400ms 이내에 착지점으로 진입한 뒤, 능동 감쇠(Active Damping)를 통해 튕겨 나가지 않도록 **충격 흡수(Cushioning) 완충 캐칭** 수행.

---

## 📊 정량적 벤치마크 성능 비교 매트릭스

| 지표 | 랜덤 베이스라인 | [준정적 ACT (Transfer Cube)](https://huggingface.co/hwihwalab/act-aloha-sim-transfer-cube) | **AlohaTossCatch (Ours, PPO)** |
| :--- | :---: | :---: | :---: |
| **캐칭 성공률** | 0.0 % | 12.4 % (분포 이탈 실패) | **100.0 %** |
| **평균 요격 소요 시간** | N/A | 3.42 초 | **0.59 초** |
| **비행 물체 최고 속도** | 0.4 m/s | 0.2 m/s | **2.61 m/s** |
| **모터 저크(Jerk) 부드러움** | 4.821 | 1.954 | **0.946 (매우 부드러움)** |
| **시뮬레이션 처리량 (Colab)** | N/A | ~45 FPS | **12,500+ FPS (헤드리스)** |

---

## 🔬 4단계 위상 상태 머신 & 다단계 보상 엔진

$$\text{총 보상} = R_{\text{phase}} - \lambda_{\text{torque}} \|\tau\|^2 + R_{\text{success}}$$

* **`Phase 0 (Grasp)`:** $R = 2.0 \cdot \exp(-15 \cdot d_{\text{left}}) + 5.0$ (물체 거리 $<4\text{cm}$ 시 전이).
* **`Phase 1 (Toss)`:** $R = 1.5 \cdot \text{clip}(V_x, 0, 2.5) + 1.2 \cdot \text{clip}(V_z, 0, 2.0) + 2.0 \cdot \exp(-5 \cdot d_{\text{apex}})$.
* **`Phase 2 (Flight)`:** $R = 3.0 \cdot \exp(-8 \cdot d_{\text{catch}}) + 10.0$ (착지점 사전 정렬 시 전이).
* **`Phase 3 (Cushion)`:** $R = 5.0 \cdot \exp(-15 \cdot d_{\text{catch}}) + 15.0 \cdot \text{contact} + 5.0 \cdot \exp(-3 \cdot \|V\|) + \mathbf{50.0}$ (대형 성공 보너스).

---

## ⚡ 빠른 시작 & 로컬 재현 (Quick Start)

### 1. 실시간 인터랙티브 사이버 콕핏 실행 (추천)
브라우저에서 60fps로 실시간 구동되는 탄도 조준 및 시뮬레이션 제어 조종석을 즉시 실행합니다:
```powershell
# 1. 패키지 로컬 설치
pip install -e .

# 2. 실시간 웹 콕핏 실행 (전용 앱 창이 자동으로 팝업 실행됩니다!)
python run_live_cockpit.py
```

### 2. 자동화 평가 및 고해상도 시연 영상 생성
```powershell
python evaluate.py --model checkpoints/aloha_toss_catch_final.zip --output assets/aloha_toss_catch_demo.gif
```

### 3. 시스템 무결성 스모크 테스트 실행
```powershell
python tests/test_smoke.py
```

### 4. 허깅페이스 모델 허브 정책 가중치 원클릭 로드
Hugging Face Model Hub에서 사전 학습된 100% 캐칭 성공 가중치를 즉시 로드합니다:
```python
from huggingface_hub import hf_hub_download
from stable_baselines3 import PPO

checkpoint_path = hf_hub_download(
    repo_id="hwihwalab/aloha-bimanual-toss-catch-model",
    filename="aloha_toss_catch_final.zip"
)
model = PPO.load(checkpoint_path)
print("✅ Hugging Face Model Hub로부터 ALOHA 정책 가중치 로드 성공!")
```

---

## 📂 저장소 아티팩트 매니페스트 (Repository Manifest)

본 파일 배포 구조는 [Hugging Face Model Hub (`hwihwalab/aloha-bimanual-toss-catch-model`)](https://huggingface.co/hwihwalab/aloha-bimanual-toss-catch-model/tree/main)의 공식 릴리즈 매니페스트를 엄격히 준수합니다:

| 파일 경로 (File Path) | 단일 책임 및 아티팩트 역할 (Single Responsibility Description) |
| :--- | :--- |
| `checkpoints/aloha_toss_catch_final.zip` | 공식 사전 학습된 PyTorch / Stable-Baselines3 PPO 정책 가중치 아카이브 (요격 성공률 100%) |
| `aloha_bimanual_toss_catch_bundle.zip` | 원클릭 오프라인 독립 실행 전체 번들 압축본 (웹 콕핏, 물리 환경, 에셋 및 PPO 평가기 포함) |
| `eval_info.json` | 정량적 벤치마크 평가 메타데이터 (100% 성공률, 0.59초 요격, 50Hz 제어, 12,500+ FPS, 저크 0.946) |
| `assets/aloha_bimanual_toss.xml` | 감쇠 완충 패들이 적용된 MuJoCo 3.x MJCF 양팔 동역학 물리 시뮬레이션 모델 |
| `assets/aloha_toss_catch_demo.gif` | 60FPS 고해상도 4대 시그니처 탄도 궤적 요격 시연 GIF |
| `assets/cockpit_preview.png` | 1280x720 고해상도 사이버 텔레메트리 콕핏 UI 프리뷰 이미지 |
| `README.md` | 허깅페이스 공식 영문 모델 카드 및 벤치마크 기술 사양서 |
| `README_KR.md` | 공식 한국어 기술 매뉴얼 및 사용자 가이드 |
| `LICENSE` | 공식 MIT 오픈소스 라이선스 |

---

## 👤 About HWIHWA LAB

> *“AI × Automation × Aesthetics — Crafting tools you'll love ✨”*

**HWIHWA LAB**은 **실용적인 온디바이스 인텔리전스, 자율 운영 자동화, 그리고 감각적인 미학**의 융합을 지향하는 독립 AI 연구 및 엔지니어링 스튜디오입니다. 1인 창업가와 인디 크리에이터가 반복적인 운영 부담에서 벗어나 온전히 창작의 기쁨과 시간의 자유를 누릴 수 있도록 돕는 견고한 소프트웨어를 만듭니다.

### 🔬 핵심 연구 및 엔지니어링 기둥 (Core Pillars)

- **임바디드 AI & 피지컬 다이내믹스 (Embodied AI & Physical Dynamics)**: 물리 법칙 기반 강화학습(RL), 비대칭 듀얼암 고속 탄도 궤적 및 능동 감쇠 제어 두뇌 연구.
- **제로 에셋 절차적 엔지니어링 (Zero-Asset Procedural Engineering)**: 무거운 텍스처 다운로드 없이 순수 코드와 수학적 모델링으로 3D 지오메트리와 오디오를 절차적으로 합성하는 초경량 웹 아키텍처 설계.
- **자율 의사결정 OS (Autonomous Decision OS)**: 복잡한 물리 환경을 위한 전두엽 스타일의 결정론적 추론 및 상태 머신 거버넌스 엔지니어링.
- **인터랙티브 크리에이티브 미학 (Interactive Creative Aesthetics)**: 기능적 자동화와 감각적인 시각적 완성도, 현대적 디자인 철학을 융합한 매력적인 인터페이스 구축.

### 🌐 HWIHWA LAB 로봇 매니퓰레이션 제품군 (Cross-Links)

- **공식 스튜디오 허브**: [🤗 Hugging Face (@hwihwalab)](https://huggingface.co/hwihwalab) · [🐙 GitHub (@Hwihwa-Lab)](https://github.com/Hwihwa-Lab)
- **🚀 로봇 암 조작 및 듀얼암 매니퓰레이션 제품군**:
  1. 🦾 **ALOHA 14자유도 Transfer Cube** · ACT 모방학습 & 3D 실시간 콕핏 ([🤗 Spaces](https://huggingface.co/spaces/hwihwalab/act-aloha-sim-transfer-cube) | [🤗 Model Hub](https://huggingface.co/hwihwalab/act-aloha-sim-transfer-cube)) — *준정적 정밀 큐브 전달 모방학습 베이스라인*
  2. 🎯 **ALOHA 14자유도 Toss & Catch** · PPO 심층 강화학습 ([🤗 Spaces](https://huggingface.co/spaces/hwihwalab/aloha-bimanual-toss-catch) | [🤗 Model Hub](https://huggingface.co/hwihwalab/aloha-bimanual-toss-catch-model)) — *(현재 프로젝트)*
  3. 🤖 **Pusher-v5 PPO** · 7자유도 단일 암 정밀 물체 조작 ([🤗 Spaces](https://huggingface.co/spaces/hwihwalab/pusher-v5-ppo) | [🤗 Model Hub](https://huggingface.co/hwihwalab/pusher-v5-ppo)) — *표준 Gymnasium & MuJoCo 7-DOF 로봇 팔 강화학습 벤치마크*

---

## 📄 라이선스 (License)

본 프로젝트는 MIT License를 따릅니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

*Trained and deployed with [ALOHA Toss & Catch Physical AI Lab](https://huggingface.co/spaces/hwihwalab/aloha-bimanual-toss-catch) by **HWIHWA LAB**.*
