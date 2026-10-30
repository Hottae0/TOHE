# TOHE experiments

확장 실험(1,800 QA, 300 subgraph, 1 seed, 원래 좌표 포함 11조건)은 [확장 실행 안내](docs/EXPANDED_STUDY.md)를 사용하세요. 결과는 검증된 영구 마운트 아래에 저장하며, 기존 small 질문과 evidence를 제외합니다.

WordNet graph → text → frozen Qwen 실험입니다. **영어 질문 생성과 QA 평가 모두 Qwen/Qwen2.5-7B-Instruct를 사용**합니다. 정답은 그래프 알고리즘으로 계산합니다. LLM과 코드북은 고정하고 evidence subgraph의 노드 좌표만 학습합니다.

7개 조건은 direct_id, direct_bfs, direct_dfs, euclidean, poincare, v1, v2입니다. 모든 조건에 같은 직접 간선을 제공하며 임베딩 조건은 고정 규칙의 반경·방향 코드를 추가합니다. 다중 부모를 보존합니다. [프로토콜](docs/PROTOCOL.md), [실행 기록](docs/VALIDATION.md), [한계](docs/RESEARCH_GAPS.md)를 참고하세요. 손실항별 ablation은 아직 포함하지 않습니다.

## 기존 실행의 질문 오류 수정

새 기본 흐름은 Qwen 생성 → 규칙 검사 → 별도 Qwen 호출로 의미 검수 → 실패 질문 재생성입니다. **[기존 임베딩을 재사용하는 업데이트·실행 안내](docs/QA_REPAIR.md)**를 먼저 확인하세요. 이전 질문·평가 결과는 보존되고 새 파일·경로를 사용합니다.

## 설치

Linux 서버에서 실행합니다. 아래 실험 가상환경은 CUDA용 PyTorch가 설치된 NGC 환경을 가정합니다.

```bash
git clone --branch codex/wordnet-pilot https://github.com/Hottae0/TOHE.git
cd TOHE
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available()'
python -m unittest discover -s tests -v
```

CPU로 임베딩을 학습할 때는 -cuda가 없는 config를 사용하세요. Dockerfile은 CPU 실험 코드만 포함하며 Qwen 서버는 포함하지 않습니다.

## Small과 전체

| 항목 | Small | 전체 |
|---|---|---|
| GPU config | configs/small-qwen2.5-7b-cuda.json | configs/qwen2.5-7b-cuda.json |
| CPU config | configs/small-qwen2.5-7b.json | configs/qwen2.5-7b.json |
| epochs / burn-in | 300 / 20 | 300 / 20 |
| seeds | 11 | 11, 22, 33 |
| evidence 수 / checkpoint 수 | 20 / 80 | 300 / 3,600 |
| 문맥당 노드 | 20·40 | 20·40·80 |
| QA 수 | 120 | 1,800 |
| 개발 / 테스트 QA | 30 / 90 | 300 / 1,500 |
| 질문 생성·검수 요청 (최소) | 240 | 3,600 |
| 개발 / 테스트 평가 요청 | 210 / 630 | 4,500 / 22,500 |

질문은 reachability, common direct parents, exact-k의 세 유형입니다. 원본 synset ID와 연산 기준으로 전역 중복 제거합니다. 개발/테스트의 노드와 부분 구조는 겹칠 수 있습니다. Small의 단일 seed 결과는 최종 연구 결과가 아닙니다.

## 데이터와 임베딩 학습

영구 저장 폴더에서 실행하세요. Small과 전체는 별도 ROOT를 사용합니다. 기존 질문 생성 모델이나 학습 설정을 바꾸고 같은 실행을 재개하지 마세요.

```bash
source .venv/bin/activate
export CONFIG=configs/small-qwen2.5-7b-cuda.json
export ROOT=runs/small-qwen-local-300ep
export OMP_NUM_THREADS=4

python scripts/run_experiment.py --stage prepare --config "$CONFIG" --root "$ROOT"
python scripts/run_experiment.py --stage specs --config "$CONFIG" --root "$ROOT"
python -u scripts/run_experiment.py --stage train --config "$CONFIG" --root "$ROOT"
```

prepare는 WordNet과 고정 upstream mammals 필터를 내려받습니다. specs는 evidence, 질문 명세, 알고리즘 정답을 고정합니다. 이 두 단계는 최초 한 번만 실행합니다.

각 evidence 내부의 노드·직접 간선·로컬 추이 관계만 사용해 네 임베딩 조건을 독립 학습합니다. 같은 evidence의 여섯 질문은 checkpoint를 재사용합니다. 전체 WordNet 좌표나 외부 negative를 사용하지 않습니다. 중단되면 train만 재실행합니다. 기존 30-epoch 또는 전체 그래프 checkpoint와 호환되지 않습니다.

## Qwen 서버

임베딩 GPU 학습이 끝난 다음 같은 서버의 별도 터미널에서 실행하세요. vLLM의 PyTorch 의존성을 분리하기 위해 새 가상환경을 사용합니다. 아래는 GPU 메모리 24GB 이상을 시작 기준으로 한 설정이며 실제 가용 메모리와 GPU/드라이버 호환성을 확인해야 합니다. 시스템 RAM과 GPU 메모리는 다릅니다.

```bash
python3 -m venv .venv-vllm
source .venv-vllm/bin/activate
python -m pip install --upgrade pip
python -m pip install vllm
mkdir -p runs/server
python -m pip freeze > runs/server/vllm-environment.txt
nvidia-smi > runs/server/gpu.txt

vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 127.0.0.1 --port 8000 \
  --dtype half --max-model-len 4096 --max-num-seqs 1 \
  --gpu-memory-utilization 0.85 --generation-config vllm --seed 11
```

최초 실행 시 모델을 다운로드합니다. 서버가 준비된 뒤 터미널을 유지하세요. 모델 revision, vLLM/PyTorch 버전, dtype, chat template, GPU와 실행 옵션을 보존하세요. [Qwen 공식 안내](https://qwen.readthedocs.io/en/v2.5/deployment/vllm.html), [vLLM 설치 안내](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)를 참고하세요.

## 영어 질문 생성과 개발 평가

원래 실험 터미널에서 실행합니다. 로컬 서버 기본 설정에는 API 키가 필요 없습니다. 인증 서버를 별도로 구성했다면 TOHE_QWEN_KEY를 export하세요. .env.example은 참고용이며 자동 로드하지 않습니다.

```bash
export TOHE_QWEN_ENDPOINT=http://127.0.0.1:8000/v1/chat/completions
unset TOHE_QWEN_KEY
curl --fail http://127.0.0.1:8000/v1/models

python scripts/run_experiment.py --stage generate --config "$CONFIG" --root "$ROOT" --max-calls 2
python scripts/run_experiment.py --stage generate --config "$CONFIG" --root "$ROOT" --max-calls 720
python scripts/run_experiment.py --stage evaluate --config "$CONFIG" --root "$ROOT" --split dev --max-calls 210
python scripts/run_experiment.py --stage report --config "$CONFIG" --root "$ROOT" --split dev
```

Qwen이 영어 질문을 작성하고, 규칙 검사 후 별도 호출로 문장 의미를 역해석해 검수합니다. 검수에는 원본 명세·정답·그래프를 주지 않습니다. 실패 문항은 다시 생성하며 완료 상태는 model_verified_v3입니다. 같은 모델의 자동 검수이므로 유형별 사람 표본 검수도 필요합니다. 모든 평가 프롬프트에는 formal task specification도 포함됩니다. 생성 temperature는 0.2, 평가 temperature는 0입니다. 동일 모델로 생성·평가하므로 문체 편향 가능성이 남습니다.

## 테스트와 전체 실행

개발 결과로 설정을 확정한 뒤 테스트합니다.

```bash
python scripts/run_experiment.py --stage evaluate --config "$CONFIG" --root "$ROOT" --split test --max-calls 630
python scripts/run_experiment.py --stage report --config "$CONFIG" --root "$ROOT" --split test
```

전체 실험은 CONFIG=configs/qwen2.5-7b-cuda.json, ROOT=runs/full-qwen-local-300ep로 바꾸고 prepare → specs → train → generate → 개발 평가/보고 → 테스트 평가/보고 순서로 실행합니다. 호출 한도는 생성·검수 10,800(최소 3,600), 개발 4,500, 테스트 22,500입니다. GPU 학습을 다시 할 때는 기존 vLLM 서버를 먼저 종료해 메모리를 확보하세요. Small에서 전체로 자동 확대하거나 결과를 합치지 않습니다.

--max-calls는 해당 실행의 새 논리 요청 수 상한입니다. 재실행하면 완료 항목을 건너뛰며 HTTP 일시 오류 재시도는 별도입니다. 질문·설정·checkpoint가 다른 캐시는 거부합니다. 현재 클라이언트는 순차 처리합니다.

질문 생성·검수 실패의 전체 기록은 data/qa.audited-v3.generation-failures.jsonl에 저장됩니다. 재시도와 검수도 max-calls에 포함됩니다. 예산 도달 시 같은 명령을 재실행하면 완료 문항과 검수 대기 후보를 재사용합니다. 상세 동작과 경로는 [QA 수정 안내](docs/QA_REPAIR.md)를 참고하세요.

## 결과

- checkpoints/<context_id>/evidence.json: 실제 학습 evidence.
- checkpoints/<context_id>/*.metrics.json: reconstruction/code 지표.
- evaluation/<model>/prompt-v2/audited-v3/<split>/predictions.jsonl: prompt, 응답, 정답, exact match, set F1, usage.
- 평가 폴더의 summary.csv, paired_comparisons.json: 조건·seed·유형별 결과와 Direct ID 대비 paired 비교. 불완전 실행에는 신뢰구간을 산출하지 않습니다.

실제 Qwen 서버와 300-epoch 전체 연구는 로컬 검증에서 실행하지 않았습니다. 테스트는 python -m unittest discover -s tests -v, 실제 WordNet 짧은 검증은 python scripts/validate_wordnet.py <graph.json> <new-output-dir>로 수행합니다. 후자는 LLM을 호출하지 않습니다.
