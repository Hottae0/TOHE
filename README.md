# TOHE experiments

WordNet graph → text → frozen LLM 실험 코드입니다. 기본 7개 비교군, 고정 serializer, v1·v2 임베딩 학습, 영어 질문 생성·평가를 제공합니다. **손실항별 ablation 코드는 포함하지 않았습니다.**

이 저장소는 연구 구현이며, 논문 성능을 재현했거나 TOHE가 우수하다고 주장하지 않습니다. 현재 검증 범위는 [실행 기록](docs/VALIDATION.md), 방법의 구체적인 정의는 [프로토콜](docs/PROTOCOL.md), 부족한 비교와 대안 데이터셋은 [연구 검토](docs/RESEARCH_GAPS.md)에 있습니다.

## 실험 구성

| 조건 | LLM 입력 | 학습 |
|---|---|---|
| direct_id | ID 순서의 직접 관계 | 없음 |
| direct_bfs | BFS 방문 순서로 배치한 모든 직접 관계 | 없음 |
| direct_dfs | DFS 방문 순서로 배치한 모든 직접 관계 | 없음 |
| euclidean | 같은 직접 관계 + 고정 규칙의 반경·방향 코드 | 기본 그래프 손실 |
| poincare | 같은 직접 관계 + 고정 규칙의 반경·방향 코드 | 기본 그래프 손실 |
| v1 | 같은 직접 관계 + v1에서 유도된 코드 | graph + radial + sector + collision |
| v2 | 같은 직접 관계 + v2에서 유도된 코드 | graph + radial + code-structure + hard |

- WordNet mammals, 상하위 방향은 **child → parent**.
- 학습은 추이적 폐쇄, LLM facts는 직접 간선입니다. 다중 부모를 보존합니다.
- LLM·코드북은 학습하지 않습니다. 노드 좌표를 학습합니다.
- 기본 설정은 5차원, 300 epochs, seed 11/22/33. 학습형 코드북/VQ/RQ 없음.
- QA 목표: 1,800개, 세 유형 각각 600개. 개발 300개 / 테스트 1,500개.
- 같은 원본 노드에 같은 연산을 묻는 질문은 전역 중복 제거합니다. 표현 변경·ID 변경도 중복으로 취급합니다. 개발/테스트에 동일 질문은 없지만 **노드와 부분 구조는 겹칠 수 있습니다**.
- 질문 명세와 정답은 그래프 알고리즘으로 생성하고, 영어 질문 문장은 LLM이 작성합니다. 모델 생성이 실제 완료된 항목만 `llm_generated`로 표시됩니다.
- GPT-4o mini와 Qwen2.5-7B-Instruct는 동일한 질문 세트를 사용합니다.

## 서버 설치

Linux 서버의 Python 3.10 이상을 사용합니다. 작은 mammals 임베딩 학습은 CPU로 실행할 수 있습니다. Qwen 추론은 별도의 GPU 서버 또는 호스팅 endpoint를 권장합니다.

```bash
git clone --branch codex/wordnet-pilot https://github.com/Hottae0/TOHE.git
cd TOHE
python -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
python -m unittest discover -s tests -v
```

선택적으로 `docker build -t tohe .`로 CPU 실험 이미지를 만들 수 있습니다. 컨테이너에는 Qwen 서버가 포함되지 않습니다. 실행 기록은 볼륨으로 보존하세요.

## 빠른 소규모 실행

처음에는 `configs/small-gpt4o-mini.json`을 권장합니다. 합성 데이터 smoke test가 아니라 **같은 WordNet mammals와 7개 비교군을 사용하는 짧은 진단 실험**입니다. 전체 설정은 유지됩니다.

| 항목 | 전체 설정 | Small 설정 |
|---|---|---|
| epochs / burn-in | 300 / 20 | 30 / 3 |
| 학습 seeds | 3개 | 1개 |
| QA | 1,800개 | 120개 (유형별 40개) |
| 개발 / 테스트 QA | 300 / 1,500 | 30 / 90 |
| 문맥당 노드 | 20·40·80 | 20·40 |
| LLM 하나의 테스트 요청 | 22,500회 | 630회 |

학습 원본 그래프, 손실 정의·가중치, 차원, batch size, negative 수, 코드 변환과 영어 프롬프트는 동일합니다. 학습 epoch×seed 작업량은 1/30이며 실제 시간은 서버에 따라 다릅니다. 낮은 정확도가 나오더라도 짧은 학습·작은 표본의 영향과 방법 자체의 한계를 구분해야 합니다. 논문용 최종 결과로 사용하지 마세요.

설치와 아래 LLM 환경변수 설정 후, **전체 실험과 다른 저장 경로**로 실행합니다:

```bash
CONFIG=configs/small-gpt4o-mini.json
ROOT=runs/small
python scripts/run_experiment.py --stage prepare --config "$CONFIG" --root "$ROOT"
python scripts/run_experiment.py --stage specs --config "$CONFIG" --root "$ROOT"
python scripts/run_experiment.py --stage train --config "$CONFIG" --root "$ROOT"
python scripts/run_experiment.py --stage generate --config "$CONFIG" --root "$ROOT" --max-calls 120
python scripts/run_experiment.py --stage evaluate --config "$CONFIG" --root "$ROOT" --split dev --max-calls 210
python scripts/run_experiment.py --stage report --config "$CONFIG" --root "$ROOT" --split dev
python scripts/run_experiment.py --stage evaluate --config "$CONFIG" --root "$ROOT" --split test --max-calls 630
python scripts/run_experiment.py --stage report --config "$CONFIG" --root "$ROOT" --split test
```

Qwen을 추가로 비교할 때는 `CONFIG=configs/small-qwen2.5-7b.json`으로 바꾸고 **같은 ROOT에서 evaluate/report만** 실행합니다. 질문 생성·학습을 반복하지 않습니다. 처음 GPT 모델 하나만 돌려도 됩니다. Small도 의미 기준 QA 중복 제거를 동일하게 적용합니다. 이미 전체 실험을 실행 중이라면 해당 실행을 자동으로 변경하거나 중단하지 않습니다.

## 1. 데이터 준비

```bash
python scripts/run_experiment.py --stage prepare --config configs/gpt4o-mini.json
python scripts/run_experiment.py --stage specs --config configs/gpt4o-mini.json
```

첫 명령은 NLTK WordNet과 고정된 upstream commit의 mammals 필터를 내려받습니다. 두 번째는 중복 없는 질문 명세·알고리즘 정답을 만듭니다. **이 단계에서는 아직 LLM이 작성한 질문이 없습니다.**

결과는 `runs/main/data/`에 저장됩니다. 준비된 데이터나 질문 세트를 실수로 덮어쓰지 않습니다. 새 실험은 `--root runs/new-study`처럼 별도 경로를 지정합니다. 모든 단계에서 같은 `--root`를 사용하세요.

## 2. 네 임베딩 조건 학습

```bash
python scripts/run_experiment.py --stage train --config configs/gpt4o-mini.json
```

4조건 × 3 seeds를 학습하고 각 checkpoint의 reconstruction/code 지표를 저장합니다. 각 epoch마다 partial checkpoint와 난수 상태를 보존합니다. 중단되면 같은 명령으로 다시 실행합니다. 완료된 checkpoint는 검증 후 재사용합니다. 학습 설정을 바꾸면 새 `--root`를 사용하세요.

기본 하이퍼파라미터는 시작 설정입니다. 추가 손실 가중치가 개발셋에서 최적화되었다고 주장하지 않습니다. 개발 평가로 설정을 고정한 다음 테스트를 실행하세요.

## 3. LLM 설정 및 영어 질문 생성

비밀 키는 저장소에 넣지 않습니다. `.env.example`을 참고해 환경변수를 설정합니다. 파일을 자동으로 읽지는 않습니다.

```bash
export OPENAI_API_KEY='YOUR_KEY'
export TOHE_GENERATOR_ENDPOINT='https://api.openai.com/v1/chat/completions'
export TOHE_CHAT_ENDPOINT='https://api.openai.com/v1/chat/completions'
```

GPT 모델명은 재현성을 위해 `gpt-4o-mini-2024-07-18`로 지정했습니다. 계정에서 지원되는지 확인해야 합니다. 모델을 바꾸면 config에 정확한 이름을 기록하세요.

```bash
python scripts/run_experiment.py --stage generate --config configs/gpt4o-mini.json --max-calls 1800
```

같은 명령을 다시 실행하면 완료된 문항을 건너뜁니다. 각 문항에는 모델명·원문 출력·사용 토큰·질문 명세·알고리즘 정답이 저장됩니다. 자연어 문장의 의미 검수는 별개입니다. `semantic_review=pending` 항목을 유형별로 표본 검수하세요. 평가 프롬프트에는 정답 없이 정확한 formal task specification도 포함됩니다.

## 4. 개발 평가 후 테스트

먼저 작은 호출 한도로 endpoint와 응답 형식을 확인할 수 있습니다. 같은 실행 명령을 재실행하면 이어서 평가합니다.

```bash
python scripts/run_experiment.py --stage evaluate --config configs/gpt4o-mini.json --split dev --max-calls 30
python scripts/run_experiment.py --stage evaluate --config configs/gpt4o-mini.json --split dev --max-calls 4500
python scripts/run_experiment.py --stage report --config configs/gpt4o-mini.json --split dev
```

개발 단계에서 설정을 결정한 뒤 테스트를 실행합니다.

```bash
python scripts/run_experiment.py --stage evaluate --config configs/gpt4o-mini.json --split test --max-calls 22500
python scripts/run_experiment.py --stage report --config configs/gpt4o-mini.json --split test
```

Direct 3조건은 seed와 무관해 한 번씩, 임베딩 4조건은 3 seeds씩 평가하므로 한 LLM당 테스트 호출은 **1,500 × (3 + 4 × 3) = 22,500회**입니다. `--max-calls`는 한 실행에서 새로 수행할 논리 요청의 상한이며, 429/일시적 서버 오류 재시도는 별도입니다. 누락·중단된 결과는 전체 결과처럼 보고하지 않습니다.

## 5. 같은 QA로 Qwen 평가

GPU 서버의 별도 환경에서 [Qwen 공식 vLLM 안내](https://qwen.readthedocs.io/en/v2.5/deployment/vllm.html)를 따라 서버를 실행합니다. 최소 예:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000
```

모델 revision, dtype, vLLM 버전, chat template 설정은 실행 기록에 남겨야 합니다. 이 저장소는 vLLM 자체나 모델 가중치를 설치하지 않습니다. Windows 검증 환경에서는 vLLM을 실행하지 않았습니다.

평가 프로세스에서는:

```bash
export TOHE_QWEN_ENDPOINT='http://127.0.0.1:8000/v1/chat/completions'
python scripts/run_experiment.py --stage evaluate --config configs/qwen2.5-7b.json --split dev --max-calls 4500
python scripts/run_experiment.py --stage evaluate --config configs/qwen2.5-7b.json --split test --max-calls 22500
python scripts/run_experiment.py --stage report --config configs/qwen2.5-7b.json --split test
```

Qwen용 질문을 다시 생성하거나 임베딩을 다시 학습하지 않습니다. 같은 QA와 checkpoint를 재사용합니다. GPT와 Qwen은 별도 결과 폴더에 저장됩니다. 평가 temperature는 둘 다 0으로 고정합니다.

## 결과와 점검

- `checkpoints/*.metrics.json`: reconstruction, hard-code 지표, 학습 시간.
- `evaluation/<model>/<split>/predictions.jsonl`: prompt, 응답, 정답, exact match, set F1, 실제 API 토큰 사용량.
- `summary.csv`: 조건·seed 및 질문 유형/크기/정답 크기/k/경로 길이별 성능.
- `paired_comparisons.json`: Direct ID 대비 paired 차이와 그래프 단위 bootstrap 구간. 불완전한 실행에는 구간을 산출하지 않습니다.
- 토큰 usage를 제공하지 않는 endpoint에서는 토큰 수를 비워 두며 글자 수를 토큰 수로 위장하지 않습니다.
- HTTP 오류, 모델 미설정, checkpoint 불일치, 미생성 QA는 명시적으로 중단합니다.

학습 없이 코드 흐름만 확인하려면 `configs/smoke.json`과 `python -m unittest discover -s tests -v`를 사용하세요. 실제 WordNet의 짧은 개발 검증은 `python scripts/validate_wordnet.py <graph.json> <new-output-dir>`로 수행할 수 있으며 **본 실험 결과가 아닙니다**.
