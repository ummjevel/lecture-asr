---
name: lecture-asr-build
description: "강의실 음성 전사 도구(lecture-asr)의 전체 빌드를 조율하는 오케스트레이터. 전처리/ASR/CLI 모듈을 병렬 구현하고 QA 검증까지 수행. 'lecture-asr 빌드', '전사 도구 구현', '모듈 구현 시작' 요청 시 사용."
---

# Lecture ASR Build Orchestrator

강의실 음성 전사 도구의 에이전트를 조율하여 전체 파이프라인을 구현하는 통합 스킬.

## 실행 모드: 서브 에이전트

## 에이전트 구성

| 에이전트 | subagent_type | 역할 | 스킬 | 출력 |
|---------|--------------|------|------|------|
| preprocess-builder | preprocess-builder | 오디오 전처리 5개 모듈 | preprocess-build | pipeline/converter~agc.py |
| asr-builder | asr-builder | ASR+후처리 4개 모듈 | asr-build | pipeline/asr~postprocess.py |
| cli-builder | cli-builder | CLI+UX 4개 모듈 | cli-build | transcribe.py + ui/*.py |
| qa-inspector | qa-inspector | 통합 정합성 검증 | qa-inspect | _workspace/qa_report.md |

## 워크플로우

### Phase 1: 준비

1. 사용자 입력 확인 — 설계 문서 `docs/design.md` 존재 확인
2. `_workspace/` 디렉토리 생성
3. 프로젝트 디렉토리 구조 생성:
   ```
   mkdir -p pipeline ui
   ```
4. 공유 데이터 모델 먼저 생성 — `pipeline/models.py` (TranscriptionResult, Segment)
   모든 에이전트가 이 파일을 참조하므로 오케스트레이터가 직접 생성한다.

### Phase 2: 병렬 구현 (팬아웃)

단일 메시지에서 3개 Agent 도구를 동시 호출:

| 에이전트 | 입력 | 출력 | model | run_in_background |
|---------|------|------|-------|-------------------|
| preprocess-builder | docs/design.md + pipeline/models.py | pipeline/converter.py ~ agc.py, pipeline/__init__.py | opus | true |
| asr-builder | docs/design.md + pipeline/models.py | pipeline/asr.py ~ postprocess.py | opus | true |
| cli-builder | docs/design.md + pipeline/models.py | transcribe.py + ui/*.py | opus | true |

각 에이전트에게 전달할 프롬프트:
- 설계 문서(`docs/design.md`)를 Read하라
- 공유 모델(`pipeline/models.py`)을 Read하라
- 해당 스킬(`Skill` 도구로 호출)을 따라 모듈을 구현하라
- `requirements.txt`에 필요한 패키지를 추가하라 (없으면 생성)

### Phase 3: 통합 검증 (팬인)

Phase 2의 3개 에이전트 모두 완료 후:

1. qa-inspector 에이전트 호출 (foreground):
   ```
   Agent(
     subagent_type: "qa-inspector",
     model: "opus",
     prompt: "docs/design.md를 Read하고, Skill 도구로 qa-inspect 스킬을 호출하여
              전체 프로젝트의 통합 정합성을 검증하라.
              결과를 _workspace/qa_report.md에 저장하라."
   )
   ```

2. QA 리포트 확인:
   - CRITICAL 이슈가 있으면 → Phase 4로
   - CRITICAL 없으면 → Phase 5로

### Phase 4: 수정 (CRITICAL 이슈 시)

QA 리포트의 CRITICAL 이슈를 수정:
1. 리포트에서 이슈 목록 추출
2. 해당 모듈의 에이전트를 다시 호출하여 수정 지시
3. 수정 후 qa-inspector를 다시 호출하여 재검증
4. 최대 2회 반복 후 강제 진행

### Phase 5: 정리

1. `requirements.txt` 통합 — 3개 에이전트가 각각 추가한 내용을 중복 제거하여 정리
2. `_workspace/` 디렉토리 보존
3. 사용자에게 결과 요약 보고:
   - 생성된 파일 목록
   - QA 결과 요약
   - 설치 및 실행 방법 안내

## 데이터 흐름

```
docs/design.md ──Read──→ [preprocess-builder] ──→ pipeline/converter~agc.py
                    │
                    ├──→ [asr-builder] ──→ pipeline/asr~postprocess.py
                    │
                    └──→ [cli-builder] ──→ transcribe.py + ui/*.py
                                              │
                                              ▼
                                    [qa-inspector] ──→ _workspace/qa_report.md
                                              │
                                      CRITICAL? ──→ 수정 루프
                                              │
                                              ▼
                                         완료 보고
```

## 에러 핸들링

| 상황 | 전략 |
|------|------|
| 에이전트 1개 실패 | 1회 재시도. 재실패 시 해당 모듈 없이 진행, 리포트에 명시 |
| 에이전트 과반 실패 | 사용자에게 알리고 진행 여부 확인 |
| QA CRITICAL 2회 연속 | 수정 중단, 이슈 리스트를 사용자에게 제시 |
| requirements.txt 충돌 | 버전이 다르면 최신 버전 채택 |

## 테스트 시나리오

### 정상 흐름
1. 사용자가 "lecture-asr 빌드" 요청
2. Phase 1에서 docs/design.md 확인, models.py 생성
3. Phase 2에서 3개 에이전트 병렬 실행, 각각 모듈 구현
4. Phase 3에서 qa-inspector가 통합 검증, CRITICAL 없음
5. Phase 5에서 requirements.txt 정리, 결과 보고
6. 예상 결과: 15개 Python 파일 + requirements.txt 생성

### 에러 흐름
1. Phase 2에서 asr-builder가 mlx-qwen3-asr API 불명확으로 실패
2. 1회 재시도 후 성공
3. Phase 3에서 QA가 CRITICAL 발견: asr.py의 반환 타입이 postprocess.py와 불일치
4. Phase 4에서 asr-builder를 재호출하여 수정
5. 재검증 통과, Phase 5로 진행
