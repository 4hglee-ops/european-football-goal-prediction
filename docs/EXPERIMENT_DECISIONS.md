# 실험 결정 기록과 노트북 결론 삽입안

## 이 문서의 목적

이 문서는 결과를 별도로 요약하는 문서이면서, 각 노트북의 기존 결론 형식을 유지한 채 내용을 채우기 위한 기준안이다.

노트북마다 제목 깊이와 결론 구조가 다르므로 공통 `## 결론`을 일괄 삽입하지 않는다. 각 절에는 다음을 명시한다.

- **적용 위치**: 기존 셀의 어느 부분을 교체하거나 어디에 새 셀을 추가할지
- **적용 방식**: 기존 구조를 유지할지 새 Markdown 셀을 만들지
- **삽입안**: 노트북에 그대로 옮길 수 있는 Markdown

결론은 다음 네 가지를 포함한다.

1. 무엇을 비교했는가?
2. 숫자로 무엇이 확인됐는가?
3. 무엇을 채택·유지·보류했는가?
4. 이번 실험의 한계와 다음 단계는 무엇인가?

---

## 01. `01_long_basic_data_understanding.ipynb`

**적용 위치:** 기존 `### 이상치 처리 결정` 다음

**적용 방식:** 새 Markdown 셀 추가. 기존 노트북이 `###` 수준으로 결과를 정리하므로 같은 수준을 사용한다.

```markdown
### 결론

- 현재 시즌 `goals`와 다음 시즌 `next_goals`의 상관계수는 약 0.65로 가장 높았고, non-penalty goals와 per90 공격 지표도 비교적 강한 양의 관계를 보였다.
- FW의 다음 시즌 평균 득점은 약 4.99골, MF는 약 1.44골로 포지션 차이가 리그 차이보다 컸다. 따라서 `position_group`을 주요 범주형 피처로 유지한다.
- 고득점 극단값은 예측하려는 핵심 사례이므로 일괄 제거하지 않는다. 일부 비정상적인 starts/minutes는 선수 성과가 아니라 집계 오류일 수 있으므로 별도 감사 대상으로 남긴다.
- `matched_next`는 Model A의 타깃 및 데이터 감사에만 사용하고 입력 피처에서는 제외한다. `next_10plus`도 `next_goals`에서 파생된 타깃이므로 회귀 피처에서 제외한다.
- 다음 단계에서는 전처리기를 Train에만 fit하고 Validation/Test에는 transform만 적용한다.
```

## 02. `02_long_basic_preprocessing.ipynb`

**적용 위치:** 기존 `### 전처리 결과 확인` 다음

**적용 방식:** 새 Markdown 셀 추가

```markdown
### 결론

- 13개 원본 피처는 수치형 표준화와 범주형 One-Hot Encoding 후 18개 입력 컬럼으로 변환됐다.
- Train 수치형 피처의 평균은 0, 표준편차는 약 1로 정상 변환됐다.
- 전처리기는 Train에만 fit하고 Validation/Test에는 transform만 적용했으므로 평가 구간의 분포 정보가 학습에 들어가지 않는다.
- 이 전처리 파이프라인을 03의 MLP baseline 입력으로 고정한다.
```

## 03. `03_long_basic_mlp_regression.ipynb`

**적용 위치:** 기존 `### Baseline MLP Validation 결과` 다음

**적용 방식:** 새 Markdown 셀 추가

```markdown
### 결론

- Baseline MLP는 단일 Validation에서 MAE 2.389, RMSE 3.486, R² 0.418을 기록했다.
- 평균 예측 baseline의 MAE 3.174보다 약 0.78골 낮아, 해당 Validation 구간에서 현재 시즌 정보가 예측에 도움이 됐다.
- R²가 약 0.42이므로 long_basic 피처만으로 다음 시즌 득점 변동 전체를 설명하지는 못한다.
- Early stopping을 이후 MLP 실험의 기본 절차로 사용한다.
- 이 결과는 한 시즌의 Validation 결과이므로 최종 모델 선택 근거로 사용하지 않고 Walk-forward 평가로 재검증한다.
```

## 04. `04_long_basic_feature_experiments.ipynb`

**적용 위치:** 마지막 embedding 및 고득점 slice 결과 다음

**적용 방식:** 새 Markdown 셀 추가

```markdown
### 결론

- 단일 Validation에서 Exp9Cb tabular MAE는 2.441, Team Embedding은 2.470, Player Embedding은 2.433, Team+Player Embedding은 2.450이었다.
- Player Embedding이 Exp9Cb보다 약 0.008골 앞섰지만 best epoch가 1이고 개선 폭이 작아 안정적인 구조 개선으로 확정하기 어렵다.
- embedding 모델은 일부 고득점 구간 MAE를 줄였지만 모든 모델에서 10+/15+/20+ 득점자를 강하게 과소예측했다.
- Exp9Cb를 이후 Walk-forward 평가의 단순하고 재현 가능한 기준선으로 유지한다. 이는 embedding의 성능이 더 나빠서 폐기한 것이 아니라 작은 단일 분할 개선만으로 복잡도를 채택하지 않은 결정이다.
- 다음 단계에서는 Exp9Cb를 시간 순서 Fold에서 평가해 단일 시즌 결과의 재현성을 확인한다.
```

## 05. `05_walk_forward_evaluation.ipynb`

**적용 위치:** `# 20. 결과 해석 — 직접 작성` 셀의 `## 내 결론` 이하

**적용 방식:** 기존 `## 확인할 질문`은 유지하고 `## 내 결론` 부분만 교체

```markdown
## 내 결론

1. MLP는 Naive baseline보다 일관되게 좋은가?  
→ Yes. 4개 Fold 모두에서 MLP MAE가 낮았다. 평균 MAE는 MLP 2.428, Naive 2.780이었다.

2. 단일 Validation 결과가 Walk-forward에서도 유지되는가?  
→ 단일 Validation MAE 2.389보다 Walk-forward 평균 2.428이 약 0.039골 높았지만, 특정 시즌에서만 작동하는 모델로 보이지는 않는다.

3. Fold 안정성은 어떤가?  
→ MAE는 2.360~2.502, 표준편차는 0.067이었다. 최악 Fold는 2021-22 입력 → 2022-23 타깃 구간이었다.

4. Early Stopping이 필요한가?  
→ 필요하다. Best epoch가 Fold별로 15, 2, 21, 10으로 크게 달라 고정 epoch보다 Fold 내부 조기 종료가 안전하다.

5. 반복적인 실패 패턴은 무엇인가?  
→ 10+ 득점자 bias가 모든 Fold에서 -5.16~-6.39골로 음수였다. 고득점자 과소예측이 구조적인 실패 패턴이다.

6. 다음 모델 비교에서도 동일한 평가 프로토콜을 사용할 것인가?  
→ Yes. 평균, 표준편차, 최악 Fold, 고득점 slice를 포함한 Walk-forward 프로토콜을 06 이후 공통 기준으로 채택한다.

- 추가 확인이 필요한 점: 모델 계열을 바꾸면 고득점자 과소예측과 Fold 안정성이 개선되는지 확인한다.
```

## 06. `06_ml_dl_baseline_comparison.ipynb`

**적용 위치:** `# 23. 결과 해석 — 직접 작성` 셀의 `## 내 결론` 이하

**적용 방식:** 기존 항목명을 유지해 교체

```markdown
## 내 결론

- 평균 MAE 기준 1위: CatBoost 2.389
- 안정성 기준: MAE 표준편차는 Baseline MLP가 0.067로 가장 작았고, 최악 Fold MAE는 CatBoost가 2.458로 가장 좋았다. 하나의 모델이 모든 안정성 지표에서 1위는 아니다.
- 고득점자 기준: 10+ MAE는 ElasticNet이 6.165로 가장 낮았고, 15+/20+ MAE는 CatBoost가 각각 8.319/10.564로 가장 낮았다. 모든 모델에서 강한 음의 bias는 유지됐다.
- 계산 비용 기준: XGBoost가 평균 약 0.86초로 가장 빨랐고 CatBoost는 약 16.8초로 가장 느렸다.
- MLP 추가 튜닝 필요성: CatBoost와 성능 차이가 작아 비교 후보로 유지하고 HPO 효과를 검증한다.
- ML 모델 추가 튜닝 필요성: 기본 비교 1위인 CatBoost를 HPO 대상에 포함한다.
- 다음 단계에서 우선 검증할 후보: CatBoost와 MLP
- 추가 확인이 필요한 점: Inner 최적화가 실제 Outer 성능 개선으로 이어지는지, 고득점자 성능과 전체 MAE 사이의 trade-off가 유지되는지 확인한다.
```

## 07. `07_nested_hpo_catboost_mlp.ipynb`

**적용 위치:** `# 32. 결과 해석 — 직접 작성` 셀의 `## 내 결론` 이하

**적용 방식:** 기존 항목명을 유지해 교체

```markdown
## 내 결론

- 전체 MAE 기준: Tuned MLP 2.392, Tuned CatBoost 2.398로 MLP가 약 0.006골 앞섰다.
- 안정성 기준: MAE 표준편차는 두 모델 모두 약 0.076으로 비슷했다. RMSE와 R²는 Tuned CatBoost가 상대적으로 좋았다.
- 고득점자 기준: 10+/15+/20+ MAE와 음의 bias는 Tuned CatBoost가 MLP보다 작았다.
- CatBoost tuning 효과: 06의 고정 CatBoost MAE 2.389보다 약 0.009골 나빠져 Outer 일반화 성능을 개선하지 못했다.
- MLP tuning 효과: 06의 Baseline MLP MAE 2.428보다 약 0.036골 개선됐다.
- Hyperparameter 안정성: Inner에서 선택된 설정이 Outer 개선을 일관되게 만들지는 못했다. HPO의 일반화 실패 또는 선택 과적합 가능성이 있다.
- 추가 trial 필요 여부: 현재 결과만으로 trial 수를 늘릴 근거는 부족하다. 먼저 외부 피처 변경 후 동일한 현상이 반복되는지 확인한다.
- 다음 단계에서 유지할 모델: CatBoost와 MLP 모두 유지하되 튜닝 모델을 자동 채택하지 않고 고정 baseline을 함께 보존한다.
- 추가 확인이 필요한 점: 외부 피처가 추가된 공간에서 두 모델의 상대 순위와 고득점자 성능이 어떻게 변하는지 확인한다.
```

## 08-01. `08_01_transfermarkt_data_integration.ipynb`

**적용 위치:** `# 36. 결과 해석 — 직접 작성` 셀의 `## 내 결론` 이하

**적용 방식:** 기존 데이터 품질 항목명을 유지해 교체

```markdown
## 내 결론

- Player ID 연결 품질: unique player 기준 90.12%, player-season row 기준 92.51%가 연결됐다.
- Transfer event 품질: summer event의 exact-date coverage는 22.83%로 낮았다. cutoff 이후 544건과 날짜 미상 4,135건을 모델 입력에서 제외했다.
- Prediction cutoff 품질: actual-game cutoff와 fallback을 분리했으며 cutoff 이후 사건은 feature에 사용하지 않는 원칙을 적용했다.
- Market value 품질: 전체 coverage는 68.16%, 2005+는 80.71%였다. 장기 전 구간보다 2005년 이후 실험에 더 적합하다.
- New team environment 품질: changed-Big5 선수의 새 팀 strength coverage는 58.46%로 추가 개선이 필요하다.
- 수동 보정 필요 여부: 선수 crosswalk, 리그 표기, transfer date 연결과 대표 이적 사례를 추가 점검해야 한다.
- 08-02 진행 가능 여부: 최초 통합 구조는 확인했지만 날짜 coverage와 새 팀 strength 문제를 먼저 보완한다.
- 추가로 확인할 문제: 후속 모델링에는 leakage-safe 규칙을 강화한 v2 snapshot을 사용한다.
```

## 08-01 v2. `08_01_transfermarkt_data_integration_v2.ipynb`

**적용 위치:** `# 36. v2 결과 해석 — 직접 작성` 셀의 `## 내 결론` 이하

**적용 방식:** 기존 항목명을 유지해 교체

```markdown
## 내 결론

- Player ID 품질: unique player 90.12%, row 92.51%로 v1 수준을 유지했다.
- Transfer date coverage: 22.83%에서 27.86%로 개선됐다. 최근 시즌일수록 coverage가 높고 2024/2025는 약 80% 수준이었다.
- Transfer timeline 품질: cutoff 이전에 날짜가 확인된 사건만 feature로 사용하고 post-cutoff 552건과 날짜 미상 4,090건은 audit 전용으로 분리했다.
- League canonicalization: 12,992개의 리그 표기를 정규화해 `Laliga`와 `La Liga` 같은 표기 차이를 통합했다.
- New-team strength coverage: changed-Big5 선수 기준 58.46%에서 88.97%로 개선됐다.
- Market value 품질: 전체 68.16%, 2005+ 80.71%로 v1 수준을 유지했다.
- Leakage check: cutoff 이후 사건과 날짜 미상 사건이 모델 피처에 들어오지 않도록 assertion과 감사 테이블을 분리했다.
- 08-02 진행 가능 여부: Yes. v2 preseason snapshot을 외부 피처 실험의 정식 입력으로 채택한다.
- 추가 보정 필요 항목: unmatched player와 cross-team alias 후보는 자동 fuzzy match하지 않고 감사 대상으로 유지한다.
```

## 08-02. `08_02_external_feature_ablation.ipynb`

**적용 위치:** `# 30. 결과 해석 — 직접 작성` 셀

**적용 방식:** 기존 A/B/C와 `### 내 해석`, `### 내 최종 결론` 제목은 유지하고 각 빈 항목만 아래 내용으로 채운다.

```markdown
### 내 해석

- Market Value 효과: Market 2005+에서 M1 MAE 2.396은 M0 2.385보다 나빠 전체 MAE 개선은 확인되지 않았다. 다만 10+/20+ MAE와 RMSE/R²는 일부 개선됐다.
- Market Momentum 추가 효과: M2 MAE 2.396으로 M1과 거의 같아 추가적인 전체 MAE 이득은 없었다.
- 안정성: 시장가치 피처의 효과는 지표에 따라 달라 일관된 전체 성능 개선으로 보기 어렵다.
- 유지할 feature: 시장가치는 단독 채택하지 않고 2017+ Combined 구간에서 transfer context와 함께 재평가한다.
```

```markdown
### 내 해석

- Transfer State: T1 MAE 2.398은 T0 2.399와 거의 같아 단독 효과가 작았다.
- Transfer Economics: T2 MAE 2.400으로 개선되지 않아 이적료·경제성 피처의 추가 가치는 확인되지 않았다.
- New-Team Environment: T3 MAE 2.393으로 transfer 실험군 중 가장 좋았다.
- 실제 이적 선수 성능: changed-team MAE는 T0 2.757에서 T3 2.741로 소폭 개선됐다.
- 유지할 feature: transfer state와 new-team environment는 유지하고 transfer economics는 제외 후보로 둔다.
```

```markdown
### 내 최종 결론

- 전체 성능 기준 best: C3 All External, MAE 2.388
- 고득점자 기준 best: 전체 외부 피처가 일관된 해결책은 아니며 강한 과소예측이 유지됐다.
- 이적 선수 기준 best: C3의 changed-team MAE 2.731
- 최종 유지할 Market feature: market value와 momentum 후보를 유지하되 단독 효과가 작음을 명시한다.
- 최종 유지할 Transfer feature: transfer state
- 최종 유지할 New-Team feature: new-team strength와 team-change context
- 재-HPO 필요 여부: feature set을 먼저 고정한 후 필요한 모델만 재검토한다.
- CatBoost / MLP 재검증 필요 여부: Yes. 외부 피처의 효과가 모델 계열에 따라 다른지 확인한다.
- 다음 단계: economics를 제외한 선택 조합 S3를 S0/S4와 동일 Fold에서 검증한다.
```

## 08-03. `08_03_selected_external_feature_validation.ipynb`

**적용 위치:** `# 22. 결과 해석 — 직접 작성` 셀

**적용 방식:** 기존 A/B/C 구조를 유지하고 두 개의 `### 내 결론` 빈칸을 교체한다. `## C. 다음 결정` 마지막에는 결정 문단을 추가한다.

```markdown
### 내 결론

- S3 vs S0: S3 MAE 2.383으로 S0 2.399보다 약 0.016골 개선됐고 4개 중 3개 Fold에서 앞섰다.
- S3 vs S4: S4 MAE 2.385보다 S3가 약 0.003골 좋았고 4개 중 3개 Fold에서 앞섰다. 차이는 매우 작다.
- Market signal: 단독 개선은 작지만 combined context의 일부로 유지할 근거가 있다.
- Transfer/New-Team signal: S3의 changed-team MAE는 S0보다 약 0.035골 개선됐다.
- Transfer Economics: 추가해도 평균 MAE가 개선되지 않아 제외 결정을 지지한다.
- 최종 Selected Feature Set: S3 `selected_no_economics`
- Feature 조합 탐색 종료 여부: 종료한다. 추가 조합 탐색보다 모델 재검증으로 이동한다.
```

```markdown
### 내 결론

- CatBoost S3: MAE 2.383으로 CatBoost S0보다 개선됐다.
- MLP S3: MAE 2.451로 MLP S0 2.374보다 약 0.077골 나빠졌다.
- 전체 성능 기준: S3 공간에서는 Fixed CatBoost가 우세하다.
- 고득점자 기준: MLP S3가 일부 10+/20+ slice를 개선했지만 전체 MAE와 안정성이 악화됐다.
- 이적 선수 기준: CatBoost S3 changed-team MAE 2.722, MLP S3 2.813으로 CatBoost가 좋았다.
- 현재 최종 후보 모델: Fixed CatBoost S3
- 재-HPO 필요 여부: 선택 피처 공간이 바뀌었으므로 마지막으로 CatBoost와 MLP를 동일 공간에서 재튜닝하되 Fixed CatBoost S3보다 좋아질 때만 채택한다.
```

```markdown
### 최종 결정

Selected External은 평균 개선 폭이 약 0.016골로 크지는 않지만 3개 Fold, changed-team slice, economics 제외 비교에서 같은 방향을 보였다. S3를 최종 feature set으로 고정하고 피처 조합 탐색을 종료한다. 고득점자 문제는 별도 objective 문제로 분리한다.
```

## 08-04. `08_04_final_hpo_selected_features.ipynb`

**적용 위치:** `# 28. 결과 해석 — 직접 작성` 셀

**적용 방식:** 기존 A/B/C 구조와 항목명을 유지해 교체

```markdown
### 내 결론

- CatBoost HPO 효과: Tuned CatBoost S3 MAE 2.399로 Fixed CatBoost S3 2.383보다 나빠졌다.
- 평균 MAE: 2.399
- 안정성: MAE 표준편차 0.074, 최악 Fold 2.464로 고정 S3를 넘어서는 안정성 개선은 없었다.
- 고득점자: 10+/15+/20+ MAE가 고정 S3보다 모두 나빠졌다.
- 이적 선수: changed-team MAE 2.763으로 고정 S3 2.722보다 나빠졌다.
- Hyperparameter 안정성: HPO 결과가 Outer 개선으로 이어지지 않아 튜닝 모델을 채택하지 않는다.
```

```markdown
### 내 결론

- MLP retuning 효과: Frozen MLP S3 MAE 2.451에서 2.401로 회복됐다.
- 전체 MAE: 기존 MLP S0 2.374와 Fixed CatBoost S3 2.383에는 미치지 못했다.
- 고득점자: 10+/15+/20+ MAE는 Tuned CatBoost보다 낮아 보조적인 장점이 있다.
- Bias: -0.013으로 전체 bias는 0에 가깝다.
- 학습 안정성: 전체 MAE와 고득점 성능을 동시에 최선으로 만들지는 못했다.
- Hyperparameter 안정성: 재튜닝으로 회복은 있었지만 최종 주 모델을 바꿀 정도의 일반화 개선은 아니었다.
```

```markdown
### 최종 결론

- 전체 MAE 1위: Fixed CatBoost S3, 2.383
- RMSE/R² 1위: Fixed CatBoost S3
- 고득점자 1위: Tuned MLP S3가 일부 slice에서 상대적으로 좋지만 전체 예측 성능과 trade-off가 있다.
- 이적 선수 1위: Fixed CatBoost S3
- 안정성 1위: Fixed CatBoost S3
- 최종 Model B 후보: Fixed CatBoost S3
- 보조 Model 후보: Tuned MLP S3는 고득점 진단용 후보로만 유지한다.
- 추가 HPO 필요 여부: 없음
- Feature 실험 종료 여부: 종료하고 Model A/Model B 구조로 이동한다.
```

## 08-04 Colab. `08_04_final_hpo_selected_features_colab.ipynb`

**적용 위치:** 노트북 첫 설명 셀과 `# 28. 결과 해석 — 직접 작성` 셀

**적용 방식:** 첫 설명 셀에 아래 상태 문구를 추가하고, 결과 해석 셀은 위 08-04와 동일한 A/B/C 결론을 사용한다.

```markdown
> 상태: 이 노트북은 08-04 실험의 Colab GPU 실행 변형이다. 모델 선택은 실행환경이 아니라 동일한 `08_04_final_hpo_*` 산출물을 기준으로 하며 최종 결론은 로컬 08-04와 동일하다.
```

## 09-01. `09_01_model_a_big5_presence_classifier.ipynb`

**적용 위치:** `# 27. 결과 해석 — 직접 작성` 셀

**적용 방식:** 기존 A~D의 `### 내 해석`과 마지막 `# 최종 결정` 빈칸을 교체

```markdown
### 내 해석

- Presence 비율: 시즌별 약 78.7~86.9%로 다수 클래스다.
- Exit 비율: 시즌별 약 13.1~21.3%로 소수 클래스다.
- Accuracy 주의점: 모든 선수를 presence로 예측해도 약 79~87% Accuracy가 나오지만 exit recall은 0이 된다. 따라서 Accuracy 단독으로 Model A를 평가할 수 없다.
```

```markdown
### 내 해석

- Probability 기준 best: CatBoost. LogLoss 0.258, Brier 0.073, ROC-AUC 0.911, Exit PR-AUC 0.766으로 Logistic보다 전반적으로 좋았다.
- Exit 탐지 기준 best: CatBoost. 평균 exit recall 0.708, exit F1 0.702, Macro F1 0.821을 기록했다.
- Threshold 안정성: Fold별 선택 threshold의 평균은 약 0.68이었다. 고정 0.5가 아니라 각 Outer Train 내부 마지막 시즌에서 선택했다.
- Fold 안정성: CatBoost가 확률 품질과 분류 품질 모두에서 Logistic보다 우세했다.
- Model A 최종 후보: CatBoost classifier
```

```markdown
### 내 해석

- Calibration 품질: 중간·높은 확률 구간은 대체로 관측 presence rate와 가까웠다. 가장 낮은 두 구간에서는 예측 0.232/0.610 대비 관측 0.157/0.552로 presence 확률을 다소 높게 추정했다.
- 추가 calibration 필요 여부: 즉시 별도 calibration을 채택할 정도의 일관된 왜곡은 확인되지 않았다. Two-stage에서 Soft/Hard를 모두 비교해 확률 사용 방식을 검증한다.
```

```markdown
### 내 해석

- Transfer signal: preseason transfer와 destination 정보는 exit 가능성을 구분하는 Model A 전용 context로 유지한다.
- Market signal: 시장가치는 선수의 Big5 잔류 가능성을 설명하는 보조 신호로 사용한다.
- 어려운 exit 유형: 명확한 preseason exit 정보가 없거나 cutoff 이후에 상태가 변한 선수다.
- 어려운 presence 유형: 실제로는 Big5에 남지만 cutoff 시점의 transfer context가 불완전하거나 원본 매칭이 실패한 선수다.
```

```markdown
# 최종 결정

- Model A 최종 모델: Fixed CatBoost classifier
- Model A threshold: Fold 내부에서 Macro F1을 최대화하도록 선택하며 평균은 약 0.68
- Soft probability 사용 가능 여부: 후보로 유지하되 09-02에서 Hard gate 및 Direct Regression과 비교한다.
- Probability calibration 필요 여부: 현재 단계에서는 추가하지 않는다.
- 다음 단계: `matched_next=False`에 포함된 매칭 오류 가능성을 감사한 후 audited label로 Model A와 Model B를 재검증한다.
```

## 09-01A. `09_01A_matched_next_label_audit.ipynb`

**적용 위치:** `# Part L. 결과 해석 체크리스트`와 산출물 안내 사이

**적용 방식:** 아래 새 Markdown 셀 추가

```markdown
# Part M. 최종 결론

- `matched_next=False`에는 실제 Big5 이탈뿐 아니라 동명이인, 팀명 alias, season-to-season 매칭 실패가 섞여 있었다.
- 전체 false를 자동 수정하지 않고 증거가 명백한 HARD_CORRECTION 20건만 복원했다.
- 보정으로 95골과 10골 이상 사례 3건이 평가 모집단에 돌아왔다.
- False 사례 중 STABLE_BIG5_REVIEW는 836건, LATE_OR_UNKNOWN_TRANSFER_CONTEXT는 200건, PRESEASON_CONFIRMED_EXIT는 139건이었다.
- STABLE_BIG5_REVIEW와 날짜가 늦거나 불명확한 사례는 실제 잔류, 부상, 후반 이적, 데이터 누락 등이 섞일 수 있어 자동 보정하지 않는다.
- 20건의 보수적 audited label을 별도 컬럼에 저장하고 원본 snapshot은 덮어쓰지 않는다.
- 다음 단계에서는 원본과 audited label을 동일한 Walk-forward 프로토콜로 비교한다.
```

## 09-01B. `09_01B_audited_label_revalidation.ipynb`

**적용 위치:** 마지막 `# 실행 후 보내줄 파일` 셀 바로 앞

**적용 방식:** 아래 새 Markdown 셀 추가

```markdown
# Part M. 최종 결론

- Audited Model A는 원본 대비 LogLoss 0.254→0.236, Brier 0.072→0.067, ROC-AUC 0.912→0.924, Macro F1 0.820→0.833으로 개선됐다.
- Audited Model B의 MAE는 2.388→2.397로 소폭 나빠졌다. 어려운 득점 사례가 평가에 복원되면서 기존 점수가 다소 낙관적이었음을 의미할 수 있다.
- 데이터 감사의 목적은 점수 향상이 아니라 정답 품질 개선이므로 Model B의 작은 성능 악화를 이유로 보정을 되돌리지 않는다.
- 보수적으로 수정한 audited label을 09-02의 정식 개발 타깃으로 채택한다.
- Model A는 audited CatBoost classifier, Model B는 audited population의 Fixed CatBoost S3를 사용한다.
```

## 09-02. `09_02_two_stage_integration.ipynb`

**적용 위치:** 기존 `# 결과 해석 체크리스트`의 `## D. High Scorer` 다음, `# 실행 후 보내줄 파일` 앞

**적용 방식:** 아래 새 Markdown 내용을 같은 셀에 삽입하거나 별도 셀로 추가

```markdown
# 최종 결론

- Hard Two-stage 평균 MAE는 2.127로 Soft 2.168, Direct Regression 2.202, Conditional-No-Gate 2.385보다 낮았다.
- Conditional-No-Gate 대비 Hard gate는 전체 MAE를 약 0.259골, exit MAE를 약 1.464골 줄였다. Presence MAE도 약 0.016골 낮았다.
- 실제 exit 616명의 MAE는 Hard 0.832, Soft 1.205, Direct 1.405, No-Gate 2.319로 Hard gate가 가장 좋았다.
- 실제 presence 3,139명에서는 Soft MAE 2.356이 가장 낮았고 Hard는 2.380이었다. Hard의 전체 우위는 주로 exit 처리에서 발생했다.
- Soft gate는 10+/20+ 예측을 더 축소해 기존 고득점자 과소예측을 악화시켰다.
- 이 데이터와 Walk-forward 프로토콜에서 전체 MAE를 우선할 경우 Hard Two-stage를 최종 구조로 채택한다. RMSE/R² 및 presence slice까지 모든 지표에서 우월하다는 뜻은 아니다.
- Direct Regression은 최종 테스트에서도 유지할 진단용 경쟁 baseline으로 남긴다.
```

## 10. `10_selected_feature_stability_check.ipynb`

**적용 위치:** `# Part J. 최종 자동 판정` 결과 다음

**적용 방식:** 아래 새 Markdown 셀 추가

```markdown
# Part K. 최종 해석

- FULL Hard Two-stage의 평균 MAE는 2.127이었다.
- NO_MARKET은 2.142, NO_MODEL_A_CONTEXT는 2.139, NO_TRANSFER_CONTEXT는 2.211로 어떤 제거안도 FULL을 개선하지 못했다.
- 특히 transfer context 제거의 악화 폭이 약 0.085골로 가장 커, 이적과 새 팀 환경 정보가 구조적으로 중요한 신호임을 확인했다.
- 어떤 ablation도 사전 정의한 ‘평균 MAE 0.03 이상 개선 및 4개 중 3개 Fold 개선’ 조건을 만족하지 못했다.
- 자동 판정 `KEEP_FULL`을 수용한다.
- 추가 피처 탐색과 HPO를 종료하고 모델·피처·threshold를 동결한 뒤 Final Test로 이동한다.
```

## 11. `11_final_test_slice_error_analysis_v3.ipynb`

**적용 위치:** `# Part V. 저장` 실행 결과 다음, 마지막 산출물 안내 셀 앞

**적용 방식:** 아래 새 Markdown 셀 추가

```markdown
# Part W. 최종 결론

- 2024-25→2025-26 locked holdout 926명에서 Final Hard Two-stage는 MAE 2.003, RMSE 3.099, R² 0.406, bias +0.327을 기록했다.
- 전체 MAE는 Direct Regression 2.067과 Conditional-No-Gate 2.259보다 낮았다. 반면 Direct Regression은 RMSE 3.084와 R² 0.412로 근소하게 앞서 모델 우위는 목적 지표에 따라 다르다.
- Model A는 ROC-AUC 0.907, Macro F1 0.814, exit recall 0.680, presence recall 0.936으로 개발 구간의 분류력을 대체로 유지했다.
- 실제 exit 선수 MAE는 0.905로 gate의 효과가 확인됐지만 실제 presence 선수 MAE는 2.259였다.
- FW MAE 2.653은 MF 1.269보다 높았고 Premier League MAE 2.450도 상대적으로 높았다.
- 가장 큰 한계는 고득점자 과소예측이다. 10+ 선수 bias는 -5.295골, 20+ 선수 bias는 -9.954골이었다.
- 개발 데이터에는 09-01A audited label을 사용했지만 Final Test에는 사후 보정하지 않은 원본 locked `test.csv` 라벨을 사용했다. 따라서 최종 수치는 원본 test 라벨 기준이며 남아 있을 수 있는 매칭 오류의 영향도 포함할 수 있다.
- 최종 모델은 전체 선수의 평균 절대오차를 줄이는 데 성공했지만 고득점자 발굴용 모델로는 충분하지 않다.
- Final Test를 모델 선택에 재사용하지 않는다. 후속 고득점자·FW 연구는 새로운 개발 프로토콜과 별도 평가 구간에서 수행한다.
```

## 11 v1/v2 상태 표시

**적용 위치:** `11_final_test_slice_error_analysis.ipynb`, `11_final_test_slice_error_analysis_v2.ipynb`의 첫 번째 설명 셀

**적용 방식:** 최종 결론을 삽입하지 않고 아래 상태 배너만 추가

```markdown
> 상태: 이 노트북은 Final Test 파이프라인의 준비·오류 수정 과정에서 생성된 중간 구현이며 최종 결과 보고에는 사용하지 않는다. 행 정합성 수정이 반영된 최종 실행본은 `11_final_test_slice_error_analysis_v3.ipynb`이다.
```

### v1→v3 변경 이력

- v2: datetime 해상도 정규화와 cutoff 누수 검사 강화
- v3: 정렬 이후 이전 행 위치를 가리킬 수 있던 stale boolean mask 위험 수정
- 최종 지표: 현재 `row_id`와 `season` 기준으로 mask를 다시 계산한 v3에서 산출
- 변경 성격: test label에 맞춘 모델·피처·threshold 재선택이 아니라 데이터 행 정합성과 실행 안정성 수정

## 결론 작성 시 공통 주의사항

- 통계 검정 없이 `통계적으로 유의미하다`고 쓰지 않는다.
- 단일 Validation 결과를 일반화 성능으로 표현하지 않는다.
- 0.01골 수준의 차이를 근거 없이 ‘압도적 개선’이라고 부르지 않는다.
- HPO의 Inner score를 최종 성능으로 보고하지 않는다.
- Final Test 결과를 본 뒤 feature, model, threshold를 다시 선택하지 않는다.
- 고득점 slice의 작은 표본 수를 무시하고 전체 모집단으로 일반화하지 않는다.
- 개발 audited label과 Final Test 원본 라벨의 차이를 숨기지 않는다.
- “Hard Two-stage가 가장 좋다” 대신 “사전 지정한 전체 MAE 기준에서는 Hard Two-stage를 채택하지만 다른 지표에서는 경쟁 모델이 앞설 수 있다”고 범위를 명시한다.
