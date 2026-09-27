# 한국어 학습 데이터 만들기

convmerge로 흩어져 있는 한국어 데이터셋을 모아 하나의 학습 파일로 만드는 방법입니다.
명령과 옵션은 영어 문서([README](../../README.md), [format.md](../format.md),
[recipes.md](../recipes.md))와 같습니다. 이 문서에서는 한국어 데이터에서 자주
생기는 문제만 다룹니다.

## 검증된 한국어 데이터셋

아래 데이터셋은 테스트에서 변환하고, 매주 도는 `Datasets` 워크플로가 Hugging Face의
실제 행으로 다시 확인합니다([README의 표](../../README.md#tested-datasets)에서
`ko`로 표시됨).

| 데이터셋 | 형태 | 라이선스(카드 기준) |
|---|---|---|
| `beomi/KoAlpaca-v1.1a` | Alpaca | 카드 확인 |
| `nlpai-lab/kullm-v2` | Alpaca (GPT4ALL·Dolly·Vicuna를 DeepL로 번역) | apache-2.0 |
| `kyujinpy/KOR-OpenOrca-Platypus-v3` | Alpaca | 카드 확인 |
| `maywell/koVast` | ShareGPT 멀티턴 | mit |
| `heegyu/open-korean-instructions` | `text`에 `<usr>` / `<bot>` / `<sys>` | mit |
| `FreedomIntelligence/sharegpt-korean` | ShareGPT (JSON 배열) | 카드 확인 |
| `maywell/ko_Ultrafeedback_binarized` | 선호도(prompt / chosen / rejected 문자열) | 카드 본문: **데이터 자체의 상업적 사용 불가** |
| `kuotient/orca-math-korean-dpo-pairs` | 선호도(system / question / chosen / rejected) | cc-by-sa-4.0 |

모두 `--from auto`로 읽습니다. `open-korean-instructions`의 `<sys>`는 위치에 따라
뜻이 다릅니다. 맨 앞의 `<sys>`는 시스템 프롬프트(KorQuAD-Chat의 문서)가 되고,
사용자 발화 바로 뒤의 `<sys>`는 그 사용자 턴의 입력(KoAlpaca의 input)으로 합쳐집니다.

## 레시피로 한 번에 모으기

```yaml
# recipe.yaml
output: train/ko_sft.jsonl
sources:
  kullm:
    fetch: {hf: nlpai-lab/kullm-v2, max_rows: 20000}
    convert: {from: auto, meta: {dataset: kullm-v2}}
  kovast:
    fetch: {hf: maywell/koVast, max_rows: 20000}
    convert: {from: auto, meta: {dataset: koVast}}
  oki:
    fetch: {hf: heegyu/open-korean-instructions, max_rows: 20000}
    convert: {from: auto, meta: {dataset: open-korean-instructions}}
mix:
  weights: {kullm: 0.4, kovast: 0.4, oki: 0.2}
dedupe: true
tokens:
  tokenizer: Qwen/Qwen3-0.6B     # 학습할 모델의 토크나이저
  max_tokens: 4096
split:
  val: 0.02
```

```bash
pip install "convmerge[all]"
convmerge run recipe.yaml
```

- `meta: {dataset: ...}`: 모든 행의 `meta`에 출처를 남겨서, 섞은 뒤에도 어느
  데이터셋에서 온 행인지 알 수 있게 합니다.
- `tokens`: 한국어는 토크나이저마다 토큰 수 차이가 큽니다. 같은 문장이라도 모델에 따라
  몇 배까지 차이가 날 수 있으니, 반드시 **학습할 모델의 토크나이저**로 길이를 재세요.
  `max_tokens`를 넘는 행과 채팅 템플릿이 거부하는 행은 여기서 걸러집니다.

## 라이선스 확인

한국어 공개 데이터 중에는 번역 데이터가 많아서, 원본 데이터와 번역에 쓴 도구 양쪽의
약관이 함께 적용되는 경우가 많습니다. convmerge는 라이선스를 법적으로 판단하지 않습니다.
대신 섞은 결과에 무엇이 들어갔는지 보여줍니다.

- Hugging Face 소스는 데이터셋 카드의 `license`를 자동으로 읽습니다.
- 카드에 라이선스가 없거나 본문에만 조건이 적혀 있으면(예:
  `ko_Ultrafeedback_binarized`) 레시피에 직접 적어 주세요.

  ```yaml
  sources:
    ko_uf:
      fetch: {hf: maywell/ko_Ultrafeedback_binarized}
      license: "non-commercial (dataset card)"
      convert: {from: auto, format: preference}
  ```

- `convmerge run`이 끝나면 `build/report.json`의 `licenses`에 소스별 라이선스와
  행 수가 기록됩니다. 비상업·연구 전용·`other`·라이선스 없음은 `[license]` 경고로
  출력됩니다.

## AI Hub 같은 중첩 JSON

AI Hub 데이터는 과제마다 JSON 구조가 다릅니다. 스크립트를 새로 짜는 대신 `--from map`에
경로를 적으면 됩니다([format.md](../format.md#field-mapping---from-map)).

```yaml
sources:
  counsel:
    path: aihub/상담_대화/          # 내려받은 JSON 파일이 있는 폴더
    convert:
      map:
        turns: "dialogue[].utterances[]"
        role: speaker
        content: text
        role_map: {고객: user, 상담사: assistant}
        system: info.topic
```

- `a.b`는 키, `a[0]`은 목록의 한 항목, `a[]`는 목록 전체를 뜻합니다. `a[].b[]`처럼
  쓰면 중첩 목록이 펼쳐집니다.
- 경로가 맞지 않는 레코드는 `map_path_missing`으로 집계되고 버려집니다. 처음에는
  `--report`로 몇 행이 버려졌는지 확인하세요.
- AI Hub 데이터는 이용 약관상 재배포와 용도에 제한이 있습니다. 레시피에 `license`를
  적어 두면 리포트에 함께 남습니다.

## 번역 데이터를 섞을 때

- **번역투:** DeepL 등으로 번역한 데이터(KULLM v2, sharegpt_deepl_ko 계열)는 문장이
  어색하거나 영어 고유명사·단위가 그대로 남은 경우가 많습니다. 사람이 쓴 한국어
  데이터와 섞을 때는 `mix` 비율로 번역 데이터 비중을 조절하세요.
- **중복:** 같은 원본(Alpaca, ShareGPT)을 서로 다른 팀이 번역한 데이터셋이 많습니다.
  `dedupe: true`는 완전히 같은 행만 지우므로, 번역이 조금씩 다른 중복은 남습니다.
- **선호도 데이터:** 번역 과정에서 chosen과 rejected가 같아진 행은 `--format preference`
  변환 때 `unrepresentable_identical_pair`로 걸러집니다.
