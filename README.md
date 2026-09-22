# QCache Lab

**QCache Lab** は、HuggingFace Transformers を用いた **Query Cache 注意機構介入実験フレームワーク** です。

- **目的**: 注意機構の出力に対して、過去のクエリの読み出しを介入し、言語モデルの生成挙動を分析
- **特徴**: Teacher-forcing 無しの完全自由生成、状態フォーク、チェックポイント、繰り返し検出
- **対応モデル**: Llama / Mistral / Qwen2 (Transformers == 4.57.6)

---

## 💡 QCache とは？

**Query Cache (QCache)** は、注意機構の出力に対する介入技術です。

通常の注意機構 (Attention) では、各トークンごとに新しいクエリ Q が生成され、キー K とバリュー V に対して注意スコアを計算します。QCache では、**過去の全ての Q を保存し、その K,V に対する読み出し (readout) をキャッシュ**します。然る後、注意機構の出力を以下の式で修正します：

```
Replay モード (λ > 0):
  output = (1-λ) * attention_output + λ * mean(past_Q_readouts)

Attenuation モード (λ > 0):
  output = (1-λ) * attention_output
```

**なぜ QCache と呼ぶか？**
- **Query** (Q) を **Cache** (キャッシュ) するから
- 過去の Q を保持し、その読み出しを用いて注意機構を補正する
- 「記憶」や「文脈」を明示的に操作可能にする技術

---

## 🚀 CLI 使い方

### 全体構造

```
QCache Lab
├── V0.1.2 (legacy)  - 固定トークン比較 + 基本介入
└── V1.2+ (live)     - 自由軌跡 + 状態フォーク + 各種分岐モード
```

---

### 🔧 共通前提

```bash
# 必須依存
pip install torch transformers==4.57.6 safetensors sentencepiece

# スクリプト実行 (python3 使用)
python3 sr_qcache_lab_v14.py <command> [options]
```

---

### 📋 コマンド一覧

| コマンド | 説明 | 出力 | 実行時間 |
|---|---|---|---|
| `device-check` | デバイス・メトリクス境界のチェック | JSON | ~1秒 |
| `self-test` | PyTorch 核 + Toy モデルテスト | JSON | ~10秒 |
| `hf-smoke` | 実 HF tiny モデルアダプタテスト | JSON | ~30秒 |
| `run` (V1.2+) | **自由軌跡実験 + フォーク** | ディレクトリ | 数分~ |
| `legacy-run` (V0.1.2) | 固定トークン比較 (レガシー) | ディレクトリ | 数分~ |

---

### ✅ 環境確認系

#### 1. `device-check` — デバイス動作確認
```bash
python3 sr_qcache_lab_v14.py device-check --out qcache_device_check.json
```
- **目的**: CPU メトリクス計算が正しく動作するかを確認
- **モデル不要**

#### 2. `self-test` — 内部ロジックテスト
```bash
python3 sr_qcache_lab_v14.py self-test --out qcache_selftest.json
```
- **目的**: PyTorch 数学核 + Toy デコーダーの正当性チェック
- **モデル不要**
- **78テスト全部通過** (2026-09-23 確認)

#### 3. `hf-smoke` — HF アダプタ動作確認
```bash
python3 sr_qcache_lab_v14.py hf-smoke --out qcache_hf_smoke.json --device cpu
```
- **目的**: 実 HF ライブラリ (Llama/Mistral/Qwen2) との統合をランダム重みでテスト
- **モデルダウンロード不要** (ランダム初期化)

---

### 🔬 実験系 (V1.2+)

#### 4. `run` — 自由軌跡 + フォーク実験
```bash
python3 sr_qcache_lab_v14.py run \
  --model /path/to/llama-7b \
  --prompt "Tell me a short story." \
  --lambdas "0,0.2,0.55" \
  --branches "continue,cut,native" \
  --max-new-tokens 64 \
  --out my_experiment
```

**分岐モード:**
- `continue`: 通常の replay 介入継続
- `cut`: attenuation (出力を (1-λ)倍に弱める)
- `native`: ベースライン (介入なし)
- `parallel`: 並列ノルム正規化
- `freeze`: Q 入力停止 (読み出しは継続)
- `pulse_cut_N`: N ステップまで cut、以降 replay

**出力:**
- `reading_room.html` — ブラウザ閲覧用
- `branches.md` — Markdown レポート
- `results.json` — 全データ
- `share_packet.json` — 共有用パケット
- `checkpoints/` — 状態スナップショット

---

### 🏁 レガシーモード (V0.1.2)

```bash
python3 sr_qcache_lab_v14.py legacy-run \
  --model /path/to/model \
  --prompt "Short story about a key." \
  --lambdas "0,0.05,0.1,0.2" \
  --families "replay,attenuation" \
  --out results_qcache
```

- 固定 continuation を使用 (Toy モデル以外)
- 固定トークン比較に特化

---

## 📝 介入プロトコル

```
QCache: 注意機構出力前 (W_O 手前) での介入

Baseline (λ = 0):
  output = attention_output

Replay (λ > 0):
  history = mean(past_Q_readouts)  [過去の Q の読み出し平均]
  output = (1-λ) * attention_output + λ * history

Attenuation (λ > 0):
  output = (1-λ) * attention_output  (Q bank は使わない)
```

**特性:**
- No KV rewrite: K,V 自体は書き換えない
- Append-only: K,V は追記のみ
- Current Q excluded: 現在の Q は history 平均から除外
- Fresh state per trial: 各実験は独立

---

## 🔍 実動確認状況 (2026-09-23)

| コマンド | 状況 | 備考 |
|---|---|---|
| `device-check` | ✅ PASS | CPU/MPS 環境で正常 |
| `self-test` | ✅ 78/78 PASS | 全 Toy テスト通過 |
| `hf-smoke` | ⚠️ PARTIAL | Llama/Mistral/Qwen2 ✅, Gemma2 ❌ |

> Gemma2の失敗は環境 (Sprialton + MPS) 特有のデバイス配置問題の可能性

---

## 📚 参考

- **Transformers**: 4.57.6 (厳密)
- **Python**: 3.9+
- **PyTorch**: 2.0+
- **ライセンス**: LICENSE を参照
