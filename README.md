# topo-agent

Mirai Climbing のトポエージェント。ボルダリングジムの壁の写真で、課題のホールドを1つタップすると、
同じ課題のホールドを探して輪郭で囲む。

## このリポジトリの区切り

- **`topo_agent/baseline/`:ハッカソンより前に作った物**(mirai-climbing-v2 から持ち込んだ。最初のコミット)。
  色で輪郭を取る処理、Gemini のエージェント、難しい見本の壁。比べる相手として残している
- それ以外(このあとのコミット):Meta Global AI Developer Hackathon(2026-10-27〜11-04)で新しく作る物

## 測る

```
python -m topo_agent.eval --bench --methods color
```

## ライセンス

AGPL-3.0-or-later(COPYING)
