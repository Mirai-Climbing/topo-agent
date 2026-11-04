# Submission draft

Draft of the hackathon submission form, for the team to review. Comment on the PR or suggest edits.
The `meta` real-photo numbers (60% recall, 68% precision) are the leaderboard run on commit `6bc820c`; the agent's default path has not changed since. Siho's final rerun on the merged `main` (`eec00e7`) reproduced `yolo3s-tapfix` (59% / 60% / 17.8 s vs 60% / 60% / 17.4 s), so we keep the leaderboard numbers. The video and X links are added when they exist.
No passwords, emails or keys go here or in the form. The organizers said there is no private field: judges try the project from the public repo (README "Try it (for judges)") and the video.

提出フォームの下書き。チームで見て、PR にコメントするか、直しを提案してください。
本物の写真の `meta` の数字(見つけた割合 60%、当たりの割合 68%)は、コミット `6bc820c` でのリーダーボードの測定。その後、エージェントのふだんの動きは変わっていない。Siho さんがマージ後の `main`(`eec00e7`)で `yolo3s-tapfix` を測り直し、同じ結果が出た(59% / 60% / 17.8秒、前は 60% / 60% / 17.4秒)ので、リーダーボードの数字を使う。動画と X の URL は、できてから入れる。パスワード、メール、鍵は、ここにもフォームにも書かない。主催者の答え:非公開の欄はない。審査員は公開リポジトリ(README の「Try it (for judges)」)と動画で試す。

## Choices (proposed)

| Field | Proposal | Why |
|---|---|---|
| Team Name | Mirai Climbing | |
| Team Members | Yuzuki, Siho, Ajay, Pao | |
| Track | Indie Developer Track | Confirmed by Yuzuki |
| Problem statement | Build for Your World | Built for climbers in our own gyms: anyone can add a problem with one tap, at any gym |
| Most Viral Track | X post link, once posted | |
| How judges try it | README "Try it (for judges)" | No login: replay recorded runs without a key, or run live with a Meta Model API key. The production app needs a login, so the video shows it |
| Partner technologies feedback | (blank) | We didn't use Fireworks, Modal or Baseten |

## Project Description

Climbing apps let climbers find a gym's problems, log their sends and share them. But first someone has to add each problem: take a photo of the wall and mark every one of its holds by hand. That takes time, and walls are reset again and again, so keeping up is hard.

topo-agent makes adding a problem one tap, so any climber at any gym can do it. Take a photo of the wall and tap one hold. SAM 3.1 outlines every hold on the wall, and a Muse Spark agent picks only the holds that belong to the same problem. The app shows what the agent did, the climber can add or remove holds, and the result is saved as a shared problem in Mirai Climbing, our climbing app, so nobody has to mark the same problem twice.

Because the problem is outlined, not just colored, it also helps climbers who can't tell hold colors apart: red-green color vision deficiency affects up to one man in twelve (men of Northern European descent; about one in twenty in Japan).

Results so far: on drawn test walls, 99% recall and 100% precision (tape problems: 14% with a color filter, 100% with the agent). On 68 routes in real gym photos, 60% recall and 68% precision; the agent still stops too early on dense walls, and every experiment is logged in the repo. We also trained a small YOLO model that replaces SAM: the same recall in about half the time, less precise for now, and it also runs offline on a phone.

Mirai Climbing existed before the hackathon. The topo agent, its evaluation and its integration into the app were built during the hackathon; pre-hackathon code is kept only as a baseline (tag `pre-hackathon`).

## Public GitHub Repository

https://github.com/Mirai-Climbing/topo-agent

## 3-Minute Demo Video

(YouTube link; script in [demo-video.md](demo-video.md))

## Describe your use of Meta products to build the submitted project

- **SAM 3.1** (Meta Model API, Responses): for each photo we send the concept prompts "climbing hold" and "climbing volume" plus a point prompt at the tapped spot, in parallel, with `store: false`. Masks are decoded with Meta's `meta-sam-parser` and turned into numbered outlines with color, size and position.
- **Muse Spark 1.3** (Chat Completions with tool calling, `reasoning_effort: low`): an agent that gets the photo, an image with every candidate numbered, a close-up of the tapped hold, and a candidate table. It chooses its own tools: `inspect_candidates`, `find_similar`, `look_closer` (zoomed crops as images, to spot tape) and `submit_holds` (the answer, with a one-sentence reason).
- Guardrails: limits on turns, tool calls and holds; text in the photo is ignored; only the photo and the tap are sent, never user data. If Muse Spark fails, the app falls back to SAM 3.1 plus a color filter.
- Each step is traced and shown in the app as "What the AI did".
- Try it from the repo without any login: `python -m topo_agent.demo` replays a recorded run on a sample wall, and `--live` runs SAM 3.1 + Muse Spark with your own key (README "Try it (for judges)").
- It also runs in production in the Mirai Climbing app (login required; shown in the video).

## Why are Meta's models important to your solution?

Picking the holds of one problem by color alone fails in the cases that matter: tape problems where every hold is a different color, the same color on the next wall, and shadows or highlights. Solving this needs two things. SAM 3.1 gives clean outlines of every hold from a simple text prompt and a tap, without training our own model. Muse Spark can look at the photo, call tools, zoom in, and reason about which holds belong together, like a climber would. On tape problems on our drawn test walls, this took recall from 14% with a color filter to 100%. Because the agent works with numbered SAM 3.1 candidates, it answers with ids instead of guessing coordinates, and every decision can be shown to the user.

## Survey questions

The form also asks about the team's experience with Meta Model API (ratings, biggest obstacle, comparison with other models). Siho submits the form as the team's Representative, using Yuzuki's answers to these (shared on Discord); tell Siho or Yuzuki if you want to add anything, for example the obstacles you hit.
