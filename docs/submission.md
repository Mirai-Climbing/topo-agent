# Submission draft

Draft of the hackathon submission form, for the team to review. Comment on the PR or suggest edits.
Numbers are updated on Nov 4 from the final 68-route rerun; the video and X links are added when they exist.
The reviewer password is never written here or in public fields; we ask the organizers how to hand reviewer access to the judges.

提出フォームの下書き。チームで見て、PR にコメントするか、直しを提案してください。
数字は 11/4 に最終の測り直しで直す。動画と X の URL は、できてから入れる。審査用のパスワードは、ここにも公開される欄にも書かない。審査員への渡し方は主催者に聞く。

## Choices (proposed)

| Field | Proposal | Why |
|---|---|---|
| Team Name | Mirai Climbing | |
| Team Members | Yuzuki, Siho, Ajay | |
| Track | Indie Developer Track | Startups Track only if Mirai Climbing is run as a company (Yuzuki to confirm) |
| Problem statement | Build for Your World | Built for climbers in our own gyms: anyone can add a problem with one tap, at any gym |
| Most Viral Track | X post link, once posted | |
| Reviewer access | Ask the organizers | The judges need the reviewer account for the production app; the password goes only where the organizers say, never in a public field |
| Partner technologies feedback | (blank) | We didn't use Fireworks, Modal or Baseten |

## Project Description

Climbing apps let climbers find a gym's problems, log their sends and share them. But first someone has to add each problem: take a photo of the wall and mark every one of its holds by hand. That takes time, and walls are reset again and again, so keeping up is hard.

topo-agent makes adding a problem one tap, so any climber at any gym can do it. Take a photo of the wall and tap one hold. SAM 3.1 outlines every hold on the wall, and a Muse Spark agent picks only the holds that belong to the same problem. The app shows what the agent did, the climber can add or remove holds, and the result is saved as a shared problem in Mirai Climbing, our climbing app, so nobody has to mark the same problem twice.

Because the problem is outlined, not just colored, it also helps climbers who can't tell hold colors apart: red-green color vision deficiency affects up to one man in twelve (about one in twenty in Japan).

Results so far: on drawn test walls, 99% recall and 100% precision (tape problems: 14% with a color filter, 100% with the agent). On 68 routes in real gym photos, 60% recall and 68% precision; the agent still stops too early on dense walls, and every experiment is logged in the repo.

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
- It runs in production in the Mirai Climbing app (https://api.mirai-climbing.com/app/) for the hackathon reviewer account.

## Why are Meta's models important to your solution?

Picking the holds of one problem by color alone fails in the cases that matter: tape problems where every hold is a different color, the same color on the next wall, and shadows or highlights. Solving this needs two things. SAM 3.1 gives clean outlines of every hold from a simple text prompt and a tap, without training our own model. Muse Spark can look at the photo, call tools, zoom in, and reason about which holds belong together, like a climber would. On tape problems on our drawn test walls, this took recall from 14% with a color filter to 100%. Because the agent works with numbered SAM 3.1 candidates, it answers with ids instead of guessing coordinates, and every decision can be shown to the user.

## Survey questions

The form also asks about the team's experience with Meta Model API (ratings, biggest obstacle, comparison with other models). Yuzuki answers these when submitting; tell Yuzuki if you want to add anything, for example the obstacles you hit.
