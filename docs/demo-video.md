# Demo video script (3 minutes)

Narration in English (subtitles if spoken narration is hard). Japanese notes are for the team.
日本語は、チームのためのメモ。読み上げるのは英語(話すのが難しければ字幕にする)。

| Time | Picture | Narration |
|---|---|---|
| 0:00–0:15 | `docs/images/colorblind.png`, one panel at a time | "In a bouldering gym, a problem is its color. But for about one man in twenty in Japan, red and orange look almost the same. Which holds are mine?" |
| 0:15–0:30 | Climbing in the gym, then the app's home screen | "I'm Yuzuki. I climb, and I build Mirai Climbing, an app for climbers. For this hackathon we built a topo agent with Meta's models." |
| 0:30–1:15 | Phone screen recording: take a photo of the wall → Find with AI → tap one hold → "What the AI did" → holds outlined → fix one hold → save as a problem | "Take a photo, tap one hold. SAM 3.1 outlines every hold on the wall. Then a Muse Spark agent decides which ones belong to the same problem, and shows what it did. I can still add or remove a hold. Saved, it becomes a problem other climbers can find and log." |
| 1:15–1:50 | Tape-problem wall: the close-up, look_closer crops, the agent's note | "Some problems are marked with tape, and the holds are all different colors. A color filter finds one hold out of seven. The agent looks at the close-up first, sees the tape, zooms in on the candidates, and finds all seven." |
| 1:50–2:20 | README results tables: drawn walls, then real photos | "On our drawn test walls, the agent finds 99% of the holds and never picks one from another problem. Real gym photos are harder: today it finds about half to 60%, and we're working on why. It stops too early on dense walls. Every experiment is logged in the repo." (update with the latest real-photo numbers before recording) |
| 2:20–2:45 | Diagram from the README; guardrails list | "The agent only uses tools that measure the photo. It has limits on turns and holds, ignores text in the photo, and sends only the photo and the tap, never names. If Meta's API fails, the app falls back to a simple color method. The final say is always the climber's." |
| 2:45–3:00 | Team names, repo URL, "Tap one hold. Get the whole problem." | "Built during the hackathon on top of our app: code at github.com/Mirai-Climbing/topo-agent. Tap one hold. Get the whole problem." |

## Shots to film (撮る物)

- [ ] Climbing in a gym, 5–10 seconds (no other people's faces; ask the gym first) / ジムで登るところ(ほかの人の顔は映さない。ジムに一言聞く)
- [ ] Phone screen recording of the whole flow, in the app with the reviewer account / 審査用のアカウントで、アプリの流れを画面録画
- [ ] The tape-problem sample wall in the app, with "What the AI did" open / テープの見本の壁で「AI がしたこと」を出したところ
- [ ] A real gym wall photo for the flow above, once real-photo results are in / 本物のジムの写真(本物の写真の結果が出たら)

## Notes

- Keep it under 3:00. Show the product first; numbers and architecture after.
- The color-blindness numbers come from the review linked in the README; say "about one man in twenty in Japan", not more.
- Say clearly that the topo agent was built during the hackathon, on top of an existing app.
