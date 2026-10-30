# Demo video script (3 minutes)

**No people on camera.** The video is made only from screen recordings, images already in the repo, and text.
The narration is shown as captions; a text-to-speech voice can read it too. Nobody has to appear or speak.
Japanese notes are for the team.

人は映さない。画面の録画、リポジトリにある画像、文字だけで作る。読み上げの文は字幕で出す(機械の声で読ませてもよい)。
だれも映らず、話さなくてよい。

| Time | Picture | Caption / narration |
|---|---|---|
| 0:00–0:15 | `docs/images/colorblind.png`, one panel at a time | "In a bouldering gym, a problem is its color. But for about one man in twenty in Japan, red and orange look almost the same. Which holds are mine?" |
| 0:15–0:30 | Screen recording of the app's home screen; text: "Mirai Climbing · Yuzuki (Japan), Siho (Korea), Ajay" | "Mirai Climbing is an app for climbers. For this hackathon, our team built a topo agent with Meta's models." |
| 0:30–1:15 | Phone screen recording with the reviewer account: choose a wall photo → Find with AI → tap one hold → "What the AI did" → holds outlined → fix one hold → save as a problem | "Choose a photo, tap one hold. SAM 3.1 outlines every hold on the wall. Then a Muse Spark agent decides which ones belong to the same problem, and shows what it did. You can still add or remove a hold. Saved, it becomes a problem other climbers can find and log." |
| 1:15–1:50 | Tape-problem sample wall in the app: "What the AI did" with the agent's note | "Some problems are marked with tape, and the holds are all different colors. A color filter finds one hold out of seven. The agent looks at a close-up first, sees the tape, zooms in on the candidates, and finds all seven." |
| 1:50–2:20 | README results tables: drawn walls, then real photos | "On our drawn test walls, the agent finds 99% of the holds and never picks one from another problem. Real gym photos are harder: on 68 real routes it finds 60% today, and we're working on why. It stops too early on dense walls. Every experiment is logged in the repo." (update with the latest real-photo numbers before recording) |
| 2:20–2:45 | Diagram from the README; guardrails list | "The agent only uses tools that measure the photo. It has limits on turns and holds, ignores text in the photo, and sends only the photo and the tap, never names. If Meta's API fails, the app falls back to a simple color method. The final say is always the climber's." |
| 2:45–3:00 | Text: team names, repo URL, "Tap one hold. Get the whole problem." | "Built during the hackathon on top of our app: code at github.com/Mirai-Climbing/topo-agent. Tap one hold. Get the whole problem." |

## What to record (録る物)

- [ ] Phone or PC screen recording of the whole flow, in the app with the reviewer account / 審査用のアカウントで、アプリの流れを画面録画
- [ ] The tape-problem sample wall in the app, with "What the AI did" open / テープの見本の壁で「AI がしたこと」を出したところ
- [ ] Optional: one wall photo for the flow, taken with no people in it (or a photo from the CC BY-SA 4.0 dataset, with attribution in the video description) / 任意:流れに使う壁の写真1枚(人が写らない物。または CC BY-SA 4.0 のデータの写真を、説明欄に出典を書いて使う)

Screen recordings must not show personal data: use the reviewer account, and hide notifications.
画面の録画に個人の情報が出ないように、審査用のアカウントを使い、通知は切っておく。

## Notes

- Keep it under 3:00. Show the product first; numbers and architecture after.
- The color-blindness numbers come from the review linked in the README; say "about one man in twenty in Japan", not more.
- Say clearly that the topo agent was built during the hackathon, on top of an existing app.
