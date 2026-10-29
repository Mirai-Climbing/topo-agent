# Real-route leaderboard

This is a snapshot exported from the topo-agent MCP server. The eight photos
and 68 annotated routes are a development set, not an independent test set.
Accuracy is recalculated by the server using `topo_agent.eval.match`.
Execution times are reported by submitters and are not independently verified.

Dataset and evaluator version: `a16e466eb5b6f7c19ca23b5c6fdd09d8077a7d6c7667f6ea9488035e701ac035`

| Rank | Author | Method | Commit | F1 | Recall | Precision | Mean seconds | Run ID |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Siho | meta | 6bc820c | 0.639 | 0.601 | 0.683 | 30.9 | d3d6a16969cb488dbcf6403ec1d27a0d |
| 2 | Siho | meta3-winner | 6bc820c | 0.569 | 0.569 | 0.569 | 16.1 | 6505b7da6eb34efeba50afdb99654361 |
| 3 | Siho | meta3 | 6bc820c | 0.501 | 0.543 | 0.464 | 12.7 | 7f1d69eda29145bda845f250cfc9f95f |
