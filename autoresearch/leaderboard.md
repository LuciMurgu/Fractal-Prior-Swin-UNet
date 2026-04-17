# AutoResearch Leaderboard

Total experiments: 5
Improvements found: 1
Best dice: 0.8096

| # | Run ID | Dice | Improvement | Mutations |
|---|--------|------|-------------|-----------|
| 1 | baseline_H | 0.8096 | ✅ | Config H baseline |
| 2 | autoresearch_000 | 0.7893 | ❌ | data.patch_size: [384, 384] → [256, 256] |
| 3 | autoresearch_001 | 0.7973 | ❌ | data.photometric_aug.contrast: 0.15 → 0.1; loss.skeleton_recall.weight: 0.15 → 0.2 |
| 4 | autoresearch_002 | 0.7969 | ❌ | loss.fractal_bce.weight: 0.1 → 0.2 |
| 5 | autoresearch_003 | 0.7971 | ❌ | loss.cldice.weight: 0.12 → 0.15 |