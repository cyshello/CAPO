# Fixed VideoMME training-size cohorts

These files freeze video-level cohorts for future VideoMME-long training-size
comparisons. Each non-empty line is one exact VideoMME `videoID`; every
selected video has three QA records and an available local video file.

Selection is deterministic:

1. collect all usable VideoMME-long videos;
2. sort their unique `videoID` values lexicographically;
3. construct one full permutation with Python `random.Random(0).sample(pool,
   len(pool))`;
4. use prefixes 10, 20, and 50 as the training cohorts;
5. use permutation positions 51 through 65 as the confirmation cohort.

This guarantees:

- `10samples.txt` is a subset of `20samplestxt`;
- `20samplestxt` is a subset of `50samples.txt`;
- `confirmation.txt` contains 15 videos disjoint from all training cohorts.

These files are selection manifests only. They do not modify the current
Phase 4 `split_manifest.json`, runtime pointers, caches, or active train roles.
