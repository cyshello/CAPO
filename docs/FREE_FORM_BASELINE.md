# OpenAI static-meta replace-body captioning path

The active fresh prompt-delta captioning path has one instruction-generator
implementation:

```text
current segment frames
  -> deterministic 0.5 FPS subset
  -> half-resolution JPEG copies
  -> gpt-4o-mini + current parent meta-prompt + frozen preceding captions
  -> one plain-text segment instruction
  -> replace the caption prompt task body
  -> local Qwen captioner on all original segment frames, without history
```

The generator never receives a downstream question, choices, answers,
correctness, trajectories, or feedback. Its request contains only the current
video/segment identity, timestamps, downscaled frame references, frozen
preceding-caption history, and generator identity.

The implementation is split across:

- `prompt_routing/static_meta_replace_body.py`: request, frame, parser, and
  composition policy;
- `prompt_routing/policies/openai_free_form_generator.py`: the sole hosted
  `gpt-4o-mini` provider;
- `prompt_routing/policies/replace_body_scaffold.py`: replace-body composition;
- `captioning/history_aware_baseline.py`: builder/worker integration and the
  history-blind captioner call.

There is no local-Qwen prompt-generator fallback. The local Qwen model remains
the captioner and DVD raw-vision backend. The property-bank router and its
deterministic scaffold remain available only for the preserved Phase 0–3 and
property-routing paths; the fresh prompt-delta runner does not select them.

Generator defaults are `gpt-4o-mini`, 512 maximum output tokens, 0.5 FPS, and
scale 0.5. The initial text is
`optimization/prompts/init_meta_prompt.json`; candidate/confirmed parent text
is passed through the same OpenAI generator without changing request shape.

Resume identity includes provider, model/backend, meta-prompt ID and hash,
request/parser/composition versions, frame policy, and max tokens. Earlier
local-Qwen or append-scaffold artifacts therefore cannot alias this path.
