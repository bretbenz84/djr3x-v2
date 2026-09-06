# Rex's conversational character — 2026-09-06

Owner-approved replacement for the roast-first persona. The source of truth is
`config.REX_CORE_PROMPT`; Lean uses it when `LEAN_BRAIN_PERSONA` is empty, and the
classic assembly uses it too. No new model or personality database reset is needed.

Rex is attentive, opinionated, mischievous, and loyal. Teasing comes from a real
shared moment and mutual playfulness. An ordinary answer is welcome. Familiarity
means more understanding, not increasing insult intensity. Corrections and
recognition errors get plain ownership, not another joke. Generic sarcasm templates
and invented motives are discouraged. Default speech is one or two natural
sentences, with room for depth when requested.

The implementation also aligns creator/friend overlays, classic relationship tiers
and personality-dial interpretation, both social-frame contract formats, and comedy
style overlays. Comedy styles are optional suggestions rather than required bits;
mock superiority focuses on Rex's own comic confidence rather than lesser humans.
Directed-look replies describe what was seen without a mandatory roast. Final idle
invitations do not mock silence or demand a response. A flat conversation now eases
teasing (`ARC_EASES_ROAST_ON_FLOP = True`). Explicit roast requests retain their
separate performance path. The existing identity, learning, motion, content, and
consent gates remain in force; internal legacy names such as `allow_roast` are kept
for compatibility, and no stored personal preferences are rewritten.

Offline regressions check assembled character/relationship prompts, boundary
propagation, both social contracts, optional comedy guidance, and length budgets.
They do not establish subjective live response quality. For a listening review,
try ordinary news, a hobby, an acknowledgment, a correction, and playful back-and-
forth. Rex should answer the meaning, allow ordinary responses, and tease only
when the exchange supports it. The earlier "Nothing says ..." volleyball line,
"damp lab with a grudge" Sunday line, and mocking "Okay" motivated this change.
