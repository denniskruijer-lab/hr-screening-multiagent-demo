# Multi-agent architecture: alternatives considered

This project uses a **supervisor pattern**: one dispatcher agent (the
supervisor) calls two worker agents (screening, calibration) as tools, then
reconciles their results. Two other common multi-agent patterns were
considered and deliberately not built. This is a design decision, not a
missing feature -- knowing when *not* to reach for a more elaborate pattern
is as much a part of the job as knowing how to build one.

## The three patterns

**Supervisor (chosen).** A single coordinator dispatches to specialist
workers and aggregates their output. The coordinator is the only agent that
talks to more than one other agent; workers don't talk to each other at all.

**Network (decentralized).** Every agent can message every other agent
directly, with no central coordinator. Used when workers genuinely need to
negotiate with each other repeatedly (e.g. multiple agents jointly refining
a shared plan).

**Hierarchical (supervisors of supervisors).** A tree of supervisors, each
coordinating a group of workers or sub-supervisors. Used when there are many
distinct domains of work, each large enough to need its own coordination
layer.

## Why supervisor fits this problem

This pipeline has exactly two workers with a fixed, one-shot relationship:
screen, then independently calibrate, then compare the two scores. There is
no back-and-forth negotiation between the workers -- calibration doesn't
need to argue with screening, it just needs to reach its own opinion and let
the supervisor compare the two.

- **Network would add complexity with no corresponding benefit.** Letting
  the two workers message each other directly raises real questions --
  who decides when the conversation is over? What stops them from
  reinforcing each other's mistakes instead of staying independent? --
  that a single dispatcher avoids by construction: the supervisor decides
  the sequence, and the workers never influence each other directly (which
  is also *why* the calibration score is a genuinely independent second
  opinion, not somethin the two agents talk each other into).
- **Hierarchical would be premature abstraction at this scale.** A second
  layer of supervision only pays for itself when there are enough distinct
  workstreams that one coordinator can't reasonably track all of them. Two
  workers don't clear that bar; adding the layer anyway would just be
  indirection with nothing to justify it.
- **Supervisor keeps the mental model consistent with the rest of the
  codebase.** The supervisor is not a new orchestration primitive -- it's
  a plain `Agent` (see `src/agent.py`) whose tools happen to call other
  agents' `.run()` methods instead of plain functions (see
  `src/agents/supervisor.py`). Anyone who understands the core loop
  already understands the supervisor.

## When this would change

If a third worker were added whose job depends on negotiating with the other
two (rather than being independently dispatched and reconciled), or if the
number of workers grew enough that a single supervisor became a bottleneck
for reasoning about the whole system, that would be the point to revisit
this decision -- not before.
