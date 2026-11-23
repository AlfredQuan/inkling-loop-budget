# M2 protocol: can the first pass predict the value of two more?

This is an offline research experiment. It does not implement within-request KV
reuse, and its accuracy and cost criteria are research gates rather than a
production promise. The primary decision is whether a hidden-state head earns
its complexity over a choice-margin threshold.

- **Unit:** one ARC question answered by the pinned frozen model at one, two
  and three passes. Parse four choices, use the largest of the four raw
  single-token logits after the model's answer prefix, and compare to the
  official key. Validate all four answer tokens on explicit arithmetic controls
  before accepting a sweep. Keep M1 as a separate synthetic benchmark.
- **Split:** ARC Easy and Challenge, 256 training, 96 validation and 192 test
  examples per subset. Use official splits, remove duplicate normalized
  question texts across all splits, and deterministically shuffle choice order.
  Report both subsets; public pretrained-data overlap is unknown.
- **Supervision:** `correct_3 - correct_1` in {-1, 0, +1}. Fit two ridge heads:
  confidence features alone, or the same features plus 96 seeded projections
  of the last prompt token's normalized readout at one pass. No later-pass
  signal is available at decision time. Keep the base weights frozen.
- **Baselines:** fixed 1/2/3; smallest 1-pass margin first; learned confidence;
  mean treatment benefit by ARC subset (an oracle-like metadata advantage);
  and a gold-informed oracle that only upgrades when it repairs a mistake.
- **Budget:** 24, 36, 48 layer visits per prompt token at fixed depths 1/2/3.
  A restart can cost 24+48. A reuse estimate costs 24+30: 24 new loop visits
  plus rerunning the six final layers. Neither proxy is measured latency.
  Tune thresholds on validation only, with a primary cap of 40% of prompt
  tokens routed to three passes (10% saving in the restart proxy at that cap).
- **Gate:** held-out saved layer visits ≥10% for restart and paired stratified
  95% bootstrap lower bound of accuracy difference vs fixed 3 ≥−5 percentage
  points. Check the state head against confidence and fixed-depth alternatives;
  treat crossed budgets and small sample differences as descriptive. Count
  missed rescues, newly harmed answers and repairs explicitly.
- **Release rule:** an experimental release can contain a negative result if
  clean CPU replay, data lineage, native inference checks, regression checks and
  claims all match. No production speed claim or online early exit is permitted
  on these artifacts alone.
