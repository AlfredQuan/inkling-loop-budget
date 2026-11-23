# GitHub release checklist

**Scope:** Publish the source and results of an offline research experiment.
The bundle has no deployed service, private dataset, model checkpoint,
customer state or external uptime commitment. The project maintainer decides
whether to publish. No public repository, tag or package has been created.

| Check | Evidence | Disposition |
| --- | --- | --- |
| Native collection | [Run record](m2/evidence/run.json): 24/36/48 layers, full offload, 1,092 inputs per pass including four controls | Pass |
| Offline regression and replay | `python -m unittest discover -s . -p test_cases.py`, `python -m unittest discover -s m2`, `python verify_release.py` in an isolated Linux CPU environment | Pass |
| Artifact integrity | SHA-256 check in `verify_release.py`; fresh training reproduces M1/M2 metrics and coefficients within tolerance and M2 policies exactly | Pass |
| Inputs and dependencies | [Backend pins](BACKEND.md), [ARC and GGUF pins](m2/REPRODUCE.md), [CPU dependencies](requirements.txt); no base model in Git | Pass |
| Distribution and attribution | [MIT](LICENSE), [data and model notices](THIRD_PARTY.md); public evidence carries method and limitations | Pass |
| Claim boundary | [Results](m2/RESULTS.md): experiment passes the replay gate; no measured speedup, deployment or cross-task generalization claim | Pass |
| Hosted CI | [Replay workflow](.github/workflows/replay.yml) mirrors local checks | Confirm first public CI run after publishing |

**Pipeline:** checksum-pinned model and ARC data → isolated H20 probe → signed
paired outcomes and compressed trace → small heads and metrics → independent
CPU replay → publish this source tree and its evidence unchanged. `m2/evidence/run.json`
identifies exactly what was checked. The replay does not access the network
after dependencies are installed. No release decision relies on a previous
cache hit.

For the initial release the default branch contains source and evidence. A
later change to model, data, prompt or router needs a new run record and a
passing replay; keep the current reference results available. If the initial
hosted replay fails, pause release announcements, preserve the failed logs,
fix the first incompatible or missing input, and rerun. If the CPU replay or
published hashes no longer match, withdraw any affected performance claim
until the checked artifacts are restored. No runtime rollback is applicable.
