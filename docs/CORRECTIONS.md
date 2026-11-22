# Corrections log

Dated record of anything that was wrong, did not reproduce, or was retracted. Newest first.

| Date | What was claimed or built | What is true | Fix |
|---|---|---|---|
| 2026-11-21 | Training and serving chose bfloat16 whenever `torch.cuda.is_bf16_supported()` was True | On a T4-class GPU that function returns True because bf16 is **emulated**. The SFT smoke test crashed (`gemm input type BFloat16 ... only supported for CUDA devices with compute capability 8.0 or higher`) and baseline serving was very slow (p50 13 s). The baseline quality numbers were produced with emulated bf16 compute. Latency numbers from that run are not representative. | `hw.py`: dtype follows the real compute capability; TRL's fp32 matmul fallback on older GPUs. Needs re-verification with a 20-step smoke test |
| 2026-11-21 | The notebook was safe to run top to bottom | After the failed SFT step the next cells still tried to serve a missing adapter and printed an unrelated 401 error from Hugging Face. | Shell steps raise on failure; adapters and merged models are asserted before use |
| 2026-11-19 | The suite was reported as passing everywhere | On the first Windows run (Python 3.14), 24 of 25 tests passed. A test read a UTF-8 file with the Windows default encoding (cp1252) and failed; `sanad show` could also crash when printing Arabic to a cp1252 console. The suite had only been run on Linux before release. Both fixed, the second with a test that fails without the fix. | `fix(tests)`, `fix(cli)` commits |
| 2026-11-18 | The 0.2 prototype shipped its own scorer, gate, mutants, baseline and results page | That duplicated citegate. It was removed; Sanad now imports citegate. | `docs/adr/0001-evaluation-is-delegated-to-citegate.md` |
| 2026-11-18 | A rule-based baseline scored 100% on the prototype's templated eval, and 94% before a case-sensitivity bug in the baseline was fixed | Templated evals are easy for rules. The lesson carries over: citegate's reference agent passes its golden set, so a high model score is a floor, not proof. | `README.md` limitations; a shift suite is a candidate upstream contribution to citegate |
| 2026-11-18 | Initial draft of this repository was assisted by an AI model | Reviewed, tested and extended by the author before any result is reported | `docs/VERIFICATION.md` |
