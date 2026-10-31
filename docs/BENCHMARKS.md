# Evaluation protocols and results

The aggregate CSV contains 19 metric rows across 11 completed local evaluations
of `step-00129185`. All local runs use context 2,048 and 0-shot prompts. Choice
tasks use teacher-forced raw completion with native BOS; generation uses native
chat unless noted. Greedy generation uses repetition penalty 1.0.

| Task | Count | Scoring and generation allowance |
| --- | ---: | --- |
| HellaSwag | 10,042 | Raw and character-normalized choice accuracy |
| PIQA | 1,838 | Raw and character-normalized choice accuracy |
| ARC-Challenge | 1,172 | Raw and character-normalized choice accuracy |
| WinoGrande | 1,267 | Choice accuracy |
| IFEval | 541 prompts / 834 instructions | Prompt and instruction strict/loose; max 1,280 new tokens |
| TriviaQA | 17,944 | Exact match; max 256; official newline/period/comma stops |
| GSM8K | 1,319 | Official strict and flexible extraction exact match; max 1,024 |
| MMLU | 14,042 / 57 subjects | Official raw multiple choice; question-weighted accuracy |
| HumanEval-Instruct | 164 | Official instruct prefill, greedy pass@1; max 1,024 |
| MBPP-Instruct | 500 | Official instruct prefill, explicit 0-shot override; max 256 |
| BBH | 6,511 / 27 tasks | Official `bbh_cot_zeroshot`, strict/flexible extraction; max 1,024 |

BBH flexible 1,310/6,511 and strict 281/6,511 are question-weighted totals, not
unweighted task means. MMLU is 3,531/14,042. GSM8K flexible 116/1,319 and strict
1/1,319 must remain separately labelled. IFEval instruction metrics have 834
constraints in their denominator; prompt metrics have 541 prompts. Selecting
only certain bars for a graphic does not remove other measured metrics.

MBPP task 493 has a 2,647-token prefix. It is a context-limit failure in the
500-example denominator, with no forwarding, truncation or code execution.
HumanEval and eligible MBPP candidates were graded twice in restricted networkless
containers; both grading passes agreed. Their local container adapter differs
from host `code_eval`. The measured historical coding image used nonroot UID,
read-only root, no mounts/network/GPU/capabilities, 256 MiB, 32 PIDs and a
3-second limit. Historical BBH text was never executed.

BBH generated 847,520 tokens, with 726 outputs reaching their token allowance.
Both official CPU grading passes agreed. Aggregate breakdowns are included in
[mmlu_subjects.csv](../results/mmlu_subjects.csv) (57 subjects) and
[bbh_tasks.csv](../results/bbh_tasks.csv) (27 variants). The tables include
integer counts and denominators; their totals match `benchmarks.csv`.
Raw responses, grading test data and historical benchmark controllers remain
outside the source distribution.

## Published reference comparisons

The simplified charts contain historical model-card and third-party scores,
alongside local measurements. Reference sources and exact caveats are recorded
in [benchmark_sources.json](benchmark_sources.json). Differences include raw vs
normalized accuracy, cloze vs choice MMLU, shot counts, IFEval averaging, base vs
instruct prompting, generation budgets, extraction and pass@1 protocols.
Do not interpret the charts as a matched ranking or evidence that LaMoE is
generally better than another model. No cross-task average is computed.

The selected chart set has HellaSwag, PIQA, ARC-Challenge, WinoGrande, IFEval,
GSM8K, MMLU and BBH. Full measured results additionally retain TriviaQA and both
coding tasks. LiveCodeBench was cancelled before scoring; CyberSecEval and
SWE-bench have no completed result. BoolQ was not run. CyberGym tasks were skipped.
No missing, cancelled or skipped task receives a zero score.

Training overlap is not known to be absent. Lexical screening against a frozen
set is not exhaustive semantic decontamination. An external WSL interruption was
recovered from verified cached GSM8K responses; final process time is not total
effort. No completed benchmark was rerun to create this public source release.
