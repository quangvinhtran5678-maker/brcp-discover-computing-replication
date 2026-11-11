from pathlib import Path
import shutil
from docx import Document
from docx.shared import Pt

ROOT=Path('/mnt/data/BRCP_Revision_2026-11-11_Followup')
OLD=Path('/mnt/data/BRCP_Revision_2026-11-11_Next')
for i in range(1,8):shutil.copy2(OLD/f'Fig{i}.pdf', ROOT/f'Fig{i}.pdf')
s=(OLD/'main.tex').read_text()
# Fix a cross-section temporal contradiction from older language.
old='''The paper does not evaluate a published time-series foundation-model checkpoint, and no model-specific claim about Chronos, TimesFM, or Moirai follows from the reported results.'''
new='''The historical main experiment does not evaluate a published time-series foundation-model checkpoint; the separate post-hoc Chronos-Bolt Small analysis in Section~\\ref{sec:chronos_posthoc} provides exploratory checkpoint-specific evidence only. No claim about TimesFM, Moirai, or training-corpus disjointness follows from the historical results.'''
assert old in s
s=s.replace(old,new)
old2='''including published foundation-model checkpoints, multi-horizon calibration, conditional rather than marginal coverage, and closely spaced breaks.'''
new2='''including independent new hold-outs and additional published foundation-model checkpoints, multi-horizon calibration, conditional rather than marginal coverage, and closely spaced breaks.'''
assert old2 in s
s=s.replace(old2,new2)

block=r'''
%====================== Independently seeded audit ========================
\section{Independent-seed detector null diagnostic and source fidelity}
\label{sec:fresh_detectors}

\paragraph{Pre-frozen diagnostic design (separate from Chronos).}
To probe whether our detector-space comparison remains fair on genuinely
new data, we created a separate \emph{detector-only} synthetic audit with
independent seeds, a correctly specified stationary Gaussian AR(1) null
($\phi=0.6$), and persistent level and innovation-variance changes.
Each series has 1,100 observations, with a 300-step detector warm-up,
250-point initial anchor, 60-point post-alarm re-anchoring, two-sided
standardized/winsorized Page increments ($k=0.8$), and 800 monitored steps.
The level intervention adds $(1-\phi)\,2$ to innovations starting at
$t=650$, while the variance intervention triples innovation scale;
when these interventions are absent, the data-generating law is exactly
stationary. Residual scores use the frozen, correctly specified AR(1)
one-step predictor. A fixed grid was selected using 90 independent
\emph{development} seeds to target 0.5 alarms per 800 null steps.
The selected thresholds ($h_{\mathrm{score}}=5$, $h_{\mathrm{data}}=16$)
were then evaluated \emph{once} on 300 disjoint test seeds, without
retuning. This is not a Chronos or BR-CP interval-coverage experiment.

\begin{table}[ht]
\centering
\small
\caption{New, independent-seed AR(1) detector-only diagnostic. The
prespecified alarm-rate difference tolerance is 0.20 alarms per 800
steps; observed test difference 0.3533 \emph{fails} that matching gate.
Post-shift hit rate is at least one alarm in the first 120 steps. These
hit rates are descriptive, \emph{not} a false-alarm-matched power comparison.}
\label{tab:fresh_detectors}
\begin{tabular}{lrrr}
\toprule
Detector & Null alarms / 800 & Level hit rate & Variance hit rate \\
\midrule
Score-space ($h=5$) & 0.5300 & 0.7067 & 0.9800 \\
Data-space ($h=16$) & 0.1767 & 0.9833 & 0.9733 \\
\bottomrule
\end{tabular}
\end{table}

The test null alarm-rate gap (score minus data) was $0.3533$ per
800 steps; a paired seed bootstrap (5,000 draws) gave an exploratory
95\% percentile interval $[0.250,0.453]$. Thus even with
independent development selection, a true no-change process and
fresh test seeds, the empirical matching target did \emph{not}
transport to the hold-out. We did not reselect thresholds using test
results. Accordingly, the post-change hit-rate comparison cannot
isolate monitoring-space effects at equal false-positive operation.
The variance-pulse conditional median delay among detected streams
was 3 steps for score-space and 24 for data-space, whereas under a
level shift score-space had the lower observed hit probability; these
are ancillary descriptive quantities under \emph{unequal} test alarm
budgets. This test does not prove any universally superior detector.
The frozen protocol, raw per-seed outcomes, explicit code and a
150-case equivalence test against the original Python detector are
included with the supplementary audit.

\paragraph{Reference-algorithm and checkpoint provenance audit.}
The implementation tagged \texttt{AgACI} in the historical archive
performs interval-score exponential-weight aggregation of ACI experts.
The published AgACI of \citet{zaffran2022adaptive} instead aggregates
interval boundaries using an online-expert algorithm such as Bernstein
Online Aggregation and a quantile-oriented loss. Hence these methods
must not be equated. Likewise, the historical \texttt{ConformalPID}
code is a hand-specified quantile tracker with a bounded tanh
integrator, omitting the learned scorecaster of
\citet{angelopoulos2023conformal}, and is better called a
\emph{simplified PI-like comparator} rather than a source-equivalent
full PID replication. These archived names are preserved solely for
replay compatibility; numerical claims apply only to the implemented
variants. The official Chronos-Bolt Small model card
(\url{https://huggingface.co/amazon/chronos-bolt-small}) describes
pretraining at large scale and zero-shot results on a benchmark suite,
but the materials inspected here did not establish byte-level
train/test non-overlap for the exact six CSV versions in our audit.
We therefore do not assert strict training-set disjointness.

'''
assert '\\section{Limitations}' in s
s=s.replace('\\section{Limitations}',block+'\\section{Limitations}',1)
assert 'The post-hoc published-checkpoint audit also showed' in s
s=s.replace('The post-hoc published-checkpoint audit also showed',
'''The independent-seed synthetic detector null check also failed its prespecified held-out alarm-rate matching tolerance; consequently no general superiority of monitoring spaces is inferred from the unmatched hit rates. The post-hoc published-checkpoint audit also showed''',1)
(ROOT/'main.tex').write_text(s)

doc=Document(OLD/'BRCP_Point_by_Point_Response_POSTHOC_CHRONOS_PRE_SUBMISSION.docx')
doc.add_heading('Further independent-seed audit and reference-fidelity correction — 11 November 2026',level=1)
p=doc.add_paragraph('STATUS: NEW SYNTHETIC DETECTOR-ONLY EVIDENCE; NOT A MODEL CHECKPOINT VALIDATION AND NOT AN EQUAL-FALSE-ALARM SUPERIORITY TEST.')
p.runs[0].bold=True
doc.add_heading('Reviewer 1 and Reviewer 3 — held-out detector budgets',level=2)
doc.add_paragraph('We prospectively froze a Gaussian AR(1) detector-only protocol (phi=0.6) before drawing 90 independent development seeds and 300 new test seeds. Using 1,100 observations per series, 300-step warm-up, 250-point initial reference and 60-point re-anchoring, development-only selection yielded h_score=5 and h_data=16. On the new stationary/no-change test streams, observed alarm counts were 0.5300 versus 0.1767 per 800 steps, respectively. The gap of 0.3533 exceeded the prespecified matching tolerance of 0.20, and paired bootstrap 95% CI was [0.250, 0.453]. We did not tune on held-out outcomes. Thus a new, known-stationary null itself demonstrates that our current detector comparison has not established matched false-alarm operation. We explicitly withdraw any general detector superiority inference from the observed detection rates.')
doc.add_paragraph('Descriptive (unequal-null-budget) post-shift hit rates in the next 120 steps: level 0.7067 (score) versus 0.9833 (data), variance 0.9800 (score) versus 0.9733 (data). These are not comparisons at matched specificity. The supplementary source includes the frozen protocol, 90-seed grid, 300-seed per-method results and 150 fast/Python detector equivalence checks.')
doc.add_paragraph('Scientific status: independent detector-only null evaluation completed; equal-alarm-budget superiority remains UNRESOLVED and is not claimed. The previous Chronos post-hoc null matching statement remains an exploratory diagnostic on already inspected series.')
doc.add_heading('Reviewer 1 and Reviewer 3 — published baseline fidelity',level=2)
doc.add_paragraph('A source-to-implementation audit confirms that the archived method named AgACI is an exponential interval-score weighted ACI ensemble, whereas the published AgACI of Zaffran et al. (ICML 2022) uses online aggregation of lower/upper interval endpoints, with BOA among its evaluated algorithms. The archived ConformalPID is a simplified PI-like quantile tracker with a tanh integrator and no learned scorecaster, rather than the full published Conformal PID framework of Angelopoulos et al. (NeurIPS 2023). Results are therefore attributed strictly to locally implemented variants; the original CSV identifiers remain unchanged for reproducibility. A full reference-algorithm replication remains open and is not represented as completed.')
doc.add_heading('Published-checkpoint training-corpus provenance',level=2)
doc.add_paragraph('The Amazon Chronos-Bolt Small model card describes large-scale pretraining and claims zero-shot performance on a separate 27-dataset benchmark suite. This does not certify that each of our six exact evaluation CSV versions was absent from the model training mixture. No source-equivalent dataset membership audit or model-training replay was accomplished; the manuscript explicitly avoids a strict disjoint-training-corpus claim.')
doc.add_heading('Exact supplementary evidence and author sign-off',level=2)
doc.add_paragraph('Artifacts: fresh_detector_null_audit.py, FROZEN_PROTOCOL_SYNTHETIC_NULL.json, DEVELOPMENT_THRESHOLD_GRID.csv, FROZEN_SELECTED_THRESHOLDS.json, FRESH_HELDOUT_PER_SEED.csv, FRESH_HELDOUT_SUMMARY.csv, FRESH_PAIRED_BOOTSTRAP.csv, FRESH_STUDY_STATUS.json, plus reference-fidelity source audit. The manuscript includes the additional independent-seed section and a distinct table. This response and all attachments remain PRE-SUBMISSION. Authors must still approve author identifiers, declarations, and upload completeness; published-baseline fidelity and strict model-corpus disjointness remain unverified.')
style=doc.styles['Normal']
style.font.size=Pt(10)
doc.save(ROOT/'BRCP_Point_by_Point_Response_FURTHER_AUDIT_PRE_SUBMISSION.docx')
print('Updated LaTeX, DOCX.')
