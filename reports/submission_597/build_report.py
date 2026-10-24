"""Build the three-page Assignment 3 report, matching the supplied A2 PDF.

Run with the bundled Python runtime (reportlab). The evaluation script uses
the project venv separately. All performance/configuration facts are read
from the immutable evaluated V12 archive, not the current V14 working model.
"""
from pathlib import Path
from html import escape
import hashlib
import json
import zipfile

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    Flowable, KeepTogether,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "2023BB10156.pdf"
ARCHIVE = ROOT / "dist/rank-fusion-submission-v12.zip"
ARCHIVE_SHA = "e42c56344bcce528e48b803ba2fb265aa3adb93cddc53791876dbd9f373b6e28"
assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == ARCHIVE_SHA
with zipfile.ZipFile(ARCHIVE) as z:
    def read(name):
        return json.loads(z.read(name))
    DEV = read("results/dev_metrics_v12.json")
    TREE = read("results/tree_fit_v10.json")
    NEURAL = read("results/neural_selection_v12.json")
    CV = read("results/lambdamart_cv_v12.json")["results"][0]
    EXEC = read("results/execution.json")
    REPRO = read("results/pairwise_reproducibility_v12.json")
    CORE_SHA = hashlib.sha256(z.read("submission/pointwise_pairwise.py")).hexdigest()

SWEEP = json.loads((HERE / "tree_count_sweep.json").read_text())
LEADER = json.loads((ROOT / "results/leaderboard_v12.json").read_text())
assert LEADER["submission"] == 597
assert LEADER["protected_v12_sha256"]["model.txt"] == TREE["artifact_sha256"]
assert abs(DEV["lambdamart_ndcg@10"] - SWEEP["full_submitted_model_dev_ndcg_10"]) < 1e-12

FONT_DIR = Path("/System/Library/Fonts/Supplemental")
for name, filename in [("Arial", "Arial.ttf"), ("Arial-Bold", "Arial Bold.ttf"),
                       ("Arial-Italic", "Arial Italic.ttf")]:
    pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / filename)))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic")

BLUE = colors.HexColor("#173b63")
INK = colors.HexColor("#303030")
GRID = colors.HexColor("#b2c3da")
PALE = colors.HexColor("#edf2f7")
WIDTH, HEIGHT = A4
MARGIN = 46
CONTENT = WIDTH - 2 * MARGIN
styles = {
    "title": ParagraphStyle("title", fontName="Arial-Bold", fontSize=17.5, leading=21,
                            textColor=colors.black, spaceAfter=6),
    "subtitle": ParagraphStyle("subtitle", fontName="Arial", fontSize=9.6, leading=12,
                               textColor=colors.HexColor("#666666"), spaceAfter=8),
    "body": ParagraphStyle("body", fontName="Arial", fontSize=9.5, leading=12.0,
                           textColor=INK, spaceAfter=6),
    "small": ParagraphStyle("small", fontName="Arial", fontSize=8.45, leading=10.6,
                            textColor=INK, spaceAfter=5),
    "caption": ParagraphStyle("caption", fontName="Arial", fontSize=8.2, leading=10.1,
                              textColor=INK, spaceAfter=6),
    "heading": ParagraphStyle("heading", fontName="Arial-Bold", fontSize=11.4, leading=14,
                              textColor=BLUE, spaceBefore=5, spaceAfter=7, keepWithNext=True),
    "cell": ParagraphStyle("cell", fontName="Arial", fontSize=8.15, leading=10.0, textColor=INK),
    "cell_small": ParagraphStyle("cell_small", fontName="Arial", fontSize=7.8, leading=9.3,
                                 textColor=INK),
    "cellhead": ParagraphStyle("cellhead", fontName="Arial-Bold", fontSize=8.2, leading=10,
                               textColor=colors.black),
}


def P(text, style="body"):
    return Paragraph(text, styles[style])


class Rule(Flowable):
    def __init__(self, space=3):
        super().__init__()
        self.width, self.height = CONTENT, space

    def draw(self):
        self.canv.setStrokeColor(colors.black)
        self.canv.setLineWidth(.4)
        self.canv.line(0, self.height, self.width, self.height)


def section(title):
    return [Rule(), P(title, "heading")]


def table(headers, rows, widths, small=False):
    cell_style = "cell_small" if small else "cell"
    cells = [[P(h, "cellhead") for h in headers]]
    cells += [[P(str(x), cell_style) for x in row] for row in rows]
    t = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PALE),
        ("GRID", (0, 0), (-1, -1), .4, GRID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


class SweepPlot(Flowable):
    """Small, vector scientific plot with a fully stated evaluation split."""
    def __init__(self):
        super().__init__()
        self.width, self.height = CONTENT, 137

    def draw(self):
        c = self.canv
        panels = [(0, "Released DEV (in-sample)", "dev_ndcg_10_in_sample", .55, .76, [.55, .60, .65, .70, .75]),
                  (258, "TRAIN-only grouped CV", "train_cv_validation_mean", .44, .51, [.44, .46, .48, .50])]
        for offset, title, key, low, high, ticks in panels:
            x0, y0, w, h = offset + 31, 26, 204, 85
            c.setFillColor(INK)
            c.setFont("Arial-Bold", 8.7)
            c.drawString(offset + 5, 124, title)
            c.setFont("Arial", 7.5)
            def xp(x): return x0 + (x-200)/600*w
            def yp(y): return y0 + (y-low)/(high-low)*h
            for y in ticks:
                c.setStrokeColor(colors.HexColor("#e0e5eb")); c.setLineWidth(.4)
                c.line(x0, yp(y), x0+w, yp(y))
                c.setFillColor(INK); c.drawRightString(x0-5, yp(y)-2.5, f"{y:.2f}")
            c.setStrokeColor(INK); c.setLineWidth(.5)
            c.line(x0, y0, x0+w, y0); c.line(x0, y0, x0, y0+h)
            points = [(xp(a["trees"]), yp(a[key])) for a in SWEEP["points"]]
            c.setStrokeColor(BLUE); c.setLineWidth(1.4)
            path = c.beginPath(); path.moveTo(*points[0])
            for pt in points[1:]: path.lineTo(*pt)
            c.drawPath(path)
            for a, (x,y) in zip(SWEEP["points"], points):
                if offset:
                    e = a["train_cv_validation_std"]
                    c.setLineWidth(.6); c.line(x, yp(a[key]-e), x, yp(a[key]+e))
                    c.line(x-3, yp(a[key]-e), x+3, yp(a[key]-e))
                    c.line(x-3, yp(a[key]+e), x+3, yp(a[key]+e))
                c.setFillColor(BLUE); c.circle(x,y,2.4,stroke=0,fill=1)
                c.setFont("Arial", 7.5); c.setFillColor(INK)
                c.drawCentredString(x, y0-12, str(a["trees"]))
            c.setFont("Arial", 7.5)
            c.drawCentredString(x0+w/2, 1, "Trees in the unaugmented query-rank member")


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(GRID); canvas.setLineWidth(.5)
    canvas.line(40, 30, WIDTH-40, 30)
    canvas.setFillColor(colors.HexColor("#777777")); canvas.setFont("Arial", 8)
    canvas.drawString(40, 19, "Maaz Akhtar · 2023BB10156 | Assignment 3")
    canvas.drawRightString(WIDTH-40, 19, str(doc.page))
    canvas.restoreState()


story = []
story += [P("Rank Fusion and Learning to Rank - Report", "title"),
          P("COL 764/7364 Assignment 3 · Maaz Akhtar · 2023BB10156 · Submission #597 (V12)", "subtitle")]
story += section("1. Final system, models and data use")
story += [P("This report describes <b>submission #597</b>, the evaluated V12 archive. It implements handwritten rank fusion, pointwise and pairwise neural training, a LightGBM LambdaMART-family scorer, and a portable training recipe. Later experimental scorers are excluded."),
P("<b>Fusion.</b> The submitted method is per-query min-max <b>CombSUM</b>, with weights 0.3 for BM25 whole-document, 0.3 for BM25 title, 0.4 for LMIR.DIR and 0 for LMIR.JM. RRF and CombMNZ are also implemented. A fixed 1,146-configuration TRAIN-only comparison selected these weights; no DEV labels enter fusion tuning."),
P("<b>Neural models.</b> Pointwise training minimizes mean squared error. RankNet uses the stable loss <b>mean[softplus(other score - preferred score)]</b>; sampled pairs stay within a query, exclude equal grades and put the preferred document first. The competitive pairwise trainer additionally weights pairs by their nDCG swap effect. Each member has 136 → 128 → 64 → 1 ReLU layers, 30 MSE warm-up epochs at learning rate 0.001 and two RankNet epochs at 0.0001, with fresh Adam, batch 1,024 and at most 300 pairs/query/epoch. Seeds 0 and 41 are equally averaged and compiled into a staff-loadable 136 → 256 → 128 → 1 MLP. Pointwise uses six MSE epochs."),
P("<b>Tree features and training.</b> Six weighted LambdaRank members form one native 3,800-tree model. The 136 provided features are supplemented by query-local midrank percentiles and min-max values, giving 408 internal features. Transforms use only current candidates, preserve missing values, and map constants to 0.5. Training-only copies of queries with affine changes, feature dropout and noise encourage robustness; they do not recreate the hidden C2 versions. See Section 4 for exact member settings [1,2]."),
P("<b>Split declaration.</b> The tree was fitted on released Fold 1 <b>TRAIN+DEV</b>: 723,412 + 235,259 rows, 6,000 + 2,000 queries, with split-namespaced query IDs. Its DEV score is <b>in-sample</b>. Tree hyperparameters/weights used grouped TRAIN-only CV; final counts were fixed with no early stopping. Both neural artifacts were fitted on TRAIN only, with standardization learned from training rows. Pointwise epoch selection used internal TRAIN fold 0 with patience 5, then a six-epoch full-TRAIN refit; pairwise used fixed-epoch five-fold selection. No neural fitting or stopping used released DEV.", "small")]
story += section("2. Recorded results for the submitted system")
story += [table(["Model / method", "DEV nDCG@10", "Interpretation"], [
    ["Single BM25 feature", f'{DEV["bm25_ndcg@10"]:.6f}', "Untrained reference"],
    ["Pointwise MLP", f'{DEV["pointwise_ndcg@10"]:.6f}', "TRAIN-only fit"],
    ["Pairwise MLP", f'{DEV["pairwise_ndcg@10"]:.6f}', "TRAIN-only fit"],
    ["<b>LambdaMART ensemble</b>", f'<b>{DEV["lambdamart_ndcg@10"]:.6f}</b>', "<b>TRAIN+DEV fit; in-sample</b>"],
], [170, 96, CONTENT-266]), Spacer(1,6),
P("Fusion achieved <b>0.3090 nDCG@10 / 0.1297 MAP@10</b>. The supplied #597 grader snapshot reports <b>P = 0.4419</b> on held-out queries, <b>C2 mean = 0.3795</b>, 7/7 conformance and 10/10 correctness. Its checkpoint position was 10/35, live score 0.4722/0.65. These class-relative values are a recorded checkpoint, not a final rank; no C1 score was supplied.", "small")]
story += section("3. Correctness, runtime and transfer recipe")
story += [P("The V12 release passed <b>176 tests</b> locally, on the college server and in CI, including loss/SGD examples, pair sampling, determinism, staff loading and interface checks. Its two fresh pairwise fits were byte-identical; the tree was the retained, previously verified V10 artifact, not a new V12 fit. Recorded preparation was <b>5.680 s</b>, worst fusion <b>0.0011 s</b> and reranking <b>0.4885 s</b>, inside the 60 s / 2 s limits. Local staff loading used Python 3.12.14; HPC used Python 3.9.21.", "small"),
P("The unchanged C1 recipe accepts arbitrary feature widths, uses raw values plus query percentiles, and chooses between two leaf capacities from query count (15/31, 31/63 or 63/127). Its supplied validation file selects capacity/iteration; it uses learning rate 0.05, L2 = 10, patience 40 and a deterministic 200-500-tree budget. No hidden benchmark was identified or accessed.", "small")]

story += [PageBreak()]
story += section("4. Tree-count sweep, generalization and final configuration")
story += [P("<b>Controlled tree-count comparison.</b> The archived TRAIN-only sweep varied 200/400/800 trees in an unaugmented query-rank member, holding the other settings fixed (127 leaves, minimum leaf 100, L2 = 10, learning rate 0.05). The corresponding DEV prefix scores below were computed <b>after submission for this report, on 24 October 2026</b>, from that member inside the exact #597 artifact. They are retrospective <b>in-sample</b> measurements, not new tuning or replacements for the submitted model.", "small"),
SweepPlot(),
P("Figure 1. DEV nDCG@10 versus tree count (left); archived five-fold TRAIN validation means with ±1 fold SD (right). The two panels use different vertical scales. The member's positive export weight changes scores but not its ordering; these are component scores, not the full six-member ensemble.", "caption"),
table(["Trees", "DEV<br/>(in-sample)", "CV train", "CV validation<br/>mean ± SD", "Public diagnostic<br/>mean"], [[
    a["trees"], f'{a["dev_ndcg_10_in_sample"]:.6f}', f'{a["train_cv_training_mean"]:.6f}',
    f'{a["train_cv_validation_mean"]:.6f} ± {a["train_cv_validation_std"]:.6f}',
    f'{a["train_cv_diagnostic_mean"]:.6f}'
] for a in SWEEP["points"]], [40,83,76,153,CONTENT-352]), Spacer(1,6),
P("<b>Observed gap.</b> From 200 to 800 trees, the component's in-sample DEV rises from 0.586381 to 0.727855, while grouped-CV validation rises only from 0.482170 to 0.484133; the public diagnostic mean falls from 0.460040 to 0.454990. Thus extra fitting capacity mostly improves seen-query performance. For the final ensemble's TRAIN-only CV counterpart, mean training nDCG is <b>0.732114</b>, unseen-fold mean <b>0.487560 ± 0.009312</b>, and validation/training ratio <b>0.665962</b> (absolute gap 0.244554). This is a method-level estimate on TRAIN, not CV of the later TRAIN+DEV refit.", "small"),
P("All five folds are grouped by query (seed 0); validation queries never contribute fitting rows or standardizer statistics. Counts are fixed rather than stopped on the reported fold. Regularization combines minimum leaf size 100, L2 = 10, row/feature subsampling, query-relative features, augmentation and averaging. The final TRAIN-only diagnostic means are affine 0.479000, dropout 0.477154 and noise 0.474921 (mean 0.477025). These public perturbations are diagnostics, not estimates of staff C2.", "small"),
P("<b>Final configuration and tradeoff.</b> I retained the six-member tree below. All use LambdaRank, 127 leaves (no separate depth cap), learning rate 0.05, minimum leaf 100, L2 = 10, feature and row fractions 0.8, bagging frequency 1, seed 0 and four threads. The fifth and sixth members differ through extra-tree randomization; the sixth enables extra trees. Truncation is 30 except member 3 uses 13. The native export preserves the weighted sum; the full model's 0.682399 DEV score is not the 800-tree component's score.", "small"),
table(["Member features", "Trees", "Weight", "Training augmentation"], [
    ["1. Raw + percentile", "800", "0.0703125", "None"],
    ["2. Min-max + percentile", "200", "0.0703125", "None"],
    ["3. Raw + percentile", "800", "0.1406250", "Half-query dropout / noise"],
    ["4. Raw + percentile + min-max", "800", "0.2812500", "Half-query mixed perturbations"],
    ["5. Raw + percentile + min-max", "800", "0.1875000", "Full-query mixed perturbations"],
    ["6. Same features; extra trees", "400", "0.2500000", "Full-query mixed perturbations"],
], [192,42,70,CONTENT-304], small=True), Spacer(1,6),
P("V12's four new tree candidates / 33 settings did not pass the frozen robustness gates (mixture gain 0.000311 &lt; 0.0005; diagnostic gain 0.000551 &lt; 0.001), so the evaluated tree bytes were retained. For pairwise training, 50 fresh CV member fits favored 30-epoch warm-up over the 20-epoch control: mean 0.441474 versus 0.440371, gain 0.001103 with four positive folds. The fresh control reproduced archived evidence before promotion. Choices were committed before final fitting/DEV evaluation. I favored supported CV/robustness gains over accepting every marginal increase in seen-query accuracy.", "small"),
P("<b>What the grader adds.</b> The pairwise DEV-to-held-out difference is 0.4520 - 0.4419 = 0.0101. Tree DEV 0.6824 versus C2 0.3795 shows why the large DEV score cannot establish generalization: DEV was in fitting and C2 averages different hidden conditions. Their difference is not a matched clean-test overfitting estimate. No per-version C2 results or C1 outcome were available.", "small")]

story += [PageBreak()]
story += section("5. AI-use disclosure and code provenance")
story += [P("<b>AI use.</b> I directed the development, set experimental priorities and chose submission iterations. I used <b>OpenAI Codex</b> as a programming assistant to help read the specification, propose and implement methods, write and revise code, test, evaluate, debug, execute college-HPC experiments, interpret results, package submissions and prepare this report. I uploaded the archives and checked the grader outcomes. The assistance included substantial code and documentation generation."),
P("<b>File provenance.</b> The table groups principal files/families in the V12 archive and the later report files. <i>Assignment-specific</i> means created with the AI assistance above. The course harness, network/standardizer, metrics, grouped-fold helper and interfaces supplied the scaffolding. No external fusion library, ready-made neural ranking loss/trainer, another student's solution or extra relevance-label dataset was used. The tree learner is the permitted LightGBM implementation [1,2].", "small"),
table(["Edited / added file(s) or family", "What it does", "Source / provenance"], [
    ["submission/ranking.py", "Preparation, candidate-only fusion/reranking; stable tie handling.", "Course interface; assignment-specific internals."],
    ["submission/fusion.py<br/>submission/fusion_config.json", "Handwritten RRF, CombSUM/MNZ and weighted variants; selected weights.", "Assignment formulas / primer; TRAIN-selected settings."],
    ["submission/pointwise_pairwise.py", "MSE, pairs, stable RankNet, SGD steps, weighted training and seeded initialization.", "Course skeleton extended; handwritten loss/training logic [1]."],
    ["submission/data.py<br/>submission/features.py", "Bounded SVMLight loading; raw, percentile and min-max representations.", "Assignment-specific; course data/interface conventions."],
    ["submission/tree_ensemble.py<br/>scripts/pairwise_ensemble.py", "Compile weighted trees and two neural members into standard loadable artifacts.", "Assignment-specific numeric export; LightGBM/PyTorch formats."],
    ["submission/recipe.py", "Adaptive, budgeted training for staff-supplied benchmark files.", "Assignment-specific policy; LightGBM learner [2]."],
    ["submission/model.txt", "Retained V10 six-member, 3,800-tree scorer.", "LightGBM 4.6.0; released TRAIN+DEV only."],
    ["submission/pointwise_model.pt<br/>submission/pairwise_model.pt", "Six-epoch pointwise fit; V12 two-seed pairwise fit.", "PyTorch; TRAIN only. Pairwise independently reproduced."],
    ["scripts/train_model.py<br/>scripts/cv_lambdamart*.py<br/>scripts/lambdamart_utils.py", "Tree fitting, fixed-count grouped CV and native export checks.", "Assignment-specific / adapted entry points; course CV/metrics."],
    ["scripts/cv_pairwise*.py<br/>scripts/augmentation.py<br/>scripts/robustness.py", "Neural comparisons, query augmentation and label-preserving stress diagnostics.", "Assignment-specific; public diagnostic assumptions."],
    ["scripts/evaluate_models.py<br/>scripts/record_artifacts.py<br/>scripts/hpc_*.pbs", "Final interface evaluation, artifact hashes, PBS job execution.", "Assignment-specific orchestration; course interfaces / IITD HPC [3]."],
    ["tests/test_*.py<br/>scripts/smoke_test.sh<br/>.github/workflows/", "Hand examples, regression/export/staff-loading checks and CI.", "Course tests/scaffolding retained or extended; added checks."],
    ["configs/*.json<br/>results/*.json and evidence subfolders", "Frozen search spaces, selected settings, CV/fit/runtime/hash receipts.", "Own experiment configurations and machine-generated evidence; grader feedback labeled separately."],
    ["README.md<br/>docs/EXPERIMENTS*.md<br/>docs/SUBMISSION_NOTES.md", "Usage, method, split declarations, frozen protocols and provenance.", "Course README/docs extended; assignment-specific documentation."],
    ["requirements.txt / .gitignore<br/>reports/submission_597/", "Dependencies/exclusions; retrospective report sweep, evidence and PDF generation.", "Course manifests adapted; report-only code generated with Codex."],
], [197,171,CONTENT-368], small=True), Spacer(1,7),
P("<b>References and sources.</b> [1] Course Assignment 3 specification, RankNet primer and starter repository; Christopher J. C. Burges, <i>From RankNet to LambdaRank to LambdaMART: An Overview</i>, MSR-TR-2010-82, 2010 (<link href='https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/' color='#173b63'>Microsoft Research</link>). [2] <link href='https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html' color='#173b63'>LightGBM 4.6.0 parameter documentation</link>. [3] <link href='https://supercomputing.iitd.ac.in/' color='#173b63'>IIT Delhi HPC guide</link>. Recorded metrics/configurations: archived results/dev_metrics_v12.json, lambdamart_cv_v3.json, lambdamart_cv_v12.json, neural_selection_v12.json and tree_fit_v10.json; user-supplied #597 grader snapshot.", "caption"),
P("<b>Artifact provenance.</b> Competition archive: rank-fusion-submission-v12.zip; commit f28b61880a56cac51566890671405db013bb3c94. SHA-256:<br/><font name='Arial' size='7.7'>" + ARCHIVE_SHA + "</font><br/>The report adds a post-submission diagnostic; the evaluated archive and artifacts remain unchanged.", "caption")]

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                        topMargin=42, bottomMargin=43, title="Rank Fusion and Learning to Rank - Submission 597",
                        author="Maaz Akhtar - 2023BB10156", subject="COL 764/7364 Assignment 3 report")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(OUT)

facts = {
    "submission": 597, "report_date": "2026-10-24",
    "archive_sha256": ARCHIVE_SHA, "archive_commit": "f28b61880a56cac51566890671405db013bb3c94",
    "model_sha256": TREE["artifact_sha256"], "pointwise_pairwise_source_sha256": CORE_SHA,
    "dev": DEV, "leaderboard_checkpoint": LEADER,
    "cv_control_summary": {k: CV[k] for k in ["train_mean", "validation_mean", "validation_std", "validation_train_ratio", "scenario_means"]},
    "pairwise_cv": {"validation_mean": NEURAL["selected"]["validation_mean"],
                    "mean_gain": NEURAL["proposed_mean_gain"], "improved_folds": NEURAL["proposed_improved_folds"]},
    "retrospective_sweep": SWEEP,
    "pdf_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
}
(HERE / "report_evidence.json").write_text(json.dumps(facts, indent=2) + "\n")
