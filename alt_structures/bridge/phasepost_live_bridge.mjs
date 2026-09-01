// phasepost_live_bridge.mjs — run this ecosystem's REAL production pipeline
// (eoreader7's extractRelations -> phasepost.js, backed by live_priors' real
// ActPrior@1 lexicon) over clauses supplied by eo-lexical-analysis-2.0, and
// emit the same {op, grain, cell, standing, candidates} verdict shape
// phasepost.js itself returns. No new classification logic lives here.
//
// WHY A BRIDGE, NOT A REIMPLEMENTATION: PR#18 (this repo) found "eoreader7
// and the-fold aren't in this session's GitHub access... testing whether the
// cube earns its keep inside that live pipeline needs that access granted
// first." This script is exactly that test, now that access exists — it
// imports the real, unmodified eoreader7/live_priors modules rather than
// porting their logic into Python a second time.
//
// WHAT IS AND ISN'T "the live pipeline" here, stated plainly rather than
// implied. `extractRelations`'s own vocabulary-discovery step
// (discoverRelationVocab) anchors candidate verbs on capitalised recurring
// NAMED SURFACES across a DOCUMENT (native/adapters/text/relations.js's own
// header: "the token immediately FOLLOWING a candidate referent surface").
// eo-lexical-analysis-2.0's corpus is one decontextualized clause per row,
// with no surrounding document and often no proper-noun subject at all ("I
// am forwarding...", "Peter is looking...") -- discoverRelationVocab's own
// recurrence floor (minSurfaces) cannot clear on material shaped like that,
// confirmed empirically before this bridge was written (alt_structures/
// README.md, "Why extractRelations needs a supplied vocabulary here").
// So vocabulary
// DISCOVERY is not exercised here. What IS the real, unmodified production
// code: given a vocabulary (any Set of verb forms -- `verbs` never requires
// capitalization, `W` in relations.js is `[\p{L}\p{N}_'-]+`, no case
// constraint), extractRelations's actual subject/object capture,
// negation/polarity detection, and clause-boundary arithmetic all run
// byte-identical to any other caller -- by default at the SAME (DR4,DR5)
// = (off,off) configuration live_priors' own corpus-mining driver
// actually runs today (scripts/eot-digest.mjs's loadOrgans(), called with
// no args), not the {true,true} config its separate goldens-evaluation
// path uses (scripts/measure-dr45-at-scale.mjs, goldens/reading/
// diff-golden.mjs) -- set DR45=1 to try that reading instead; see the
// results doc for which config produced which numbers. The vocabulary this bridge supplies
// per clause is that clause's OWN main verb surface form (found by the
// Python side's POS tagger, the same heuristic act_prior_lexical.py/
// verbnet_lexical.py already use in this repo) -- so this bridge answers
// "given a clause's own verb, does the REAL extractor + REAL phasepost.js
// structure it the way production would," not "can this pipeline discover
// its own vocabulary from a single bare clause" (a question relations.js's
// own header already answers no to, by design, for exactly this kind of
// material).
//
// Protocol: argv[2] = input JSONL path ({id, clause, verb} per line),
// argv[3] = output JSONL path. Pure stdin/stdout-free file protocol so the
// Python caller can shell out without piping large text through a pipe.

import fs from "node:fs";
import path from "node:path";
import readline from "node:readline";
import { fileURLToPath } from "node:url";

// Sibling-checkout convention (act_prior_lexical.py's own pattern in this
// repo): this file lives at <repo>/alt_structures/bridge/, so three levels
// up is the directory holding every repo this session has as a sibling.
const SIBLINGS_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..", "..");
const EOREADER7_ROOT = process.env.EOREADER7_PATH || path.join(SIBLINGS_ROOT, "eoreader7");
const LIVE_PRIORS_ROOT = process.env.LIVE_PRIORS_PATH || path.join(SIBLINGS_ROOT, "live_priors");

const { extractRelations } = await import(path.join(EOREADER7_ROOT, "native", "adapters", "text", "relations.js"));
const { makePhasepost } = await import(path.join(EOREADER7_ROOT, "native", "adapters", "text", "phasepost.js"));
const { cellOf } = await import(path.join(EOREADER7_ROOT, "native", "kernel", "cube.js"));
const priors = await import(path.join(EOREADER7_ROOT, "native", "adapters", "text", "priors.js"));
const morph = await import(path.join(EOREADER7_ROOT, "legacy-eoreader6.1", "packages", "engine", "perceiver", "text", "morphology.js"));

const actPriorPath = path.join(LIVE_PRIORS_ROOT, "derived-priors", "act-priors", "act-prior-en.json");
const morphPriorPath = path.join(process.env.THE_FOLD_PATH || path.join(SIBLINGS_ROOT, "the-fold"), "eval", "fixtures", "unimorph-morphology-prior.json");

if (!fs.existsSync(actPriorPath)) {
  console.error(`ActPrior@1 fixture not found at ${actPriorPath} -- set LIVE_PRIORS_PATH to a live_priors checkout.`);
  process.exit(1);
}
if (!fs.existsSync(morphPriorPath)) {
  console.error(`UniMorph morphology prior not found at ${morphPriorPath} -- set THE_FOLD_PATH to a the-fold checkout.`);
  process.exit(1);
}

const actPrior = JSON.parse(fs.readFileSync(actPriorPath, "utf8"));
const morphPrior = JSON.parse(fs.readFileSync(morphPriorPath, "utf8"));
const { lemmasOf } = morph.createLemmatizer(morphPrior.forms, { language: morphPrior.language });

const pp = makePhasepost({
  actPrior,
  cellOf,
  definiteDeterminers: priors.DEFINITE_DETERMINERS,
  indefiniteDeterminers: priors.INDEFINITE_DETERMINERS,
  lemmasOf: (f) => lemmasOf(f),
});

const [, , inPath, outPath] = process.argv;
if (!inPath || !outPath) {
  console.error("usage: node phasepost_live_bridge.mjs <in.jsonl> <out.jsonl>");
  process.exit(1);
}

const rl = readline.createInterface({ input: fs.createReadStream(inPath), crlfDelay: Infinity });
const out = fs.createWriteStream(outPath);

let n = 0, matched = 0, multiMatch = 0;
for await (const line of rl) {
  const t = line.trim();
  if (!t) continue;
  n++;
  const row = JSON.parse(t);
  const verbSet = new Set([String(row.verb || "").toLowerCase()]);
  let rels = [];
  if (verbSet.size && [...verbSet][0]) {
    // DR4 (nounPhraseSubjects) / DR5 (phrasalPredicates) default to
    // false, matching live_priors' own actual corpus-mining driver
    // (scripts/eot-digest.mjs's `loadOrgans()`, called with no args at
    // its own top-level call site) -- NOT the `{phrasalPredicates:true,
    // nounPhraseSubjects:true}` config live_priors' separate goldens
    // evaluation path (goldens/reading/diff-golden.mjs) and its own DR4/5
    // measurement script use. Overridable via DR45=1 for the disclosed
    // alternate reading -- see the results doc for which was used where.
    const dr45 = process.env.DR45 === "1";
    rels = extractRelations(row.clause, { verbs: verbSet, nounPhraseSubjects: dr45, phrasalPredicates: dr45 });
  }
  if (rels.length > 1) multiMatch++;
  if (rels.length === 0) {
    out.write(JSON.stringify({ id: row.id, matched: false }) + "\n");
    continue;
  }
  matched++;
  const edge = rels[0];
  const verdict = pp.classify(edge);
  out.write(JSON.stringify({
    id: row.id,
    matched: true,
    nRels: rels.length,
    subject: edge.subject,
    verb: edge.verb,
    object: edge.object,
    polarity: edge.polarity,
    op: verdict.op,
    grain: verdict.grain,
    cellTerrain: verdict.cell ? verdict.cell.terrain : null,
    standing: verdict.standing,
    candidates: verdict.candidates ? verdict.candidates.map((c) => c.op) : null,
    because: verdict.because,
  }) + "\n");
}
out.end();
console.error(`bridge: ${n} clauses, ${matched} produced a real SVO match (${(100 * matched / n).toFixed(1)}%), ${multiMatch} had >1 match (kept the first)`);
