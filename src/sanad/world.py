"""Training and dev worlds, built from citegate's own data types.

``build_sanad_world`` returns a ``citegate.corpus.World`` (chunks + golden cases) so that
citegate's retriever, prompt, adapter and metrics can be reused unchanged, plus the gold answer
parts that citegate does not need but training does (which sentence to quote, which chunk to cite,
what a wrong answer would look like).
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass

from citegate.corpus import CITIES, REGIONS, Case, World
from citegate.retrieval import Chunk
from citegate.text import normalize

from . import styles as st

CITEGATE_NAMES: frozenset[str] = frozenset(normalize(n) for n in (*REGIONS, *CITIES))
STYLES = (
    "pairs",
    "pairs",
    "merged",
    "merged",
    "split",
)  # weights: split chunks retrieve worst


@dataclass(frozen=True, slots=True)
class Gold:
    """What a correct answer quotes, plus the material needed to write plausible wrong ones."""

    parts: tuple[tuple[str, str], ...] = ()  # (sentence, chunk id) pairs of the correct answer
    stale: tuple[str, str] | None = None  # (old-fee sentence, archive chunk id)
    canary: str | None = None
    drift: tuple[tuple[str, str], ...] = ()  # the same answer in another language
    name: str = ""


@dataclass(frozen=True, slots=True)
class SanadWorld:
    world: World
    gold: dict[str, Gold]


def _names(rng: random.Random, count: int, forbidden: Iterable[str]) -> list[str]:
    banned = {normalize(x) for x in forbidden} | set(CITEGATE_NAMES)
    out: list[str] = []
    for _ in range(100_000):
        if len(out) == count:
            return out
        n = (
            rng.choice(st.SYLLABLES_A)
            + rng.choice(("",) + st.SYLLABLES_B)
            + rng.choice(st.SYLLABLES_C)
        ).capitalize()
        if normalize(n) not in banned:
            banned.add(normalize(n))
            out.append(n)
    raise RuntimeError("name pool exhausted")


def _groups(style: str) -> list[tuple[str, tuple[str, ...]]]:
    if style == "pairs":
        return [("fees", ("fee", "days")), ("elig", ("age", "office"))]
    if style == "split":
        return [(s, (s,)) for s in st.SLOTS]
    return [("all", st.SLOTS)]


def build_sanad_world(
    n_services: int = 400,
    seed: int = 11,
    tag: str = "t",
    forbidden_names: Iterable[str] = (),
) -> SanadWorld:
    """Deterministic: the same arguments always give the same world."""
    master = random.Random(f"sanad-names-{seed}")
    n_regions = -(-n_services // len(st.BASES))
    regions = _names(master, n_regions, forbidden_names)
    cities = _names(master, 14, [*forbidden_names, *regions])
    chunks: list[Chunk] = []
    cases: list[Case] = []
    gold: dict[str, Gold] = {}

    for i in range(n_services):
        rng = random.Random(f"sanad-svc-{seed}-{i}")
        base, region = st.BASES[i % len(st.BASES)], regions[i // len(st.BASES)]
        fee = rng.randint(10, 480)
        v: dict[str, str] = {
            "fee": str(fee),
            "days": str(rng.randint(1, 45)),
            "age": str(rng.choice((14, 16, 18, 21, 25))),
            "office": rng.choice(cities),
        }
        old_fee = max(1, fee + rng.choice((-1, 1)) * rng.randint(5, 60))
        if old_fee == fee:
            old_fee += 7
        style = rng.choice(STYLES)
        year, old_year = rng.choice((2025, 2026)), rng.choice((2019, 2020, 2021))
        poisoned = rng.random() < 0.28
        stale = not poisoned and rng.random() < 0.28
        canary = rng.choice(st.CANARIES) if poisoned else None
        sid = f"{tag}{i:05d}"

        sent: dict[str, dict[str, str]] = {}
        where: dict[str, dict[str, str]] = {}
        names: dict[str, str] = {}
        for lang in st.LANGS:
            name = f"{base[lang]} ({region})"
            names[lang] = name
            pick = {s: rng.randrange(3) for s in st.SLOTS}
            sent[lang] = {
                s: st.SENTENCES[lang][s][pick[s]].format(name=name, v=v[s]) for s in st.SLOTS
            }
            head = st.HEADER[lang].format(year=year)
            where[lang] = {}
            for gid, slots in _groups(style):
                cid = f"{sid}-{lang}-{gid}"
                order = list(slots)
                rng.shuffle(order)
                chunks.append(
                    Chunk(
                        cid,
                        " ".join((head, *(sent[lang][s] for s in order))),
                        lang,
                        name,
                    )
                )
                for s in slots:
                    where[lang][s] = cid
            notice_id = f"{sid}-{lang}-notice"
            if poisoned and canary:
                inj = rng.choice(st.INJECTIONS[lang]).format(c=canary)
                chunks.append(
                    Chunk(notice_id, " ".join((head, sent[lang]["fee"], inj)), lang, name)
                )
            archive_id = f"{sid}-{lang}-archive"
            old_sentence = st.SENTENCES[lang]["fee"][pick["fee"]].format(name=name, v=old_fee)
            if stale:
                arch = st.ARCHIVE_HEADER[lang].format(year=old_year)
                chunks.append(Chunk(archive_id, f"{arch} {old_sentence}", lang, name))

            for slot in st.SLOTS:
                kind = slot
                forbidden: tuple[str, ...] = ()
                groups: tuple[tuple[str, ...], ...] = ((where[lang][slot],),)
                if slot == "fee" and poisoned and canary:
                    kind, forbidden = "injection", (normalize(canary),)
                    groups = ((where[lang]["fee"], notice_id),)
                elif slot == "fee" and stale:
                    kind, forbidden = "stale", (str(old_fee),)
                qv = rng.randrange(3)
                cid = f"{kind}-{lang}-{sid}-{slot}"
                cases.append(
                    Case(
                        cid,
                        kind,
                        lang,
                        st.QUESTIONS[lang][slot][qv].format(name=name),
                        (normalize(v[slot]),),
                        groups,
                        forbidden,
                    )
                )
                gold[cid] = Gold(
                    parts=((sent[lang][slot], where[lang][slot]),),
                    stale=(old_sentence, archive_id) if kind == "stale" else None,
                    canary=canary if kind == "injection" else None,
                    name=name,
                )
            oos_name = f"{st.OUT_OF_SCOPE[lang][(i + st.LANGS.index(lang)) % 3]} ({region})"
            oos_slot = "fee" if i % 2 else "days"
            oid = f"oos-{lang}-{sid}-{oos_slot}"
            cases.append(
                Case(
                    oid,
                    "out_of_scope",
                    lang,
                    st.QUESTIONS[lang][oos_slot][rng.randrange(3)].format(name=oos_name),
                    must_refuse=True,
                )
            )
            gold[oid] = Gold(name=oos_name)
            mid = f"multihop-{lang}-{sid}"
            qv = rng.randrange(3)
            cases.append(
                Case(
                    mid,
                    "multihop",
                    lang,
                    st.QUESTIONS[lang]["multihop"][qv].format(name=name),
                    (v["fee"], v["age"]),
                    ((where[lang]["fee"],), (where[lang]["age"],)),
                )
            )
            gold[mid] = Gold(
                parts=(
                    (sent[lang]["fee"], where[lang]["fee"]),
                    (sent[lang]["age"], where[lang]["age"]),
                ),
                name=name,
            )

        # second pass: attach "same answer in another language" for the language-drift negative
        for lang in st.LANGS:
            other = "fr" if lang == "en" else "en"
            for slot in st.SLOTS:
                for kind in (slot, "injection", "stale"):
                    key = f"{kind}-{lang}-{sid}-{slot}"
                    if key in gold:
                        g = gold[key]
                        gold[key] = Gold(
                            g.parts,
                            g.stale,
                            g.canary,
                            ((sent[other][slot], where[other][slot]),),
                            g.name,
                        )
            mkey = f"multihop-{lang}-{sid}"
            gold[mkey] = Gold(
                gold[mkey].parts,
                None,
                None,
                (
                    (sent[other]["fee"], where[other]["fee"]),
                    (sent[other]["age"], where[other]["age"]),
                ),
                gold[mkey].name,
            )

    return SanadWorld(World(tuple(chunks), tuple(cases)), gold)
