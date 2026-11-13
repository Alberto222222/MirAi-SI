"""Sistema di raccomandazione «Monumenti simili a questo» (estensione per il corso di Sistemi Intelligenti).

Paradigma **content-based**: l'utente non ha uno storico di voti, quindi la foto che carica è il suo profilo
(«altri item come quello che mi interessa»). Gli item sono gli **edifici** del catalogo, non le singole foto.

Pipeline
1. Profilo dell'item: vettore di ogni foto nello spazio della testa MLP (512 attivazioni ReLU), normalizzato;
   il vettore dell'edificio è la media normalizzata delle sue foto.
2. Profilo dell'utente (query): vettore della foto caricata nello stesso spazio.
3. Punteggio di rilevanza: somiglianza del coseno nello spazio della testa (stile), eventualmente combinata con
   quella delle caratteristiche grezze (tipo di edificio, materiali), solo tra gli edifici dello stile predetto.
4. Diversità: riordino con **MMR** (Maximal Marginal Relevance) sui primi `n_pool` candidati:
   t* = argmax_t [ λ · Sim1(t, q) − (1 − λ) · max_{t_j già scelti} Sim2(t, t_j) ]
   Sim1 = somiglianza nello spazio della testa (stile), Sim2 = somiglianza delle caratteristiche visive grezze
   CLIP + DINOv2 (paesaggio, materiali, inquadratura). λ = 1 → nessuna diversificazione.
5. Relevance feedback di **Rocchio**: q' = α·q + β·media(item graditi) − γ·media(item non graditi).
6. Spiegazione: perché l'item è stato consigliato (stile/famiglia in comune, somiglianza, Paese).

Valutazione offline (notebook 09): rilevanza graduata rel = 3 stesso stile e stesso tipo di edificio (chiesa,
castello, palazzo, rovine antiche…), 2 stesso stile, 1 stessa famiglia, 0 altrimenti; Precision@k e Recall@k
(rilevante = stesso stile), nDCG@k, MRR, e metriche «beyond accuracy»: diversità intra-lista (ILD), varietà di Paesi,
copertura del catalogo e indice di Gini dell'esposizione (long tail).

Il modulo usa solo NumPy: i vettori nello spazio della testa si calcolano fuori (app_logic / notebook).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def normalizza(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float32)
    return A / (np.linalg.norm(A, axis=-1, keepdims=True) + 1e-12)


# ---------------------------------------------------------------------------
# Tipo di edificio (dalla proprietà Wikidata «istanza di»), raggruppato in 6 categorie
# ---------------------------------------------------------------------------
CATEGORIE = ("religioso", "fortificato", "antico", "residenza", "civile", "altro")
_PAROLE = {
    "antico": ("romano", "romana", "greco", "greca", "tempio", "anfiteatro", "sito archeologico", "rovine", "ninfeo",
               "arco trionfale", "foro", "acquedotto", "terme", "mausoleo"),
    "religioso": ("chies", "cappell", "cattedral", "basilic", "abbazi", "monaster", "convent", "oratori", "eremo",
                  "santuari", "collegiat", "sinagog", "chapel", "church", "duomo", "pieve", "canonica", "campanar",
                  "priorat", "priori", "battister", "moschea", "croce", "via crucis", "cimiter", "certosa", "kirche",
                  "église", "chiostro"),
    "fortificato": ("castell", "fortezz", "forte", "torre", "mura", "porta cittadina", "casaforte", "rocca", "bastion",
                    "fortificaz", "cinta", "caserma"),
    "residenza": ("casa", "palazz", "villa", "maniero", "château", "masia", "hôtel", "magione", "abitativ", "kamienica",
                  "residenz", "dimora", "manor", "house", "residential", "immeuble", "appartament"),
    "civile": ("rathaus", "municipio", "comunale", "teatro", "museo", "ponte", "fontana", "stazione", "albergo",
               "ospedale", "universit", "scuola", "scolastic", "consistorial", "bibliotec", "borsa", "mercato", "loggia",
               "monumento", "uffici", "museal", "mulino", "piazza"),
}


def categoria_edificio(tipo) -> str:
    """Categoria del tipo di edificio; le foto WikiChurches (senza tipo) sono chiese. «altro» = tipo generico o ignoto
    (es. «edificio»): per queste query la rilevanza «stile + tipo» non si calcola."""
    t = str(tipo).lower().strip()
    if t in ("", "nan", "none"):
        return "religioso"
    for cat in ("antico", "religioso", "fortificato", "residenza", "civile"):   # «teatro romano» → antico
        if any(p in t for p in _PAROLE[cat]):
            return cat
    return "altro"


# ---------------------------------------------------------------------------
# Catalogo degli item (edifici)
# ---------------------------------------------------------------------------
@dataclass
class Catalogo:
    """Edifici raccomandabili. H = vettori nello spazio della testa, G = caratteristiche grezze (diversità)."""
    H: np.ndarray                 # (n, d1) normalizzati
    G: np.ndarray                 # (n, d2) normalizzati
    stile: np.ndarray             # (n,) indice dello stile
    famiglia: np.ndarray          # (n,) indice della famiglia
    edificio: np.ndarray          # (n,) identificativo (QID Wikidata)
    nome: np.ndarray              # (n,) nome dell'edificio
    paese: np.ndarray             # (n,) Paese
    foto: np.ndarray              # (n,) foto rappresentativa (percorso relativo)
    categoria: np.ndarray = field(default=None)   # (n,) categoria del tipo di edificio
    n_foto: np.ndarray = field(default=None)

    def __len__(self) -> int:
        return len(self.stile)

    @classmethod
    def da_foto(cls, H: np.ndarray, G: np.ndarray, meta: pd.DataFrame, s2f: np.ndarray) -> "Catalogo":
        """Aggrega le foto per edificio. meta: colonne path, label, church_id, church_name, country e (se c'è) type,
        nello stesso ordine di H."""
        meta = meta.reset_index(drop=True)
        Hn, Gn = normalizza(H), normalizza(G)
        gruppi = meta.groupby(meta["church_id"].astype(str), sort=True).indices
        ids = list(gruppi)
        Hi = np.stack([Hn[gruppi[i]].mean(0) for i in ids])
        Gi = np.stack([Gn[gruppi[i]].mean(0) for i in ids])
        prima = np.array([gruppi[i][0] for i in ids])
        lab = np.array([np.bincount(meta["label"].to_numpy()[gruppi[i]]).argmax() for i in ids])
        col = lambda c, d="": (meta[c].astype(str).to_numpy()[prima] if c in meta else np.full(len(ids), d))
        return cls(H=normalizza(Hi), G=normalizza(Gi), stile=lab, famiglia=s2f[lab], edificio=np.array(ids),
                   nome=col("church_name"), paese=col("country"), foto=col("path"),
                   categoria=np.array([categoria_edificio(t) for t in col("type")]), n_foto=np.array([len(gruppi[i]) for i in ids]))

    def sottoinsieme(self, idx: np.ndarray) -> "Catalogo":
        idx = np.asarray(idx)
        return Catalogo(**{k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()})


# ---------------------------------------------------------------------------
# Raccomandazione
# ---------------------------------------------------------------------------
def mmr(rilevanza: np.ndarray, G: np.ndarray, k: int, lam: float) -> np.ndarray:
    """Riordino MMR: posizioni (in `rilevanza`) degli k item scelti uno alla volta."""
    n = len(rilevanza)
    k = min(k, n)
    if lam >= 1.0 or n <= 1:
        return np.argsort(-rilevanza, kind="stable")[:k]
    scelti = [int(np.argmax(rilevanza))]
    max_sim = G @ G[scelti[0]]
    libero = np.ones(n, dtype=bool)
    libero[scelti[0]] = False
    while len(scelti) < k:
        punt = np.where(libero, lam * rilevanza - (1 - lam) * max_sim, -np.inf)
        j = int(np.argmax(punt))
        scelti.append(j)
        libero[j] = False
        max_sim = np.maximum(max_sim, G @ G[j])
    return np.array(scelti)


def raccomanda(cat: Catalogo, h: np.ndarray, k: int = 5, stile: int | None = None, lam: float = 1.0,
               n_pool: int = 50, escludi: np.ndarray | None = None, g: np.ndarray | None = None,
               peso_testa: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Indici degli item consigliati e loro punteggio di rilevanza.

    h: vettore della query nello spazio della testa; g: vettore della query nelle caratteristiche grezze;
    peso_testa: punteggio = peso · cos(testa) + (1 − peso) · cos(grezze) (1 = solo spazio della testa);
    stile: filtro sullo stile predetto (None = tutti); escludi: maschera degli item da non proporre."""
    s = cat.H @ normalizza(h)
    if peso_testa < 1.0:
        if g is None:
            raise ValueError("per la somiglianza combinata serve il vettore grezzo g della query")
        s = peso_testa * s + (1 - peso_testa) * (cat.G @ normalizza(g))
    ok = np.ones(len(cat), dtype=bool) if stile is None else (cat.stile == stile)
    if escludi is not None:
        ok &= ~escludi
    cand = np.flatnonzero(ok)
    if len(cand) == 0:
        return np.array([], dtype=int), np.array([])
    pool = cand[np.argsort(-s[cand], kind="stable")[:max(n_pool, k)]]
    scelti = pool[mmr(s[pool], cat.G[pool], k, lam)]
    return scelti, s[scelti]


def rocchio(h: np.ndarray, graditi: np.ndarray, non_graditi: np.ndarray,
            alpha: float = 1.0, beta: float = 0.75, gamma: float = 0.15) -> np.ndarray:
    """Relevance feedback di Rocchio nello spazio della testa (vettori già normalizzati)."""
    q = alpha * normalizza(h)
    if len(graditi):
        q = q + beta * np.asarray(graditi).mean(0)
    if len(non_graditi):
        q = q - gamma * np.asarray(non_graditi).mean(0)
    return normalizza(q)


def spiegazione(cat: Catalogo, j: int, sim: float, stile_q: int, fam_q: int, nomi_stili: list[str],
                nomi_famiglie: list[str], lang: str = "it") -> str:
    """Breve motivazione della raccomandazione (le spiegazioni aumentano la fiducia nel RS)."""
    it = lang == "it"
    if cat.stile[j] == stile_q:
        motivo = (f"stesso stile ({nomi_stili[stile_q]})" if it else f"same style ({nomi_stili[stile_q]})")
    elif cat.famiglia[j] == fam_q:
        motivo = (f"stessa famiglia ({nomi_famiglie[fam_q]})" if it else f"same family ({nomi_famiglie[fam_q]})")
    else:
        motivo = "aspetto simile" if it else "similar look"
    paese = str(cat.paese[j]) if str(cat.paese[j]) not in ("", "nan") else ""
    sim_t = f"{'somiglianza' if it else 'similarity'} {sim:.0%}"
    return " · ".join(x for x in (motivo, sim_t, paese) if x)


# ---------------------------------------------------------------------------
# Metriche (dispensa §3.7)
# ---------------------------------------------------------------------------
def rilevanza_graduata(cat: Catalogo, idx: np.ndarray, stile_q: int, fam_q: int, cat_q: str | None = None) -> np.ndarray:
    """3 = stesso stile e stesso tipo di edificio, 2 = stesso stile, 1 = stessa famiglia, 0 = altrimenti."""
    rel = np.where(cat.stile[idx] == stile_q, 2, np.where(cat.famiglia[idx] == fam_q, 1, 0))
    if cat_q is not None and cat.categoria is not None:
        rel = rel + ((rel == 2) & (cat.categoria[idx] == cat_q))
    return rel


def dcg(rel: np.ndarray) -> float:
    rel = np.asarray(rel, dtype=float)
    return float(((2 ** rel - 1) / np.log2(np.arange(2, len(rel) + 2))).sum())


def ndcg(rel: np.ndarray, rel_ideale: np.ndarray) -> float:
    ideale = dcg(np.sort(np.asarray(rel_ideale))[::-1][:len(rel)])
    return dcg(rel) / ideale if ideale > 0 else 0.0


def ild(G: np.ndarray) -> float:
    """Diversità intra-lista: 1 − somiglianza media tra le coppie di item consigliati (caratteristiche grezze)."""
    n = len(G)
    if n < 2:
        return 0.0
    S = G @ G.T
    return float(1 - (S.sum() - np.trace(S)) / (n * (n - 1)))


def gini(conteggi: np.ndarray) -> float:
    """Concentrazione dell'esposizione: 0 = tutti gli item consigliati ugualmente, 1 = sempre gli stessi."""
    x = np.sort(np.asarray(conteggi, dtype=float))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def metriche_lista(cat: Catalogo, idx: np.ndarray, stile_q: int, fam_q: int, k: int, cat_q: str | None = None) -> dict:
    """Metriche di una lista. Rilevante (P, R, MRR) = stesso stile; «stile+tipo» = anche stesso tipo di edificio."""
    rel = rilevanza_graduata(cat, idx, stile_q, fam_q, cat_q)
    stesso = cat.stile == stile_q
    n_tipo = int((stesso & (cat.categoria == cat_q)).sum()) if cat_q is not None else 0
    n_stile, n_fam = int(stesso.sum()) - n_tipo, int(((cat.famiglia == fam_q) & ~stesso).sum())
    alto = 3 if cat_q is not None else 2
    ideale = np.r_[np.full(n_tipo, alto), np.full(n_stile, 2), np.full(n_fam, 1)][:k]
    hit = rel >= 2
    primo = np.flatnonzero(hit)
    out = {f"P@{k}": float(hit.sum() / k),
           f"R@{k}": float(hit.sum() / max(int(stesso.sum()), 1)),
           f"famiglia@{k}": float((rel >= 1).sum() / k),
           f"nDCG@{k}": ndcg(rel, ideale),
           "MRR": float(1 / (primo[0] + 1)) if len(primo) else 0.0,
           "ILD": ild(cat.G[idx]),
           "Paesi diversi": float(len(set(cat.paese[idx].tolist())) / max(len(idx), 1))}
    if cat_q is not None:
        out[f"P@{k} stile+tipo"] = float((rel == 3).sum() / k)
    return out


def valuta(cat: Catalogo, Hq: np.ndarray, stile_vero: np.ndarray, s2f: np.ndarray, sistema, k: int = 5,
           cat_vera: np.ndarray | None = None, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """Applica `sistema(i, h, rng) -> indici` a ogni query; restituisce metriche per query ed esposizione degli item.
    cat_vera: categoria del tipo di edificio di ogni query (attiva la rilevanza «stile + tipo»)."""
    rng = np.random.default_rng(seed)
    righe, esposizione = [], np.zeros(len(cat), dtype=int)
    for i, h in enumerate(Hq):
        idx = np.asarray(sistema(i, h, rng))[:k]
        np.add.at(esposizione, idx, 1)
        cq = None if cat_vera is None or str(cat_vera[i]) == "altro" else str(cat_vera[i])   # tipo ignoto
        righe.append(metriche_lista(cat, idx, int(stile_vero[i]), int(s2f[stile_vero[i]]), k, cq))
    return pd.DataFrame(righe), esposizione


def riepilogo(per_query: pd.DataFrame, esposizione: np.ndarray) -> dict:
    out = per_query.mean().to_dict()
    out["copertura catalogo"] = float((esposizione > 0).mean())
    out["Gini esposizione"] = gini(esposizione)
    return out
