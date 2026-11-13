"""Logica dell'app MirAi, separata dall'interfaccia (si può provare da un notebook senza Gradio).

Flusso di una foto:
1. `prepara`: orientamento EXIF, lato corto 512 px, ritaglio centrale 448 × 448;
2. `analizza` (≈ 1 s): caratteristiche CLIP + DINOv2 (media foto originale + ribaltata, come nel test), testa MLP,
   risposta gerarchica, affidabilità, controlli (fuori dominio, disegno), posizione sulla SOM e **monumenti simili
   consigliati** (sistema di raccomandazione content-based, `src/raccomandazione.py`, notebook 09);
3. `spiega` (≈ 20 s): LIME sul modello definitivo e nomi degli elementi (xai.spiega);
4. `altri`: seconda pagina di consigli con il relevance feedback di Rocchio sulle preferenze dell'utente;
5. `salva_riscontro` / `salva_preferenze`: correzioni dello stile e preferenze sui consigli su Drive (le preferenze
   sono il primo storico di voti utente-item, utilizzabile in futuro per un Collaborative Filtering).
"""
from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import json

import numpy as np
from PIL import Image, ImageOps

from src import config, hierarchy as hi, raccomandazione as rc, xai

# parametri del sistema di raccomandazione: scelti sul validation nel notebook 09 (valori di riserva se manca il file)
PARAMETRI_RACC = "reports/metrics/09/raccomandazione.json"


def prepara(foto, size: int = config.IMG_SIZE_HR, short: int = config.IMG_SHORT_SIDE_HR) -> np.ndarray:
    """PIL o numpy → numpy uint8 448 × 448 (stessa geometria di xai.load_square)."""
    im = foto if isinstance(foto, Image.Image) else Image.fromarray(np.asarray(foto).astype(np.uint8))
    im = ImageOps.exif_transpose(im).convert("RGB")
    s = short / min(im.size)
    im = im.resize((max(size, round(im.width * s)), max(size, round(im.height * s))), Image.LANCZOS)
    l, t = (im.width - size) // 2, (im.height - size) // 2
    return np.asarray(im.crop((l, t, l + size, t + size)))


def _norm(A: np.ndarray) -> np.ndarray:
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)


class MiraiApp:
    """fm: xai.FinalModel; namer: xai.ElementNamer (o None);
    db: dict con F (caratteristiche di training + validation del notebook 03f), path, label;
    is_train: maschera delle righe di training (riferimento per il filtro fuori dominio);
    som: contenuto di som_finale.npz (o None); nomi: {path: nome dell'edificio}; info: modello_finale.json;
    meta: tabella delle foto con path, church_id, church_name, country, type (per raggruppare le foto in edifici;
    se manca, ogni foto è un edificio)."""

    def __init__(self, fm, namer, db: dict, is_train: np.ndarray, som: dict | None, nomi: dict, info: dict,
                 img_root: str = config.COLAB_PROCESSED_DIR_HR, dir_riscontri: str | None = None, k: int = config.KNN_K,
                 meta=None):
        import torch
        self.fm, self.namer, self.som, self.nomi, self.k = fm, namer, som, nomi, k
        self.img_root = img_root
        self.soglia_stile = float(info.get("soglia_stile", 0.80))
        self.soglia_fam = float(info.get("soglia_famiglia", 0.60))
        F = np.asarray(db["F"], dtype=np.float32)
        self.path = np.asarray(db["path"]).astype(str)
        self.label = np.asarray(db["label"]).astype(int)
        self.foto_intere = ~np.char.startswith(self.path, "elementi/")      # niente ritagli tra gli edifici simili
        # spazio della testa (512 attivazioni dello strato nascosto): edifici simili
        self._torch = torch
        self.H = _norm(self._testa(F))
        # filtro fuori dominio (notebook 04b): distanza media dai 5 vicini di training, caratteristiche standardizzate
        if som is not None:
            self.mu, self.sd = np.asarray(som["mu"]), np.asarray(som["sd"])
        else:
            self.mu, self.sd = F[is_train].mean(0), F[is_train].std(0) + 1e-6
        self.D_tr = _norm((F[is_train] - self.mu) / self.sd).astype(np.float32)
        va = _norm((F[~is_train] - self.mu) / self.sd).astype(np.float32)
        self.soglia_ood = float(np.quantile(self._dist_vicini(va), 0.95)) if len(va) else 1.0
        self.dir_riscontri = Path(dir_riscontri) if dir_riscontri else None
        # catalogo del sistema di raccomandazione: edifici (foto intere di training + validation)
        import pandas as pd
        m = pd.DataFrame({"path": self.path, "label": self.label})
        if meta is not None:
            m = m.merge(meta.drop(columns=["label"], errors="ignore").drop_duplicates("path"), on="path", how="left")
        for c in ("church_id", "church_name", "country", "type"):
            if c not in m:
                m[c] = np.nan
        m["church_id"] = m["church_id"].fillna(m["path"])
        m["church_name"] = m["church_name"].fillna(m["path"].map(nomi)).fillna("")
        m[["country", "type"]] = m[["country", "type"]].fillna("")
        intere = self.foto_intere
        self.cat = rc.Catalogo.da_foto(self.H[intere], self._grezze(F[intere]), m[intere], hi.style_to_family())
        par = json.loads(Path(PARAMETRI_RACC).read_text()) if Path(PARAMETRI_RACC).exists() else {}
        self.peso_testa = float(par.get("peso_testa", 1.0))
        self.lam = float(par.get("lambda", 0.7))

    # ------------------------------------------------------------------
    def _testa(self, F: np.ndarray, batch: int = 4096) -> np.ndarray:
        torch, head = self._torch, self.fm.head
        out = []
        with torch.no_grad():
            for i in range(0, len(F), batch):
                x = torch.tensor(self.fm.std(F[i:i + batch]).astype(np.float32))
                out.append(torch.relu(head[0](x)).numpy() if hasattr(head, "__getitem__") else x.numpy())
        return np.concatenate(out)

    def _dist_vicini(self, Q: np.ndarray, batch: int = 512) -> np.ndarray:
        d = []
        for i in range(0, len(Q), batch):
            S = Q[i:i + batch] @ self.D_tr.T
            d.append(1 - np.sort(S, axis=1)[:, -self.k:].mean(1))
        return np.concatenate(d)

    def _grezze(self, F: np.ndarray) -> np.ndarray:
        """Caratteristiche visive grezze standardizzate e normalizzate (diversità e somiglianza combinata)."""
        return _norm((F - self.mu) / self.sd).astype(np.float32)

    def _consigli(self, R: dict, h: np.ndarray, g: np.ndarray, stile: int | None, escludi=None) -> list[dict]:
        idx, sim = rc.raccomanda(self.cat, h, self.k, stile=stile, lam=self.lam, g=g, peso_testa=self.peso_testa,
                                 escludi=escludi)
        from src import descrizioni as D
        def perche(j, s, lang):
            return rc.spiegazione(self.cat, j, s, R["stile"], R["famiglia"],
                                  [D.nome_stile(c, lang) for c in config.CLASSES],
                                  [D.nome_famiglia(f, lang) for f in config.FAMIGLIE], lang)
        return [{"i": int(j), "path": f"{self.img_root}/{self.cat.foto[j]}", "nome": str(self.cat.nome[j]),
                 "stile": int(self.cat.stile[j]), "sim": float(s),
                 "perche": {lang: perche(j, s, lang) for lang in ("it", "en")}}
                for j, s in zip(idx, sim)]

    def caratteristiche(self, img: np.ndarray) -> np.ndarray:
        """Media delle caratteristiche di foto originale e ribaltata (come validation e test del 03f)."""
        F = self.fm.features([img, np.ascontiguousarray(img[:, ::-1])])
        return F.mean(0, keepdims=True).astype(np.float32)

    # ------------------------------------------------------------------
    def analizza(self, foto) -> dict:
        img = prepara(foto)
        F = self.caratteristiche(img)
        p = self.fm.fd.predict_proba(self.fm.head, self.fm.std(F).astype(np.float32))[0]
        fam, stile, pf = hi.predict(p[None])
        fam, stile, pf = int(fam[0]), int(stile[0]), pf[0]
        ordine = [int(j) for j in np.argsort(-p)]
        rivale = next(j for j in ordine if j != stile)
        # controlli
        d_ood = float(self._dist_vicini(_norm((F - self.mu) / self.sd).astype(np.float32))[0])
        p_disegno = float(xai.prob_non_foto(self.fm, self.fm.image_embed([img]))[0])
        # monumenti simili consigliati: edifici dello stile predetto, somiglianza combinata e MMR (notebook 09)
        h, g = _norm(self._testa(F))[0], self._grezze(F)[0]
        # mappa di Kohonen
        bmu = None
        if self.som is not None:
            z = (((F - self.som["mu"]) / self.som["sd"]) - self.som["centro"]) @ self.som["pca"].T
            bmu = int(((np.asarray(self.som["W"]) - z) ** 2).sum(1).argmin())
        out = {"img": img, "F": F, "p": p, "pf": pf, "famiglia": fam, "stile": stile, "rivale": rivale,
                "conf_stile": float(p.max()), "conf_famiglia": float(pf.max()),
                "sicura_stile": bool(p.max() >= self.soglia_stile), "sicura_famiglia": bool(pf.max() >= self.soglia_fam),
                "d_ood": d_ood, "fuori_dominio": bool(d_ood > self.soglia_ood),
                "p_disegno": p_disegno, "disegno": bool(p_disegno >= xai.SOGLIA_NON_FOTO),
                "bmu": bmu, "q_h": h, "q_g": g}
        out["simili"] = self._consigli(out, h, g, stile)
        out["mostrati"] = [c["i"] for c in out["simili"]]
        return out

    def altri(self, R: dict, graditi: list[int]) -> list[dict]:
        """Nuovi consigli con il relevance feedback di Rocchio: graditi = indici (nel catalogo) degli edifici scelti
        dall'utente tra quelli mostrati; gli altri mostrati contano come non graditi. Aggiorna R["simili"]."""
        visti = np.array(R["mostrati"], dtype=int)
        si = np.isin(visti, graditi)
        h = rc.rocchio(R["q_h"], self.cat.H[visti[si]], self.cat.H[visti[~si]])
        g = rc.rocchio(R["q_g"], self.cat.G[visti[si]], self.cat.G[visti[~si]])
        stile = int(np.bincount(self.cat.stile[visti[si]]).argmax()) if si.any() else None
        escludi = np.zeros(len(self.cat), dtype=bool)
        escludi[visti] = True
        R["q_h"], R["q_g"] = h, g
        R["simili"] = self._consigli(R, h, g, stile, escludi)
        R["mostrati"] = list(visti) + [c["i"] for c in R["simili"]]
        return R["simili"]

    def salva_preferenze(self, R: dict, graditi: list[int], lingua: str) -> Path | None:
        """Una riga per edificio mostrato: 1 = gradito, 0 = non gradito (feedback esplicito utente-item)."""
        if self.dir_riscontri is None:
            return None
        self.dir_riscontri.mkdir(parents=True, exist_ok=True)
        f = self.dir_riscontri / "preferenze.csv"
        nuovo = not f.exists()
        ora = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f, "a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if nuovo:
                w.writerow(["data", "sessione", "lingua", "stile_predetto", "edificio", "stile_edificio", "gradito"])
            for c in R["simili"]:
                w.writerow([ora, R.setdefault("sessione", f"{ora}_{np.random.randint(1e6):06d}"), lingua,
                            config.CLASSES[R["stile"]], self.cat.edificio[c["i"]], config.CLASSES[c["stile"]],
                            int(c["i"] in graditi)])
        return f

    def spiega(self, R: dict, num_samples: int = 600, progress=None) -> dict:
        return xai.spiega(self.fm, R["img"], self.namer, num_samples=num_samples, p=R["p"], progress=progress)

    # ------------------------------------------------------------------
    def salva_riscontro(self, R: dict, stile_corretto: str, commento: str, lingua: str) -> Path | None:
        if self.dir_riscontri is None:
            return None
        d = self.dir_riscontri
        (d / "foto").mkdir(parents=True, exist_ok=True)
        ora = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        nome = f"{ora}_{np.random.randint(1e6):06d}.jpg"
        Image.fromarray(R["img"]).save(d / "foto" / nome, quality=90)
        f = d / "riscontri.csv"
        nuovo = not f.exists()
        with open(f, "a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if nuovo:
                w.writerow(["data", "foto", "lingua", "famiglia_predetta", "stile_predetto", "confidenza_stile",
                            "stile_corretto", "commento", "fuori_dominio", "disegno", "verificato"])
            w.writerow([ora, nome, lingua, list(config.FAMIGLIE)[R["famiglia"]], config.CLASSES[R["stile"]],
                        round(R["conf_stile"], 3), stile_corretto, commento.replace("\n", " ").strip(),
                        R["fuori_dominio"], R["disegno"], ""])
        return f
