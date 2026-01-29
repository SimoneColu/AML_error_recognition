#  EgoVLP Feature Extraction Pipeline (Kaggle Edition)

Questo repository contiene una pipeline ottimizzata per l'estrazione di feature video utilizzando **EgoVLP** in un ambiente Kaggle con accelerazione GPU (T4 x2).

Il flusso di lavoro è diviso in tre fasi principali: Setup, Estrazione Parallela e Proiezione delle Feature.

---

## 1. Setup dell'Ambiente (`setup.ipynb`)

Questa è la fase di inizializzazione critica. Lo script (o la prima cella del notebook) prepara l'ambiente di esecuzione effimero di Kaggle.

**Cosa fa esattamente:**
1.  **Installazione Dipendenze:** Installa librerie non presenti di default su Kaggle ma necessarie per il video processing, tra cui:
    * `pytorchvideo` (manipolazione video)
    * `timm`, `transformers` (architetture deep learning)
    * `gdown` (download da Google Drive)
2.  **Clonazione Repository:** Scarica il codice sorgente di EgoVLP e gli script di utilità.
3.  **Download Asset Pesanti:** Scarica automaticamente da Google Drive:
    *  **Pesi del Modello (`egovlp.pth`):** Il checkpoint pre-addestrato necessario per l'inferenza.
    *  **Dataset Video:** L'archivio `.zip` contenente i video grezzi da analizzare. Abbiamo usato un dataset più piccolo di quello originale per semplicità.

> **Nota:** Senza questa fase, gli script successivi falliranno per mancanza di librerie o file.

---

## 2. Estrazione Feature Parallela
**File:** `segment_feature_extractor_kaggle_parallelize.py`

Questo è il "motore" principale della pipeline. È stato riscritto per sfruttare il **multi-GPU** di Kaggle (2x NVIDIA T4).

**Funzionamento:**
* **Parallelismo:** Rileva il numero di GPU disponibili e divide la lista dei video in due "chunk". Ogni GPU processa metà dei video indipendentemente (evitando colli di bottiglia di `DataParallel`).
* **Segmentazione:** Taglia i video in segmenti temporali (finestre) e per ogni segmento estrae un vettore di feature.
* **Robustezza:** Include gestione degli errori (skip file corrotti) e logging separato per ogni worker GPU.
* **Output:** Salva i file `.npz` contenenti le feature grezze estratte dal backbone video.

---

## 3. Proiezione Feature (Spazio Latente)
**File:** `project_feature.py`

**A cosa serve:**
Questo script si occupa dell'allineamento dimensionale delle feature.

* **Input:** Feature video grezze dimensione **768** (output standard del backbone video di EgoVLP).
* **Operazione:** Applica una proiezione lineare (o MLP) appresa durante il training di EgoVLP.
* **Output:** Feature video dimensione **256**.
* **Perché è necessario?**
    Il modello EgoVLP è addestrato per confrontare video e testo.
    * Il testo viene codificato in uno spazio a 256 dimensioni.
    * Il video esce dal backbone a 768 dimensioni.
    * Questo script "comprime" e **allinea** le feature video nello stesso spazio matematico del testo (256 dim), permettendo di calcolare la similarità (es. cosine similarity) per task di retrieval o classificazione zero-shot.

---

##  Istruzioni di Esecuzione

1.  **Esegui il Setup:** Lancia la cella di installazione/download. Assicurati di decomprimere i file scaricati (lo zip dei pesi e lo zip del dataset).
2.  **Estrai le Feature:**
    ```bash
    python segment_feature_extractor_kaggle_parallelize.py --backbone egovlp
    ```
3.  **Proietta le Feature (Opzionale ma raccomandato per text-matching):**
    ```bash
    python project_feature.py --input_dir /path/to/768_features --output_dir /path/to/256_features
    ```