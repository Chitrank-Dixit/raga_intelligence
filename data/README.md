# Indian Classical Music Datasets & Governance

## 1. Candidate Research Datasets

### A. Saraga Hindustani Dataset
- **Source URL**: [https://doi.org/10.5281/zenodo.4301737](https://doi.org/10.5281/zenodo.4301737)
- **Institution**: Music Technology Group (MTG), Universitat Pompeu Fabra (Barcelona)
- **Musical Tradition**: Hindustani classical music (primarily vocal, khyal style)
- **Audio & Annotations**:
  - 108 recordings (~108 hours total duration)
  - 44.1 kHz, 16-bit stereo MP3/WAV format
  - Pitch contours extracted and verified (predominant melody)
  - Ground truth tonic (fundamental frequency $S_a$) for every performance
  - Time-aligned section boundaries (Alap, Vilambit, Drut) and Sama markers
- **Raga Representation**: Over 30 ragas represented across prominent artists and gharanas
- **Label Quality**: Gold-standard research quality; manually curated by Indian musicologists
- **Licensing & Redistribution**: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). Freely usable for non-commercial research; redistribution permitted under identical license terms.
- **Suitability**: Excellent for raga representation learning, tonic normalization, and baseline evaluation.

---

### B. CompMusic IAM Raga Recognition Dataset (Features & Audio)
- **Source URL**: 
  - Features: [https://doi.org/10.5281/zenodo.7278505](https://doi.org/10.5281/zenodo.7278505)
  - Audio: [https://doi.org/10.5281/zenodo.7278510](https://doi.org/10.5281/zenodo.7278510)
- **Institution**: MTG-UPF
- **Traditions**: Hindustani (HMD) and Carnatic (CMD)
- **Recordings & Classes**:
  - HMD contains 300+ recordings spanning 30+ Hindustani ragas
- **Access Restrictions**: Audio is restricted access due to commercial music labels; features (pitch tracks, tonic, metadata) are open access under CC BY-NC-ND 4.0.
- **Suitability**: Features dataset is directly usable for pitch-contour and swara histogram baselines; raw audio requires formal access request.

---

### C. Hindustani Raga Phrase Dataset (IIT Bombay / TIFR)
- **Source URL**: [https://github.com/kaustuvkanti/hindustani_raga_phrase_dataset](https://github.com/kaustuvkanti/hindustani_raga_phrase_dataset)
- **Tradition**: Hindustani vocal (Khyal)
- **Ragas**: Focused on paired ragas with identical swara material but distinct movement (e.g., *Bhoopali* vs. *Deshkar*)
- **Licensing**: Academic research open-source
- **Suitability**: High-precision evaluation of melodic phrase dynamics (pakads, prayogas).

---

## 2. Selected Core Hindustani Ragas (Initial Cohort: 10 Ragas)

Based on frequency of occurrence in research datasets and structural diversity across Thhats (parent scales):

| Raga | Thhat | Arohana (Ascent) | Avarohana (Descent) | Vadi / Samvadi | Mood / Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Bhairav** | Bhairav | S r G M P d N S' | S' N d P M G r S | d / r | Dawn, Devotional |
| **Bhairavi** | Bhairavi | S r g M P d n S' | S' n d P M g r S | M / S | Morning/Anytime, Pathos |
| **Yaman** | Kalyan | 'N R G M' D N S' | S' N D P M' G R S | G / N | Evening, Romance/Serenity |
| **Bhoopali** | Kalyan | S R G P D S' | S' D P G R S | G / D | Early Night, Peace |
| **Deshkar** | Bilawal | S R G P D S' | S' D P G R S | D / G | Morning, Bright |
| **Bageshree** | Kafi | S g M D n S' | S' n D M g R S | M / S | Late Night, Longing |
| **Malkauns** | Bhairavi | S g M d n S' | S' n d M g S | M / S | Midnight, Meditative |
| **Todi (Miyan ki Todi)** | Todi | S r g M' P d N S' | S' N d P M' g r S | d / g | Late Morning, Intense |
| **Darbari Kanada** | Asavari | S R g M P d n S' | S' d n P M g M R S | R / P | Midnight, Royal/Gravity |
| **Kafi** | Kafi | S R g M P D n S' | S' n D P M g R S | P / S | Spring / Late Evening, Joyful |

---

## 3. Data Leakage Prevention Strategy

1. **Split Entity**: Splits are strictly generated at the **recording / performance and artist** level.
2. **Audio Windowing**: Temporal slicing (e.g., 5-second overlapping frames) occurs *strictly after* the train/validation/test split has partitioned whole recordings.
3. **No Performer Overlap in Test**: For classes with multiple artists, test sets prioritize artists unseen during training to evaluate true generalization to varied vocal timbres and personal phrasing styles.

