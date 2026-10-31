# Cross-Genre Melodic Similarity Evaluation Report

Evaluation of melodic resemblance across Indian Film (Bollywood), Western Pop, Blues, and Atonal audio.

| Test Track | Genre | Expected Raga Resemblance | Top Predicted Match | Similarity Score | Confidence | Abstention Status |
| :--- | :--- | :--- | :--- | :---: | :--- | :---: |
| **Bollywood (Bhairavi Melody)** | Indian Film / Bollywood | `Bhairavi` | **Bhairavi** | `0.863` | High | No |
| **Bollywood (Yaman Ballad)** | Indian Film / Bollywood | `Yaman` | **Yaman** | `0.852` | High | No |
| **Western Pop (Major Pentatonic)** | Western Pop / Folk | `Bhoopali` | **Bhoopali** | `0.918` | High | No |
| **Western Blues (Minor Pentatonic)** | Western Blues / Rock | `Malkauns` | **Malkauns** | `0.953` | High | No |
| **Atonal Noise Control** | Unpitched Control | `Abstain` | **None** | `0.000` | N/A | Yes (Correct) |


## Qualitative Analysis
1. **Bollywood / Film Music**: High similarity achieved on modal scale alignments (e.g. Bhairavi, Yaman) with explicit note-by-note evidence.
2. **Western Pop / Folk**: Major pentatonic melodies naturally map to Raga *Bhoopali* (`S R G P D`), illustrating cross-cultural modal equivalence.
3. **Western Blues / Rock**: Minor pentatonic contours map to Raga *Malkauns* (`S g M d n`), reflecting shared pitch intervals without false assertions of classical improvisation.
4. **Abstention Gate**: Unpitched audio and atonal noise are rejected cleanly by the multi-criterion gate, preventing spurious raga attribution.
