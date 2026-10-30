# Evaluation dataset

`babe_sg2.csv` is the SG2 subset of **BABE (Bias Annotations By Experts)**:
3,673 sentences from US news articles, each labelled `Biased` or `Non-biased`
by majority vote of 5 expert annotators (1 sentence with "No agreement" removed).
Columns kept: text, outlet, topic, type (outlet political orientation), label_bias.

Source: https://github.com/Media-Bias-Group/Neural-Media-Bias-Detection-Using-Distant-Supervision-With-BABE
(file `data/final_labels_SG2.csv`)

Citation:
T. Spinde, M. Plank, J.-D. Krieger, T. Ruas, B. Gipp, A. Aizawa.
"Neural Media Bias Detection Using Distant Supervision With BABE - Bias Annotations By Experts."
Findings of EMNLP 2021.

Used here only to evaluate the bias detector (`python -m scripts.evaluate`).
Note: BABE contains article sentences, not headlines, so it is a related but
not identical domain to the RSS headlines this project collects.
