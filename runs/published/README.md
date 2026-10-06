# Published experiment reports

Portable snapshots of the completed local runs. The original runs remain on the
training laptop; saved weights are not included here. The trained WavLM bundle is
at https://huggingface.co/saatweek/wavlm-ravdess-emotion.

## Start here

- [WavLM test results](wavlm_comparison/report.html)
- [Five-candidate validation comparison and WavLM test actors](wavlm_comparison/comparison.html)
- [Original CNN test results](final/report.html)
- [Original four-candidate comparison](final/comparison.html)
- [Dataset audit](analysis/dataset_report.html)
- [Audio feature illustrations](analysis/audio_features.html)
- [Fourier teaching example](analysis/fourier.html)

GitHub shows HTML as source. Clone/download the repository, then open the HTML
files in a browser. Keep the _assets folder beside the run folders: the reports
share one local Plotly library and work offline without a CDN or Python server.
The long sample-audio feature illustration is much larger than the other charts.

Each training folder includes configuration, history, split manifest, validation
metrics and report. final/ and wavlm_comparison/ additionally include selection,
test metrics, per-recording prediction CSVs and actor comparisons. prediction*
and wavlm_prediction* folders contain unlabeled functionality examples, not new
accuracy experiments. CPU/CUDA scores may differ slightly.

## Portable metadata

Local dataset paths are replaced with DATASET_ROOT/Actor_XX/filename.wav. In
configuration and other metadata, PROJECT_ROOT and HOME_ROOT denote local paths.
These placeholders are documentation, not existing directories. To evaluate from
a published splits.json, first replace DATASET_ROOT with your extracted dataset
directory and supply a downloaded/trained checkpoint. Do not overwrite your
operational local run directories with these snapshots.

No audio recordings, checkpoints, encoder weights, tokens, caches or environment
files are copied into this folder. index.json records original/published hashes
and the export transformations. Metric values and prediction ordering are kept;
only filesystem paths and the placement of Plotly JavaScript are changed.
