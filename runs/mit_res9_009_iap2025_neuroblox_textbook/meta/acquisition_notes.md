# Acquisition Notes

This run is in course mode. The official MIT OCW page points to the Neuroblox course home, and the Neuroblox site contains the actual lecture pages and the public YouTube embeds.

## Official Playlist Resolution

The MIT Video Productions channel does expose a public YouTube playlist that contains the full Neuroblox lecture set:

- `IAP 2025`: `https://www.youtube.com/playlist?list=PLKHPCGvTwsmGEplF2c_Y8WEZ-Woqx-dlN`

This is not a Neuroblox-only playlist; it is a broader MIT IAP 2025 channel playlist. However, it contains all 11 numbered `IntroCompNeuroNeuroblox` lecture videos, so it is the correct public YouTube list to record for this run.

## Resolved Video Set

There are 13 public YouTube videos referenced on the official course site:

- 11 numbered lecture videos, `1` through `11`
- 2 supplementary Julia reference videos linked from the `Introduction to Julia` page

The two supplementary videos are official page links rather than lesson iframes, but they are still public YouTube sources exposed by the course site and should remain in the acquisition record.

The lecture pages with public YouTube content are:

- `Introduction to Computational Neuroscience with Neuroblox` -> videos `2dbAePEmbhQ`, `XlBRJps84zE`
- `Introduction to Julia` -> video `ekwu47RHCHE`, plus linked references `Fi7Pf2NveH0` and `7hVV5uoEo-0`
- `Differential Equations with ModelingToolkit` -> `6EaolLVhnug`
- `Blox and Connections in Neuroblox` -> `8XcN9j5njgg`
- `Neurons, Neural Masses and Sources` -> `Ptqv16fhOtg`
- `Circuit Models in Neuroblox` -> `ih7IELQ5W50`
- `Decision Making in a Circuit Model` -> `ULAe2VvQgms`
- `Synaptic Plasticity and Reinforcement Learning` -> `pBvgcIHK6GY`
- `Parameter Fitting using Spectral Dynamic Causal Modeling` -> `OEeyks_HIMI`
- `Experimental Design` -> `NU9K8l-gg-Q`

All 11 numbered lecture videos above were also confirmed to appear in the MIT Video Productions `IAP 2025` playlist.

Pages crawled that did not expose a public YouTube video on the page source were:

- `Getting Started`
- `Plotting with Makie`
- `Biomimetic Model of Corticostriatal Microassemblies`
- `Pyramidal-Interneuron Gamma Network`
- `Parameter Fitting using Optimization`

## Subtitle and Thumbnail State

- The 11 lecture videos all have manual English subtitles available through `yt-dlp`.
- Subtitle language labels are a mix of `en` and `en-US`.
- The 2 supplementary Julia videos linked from `Introduction to Julia` do not expose a manual subtitle track in `yt-dlp`; they only show automatic captions.
- Thumbnail URLs resolve from YouTube metadata, usually as `maxresdefault.jpg`; one supplementary video falls back to `sddefault.jpg`.

## Local Acquisition Status

- The first 3 lecture packages already exist in `raw/videos/01_2dbAePEmbhQ`, `raw/videos/02_XlBRJps84zE`, and `raw/videos/03_ekwu47RHCHE`.
- Those existing packages already contain `*.info.json`, `*.webp`, and subtitle files.
- No large MP4 downloads were performed for this inventory pass.

## Practical Caveats

- The course lecture videos were straightforward to resolve with metadata-only `yt-dlp` calls, so there is no reason to download the full media files yet.
- If later transcript or figure work needs local assets, fetch subtitles and thumbnails first, then defer video downloads until frame extraction is actually required.
- The two supplementary Julia reference videos are optional for the textbook run unless a later coverage pass explicitly decides they are needed.
