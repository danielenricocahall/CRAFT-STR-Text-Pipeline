# CRAFT-STR-Text-Pipeline

This is a simple repository which just combines two awesome pieces of work by CLOVA AI Research (https://github.com/clovaai) into one simple script to perform text detection + recognition.

## Instructions

### Clone the repository
`git clone git@github.com:danielenricocahall/CRAFT-STR-Text-Pipeline.git`


### Run the configure script
`. configure.sh`

The configure script will clone the submodules (https://github.com/clovaai/deep-text-recognition-benchmark and https://github.com/clovaai/CRAFT-pytorch), download the models, and set up/launch a virtual Python environment. 

### Run

After running the configure script, run `python pipeline.py`, along with any arguments you may want to supply - by default, the data folder is `./data/`. To supply your own data directory, just use the `--data` flag i.e; `python pipeline --data DATA_DIRECTORY_HERE`. By default, the results directory is `./results/`, but that can be changed using the `--result_dir` flag.

## Examples

To play around, just pick some text detection datasets from this cool repository: https://github.com/cs-chan/Total-Text-Dataset.

## Contact

If you have any questions or concerns, feel free to contact me at danielenricocahall@gmail.com, or just submit a git issue.

## Contribution

This was a bit hastily thrown together but I wanted to get it out there in case it could benefit others. Feel free to submit PRs!
