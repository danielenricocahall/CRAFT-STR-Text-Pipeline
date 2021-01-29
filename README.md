# CRAFT-STR-Text-Pipeline

This is a simple repository which just combines two awesome pieces of work by CLOVA AI Research (https://github.com/clovaai),  Character Region Awareness for Text Detection (CRAFT) and Scene Text Recognition (STR) into one simple script to perform text detection + recognition.

## Instructions

### Clone the repository
`git clone git@github.com:danielenricocahall/CRAFT-STR-Text-Pipeline.git`


### Run the configure script
`. configure.sh`

The configure script will clone the submodules (https://github.com/clovaai/deep-text-recognition-benchmark and https://github.com/clovaai/CRAFT-pytorch), download the pretrained models, and set up/launch a virtual Python environment. 

### Run

After running the configure script, run `python pipeline.py`, along with any arguments you may want to supply - by default, the data folder is `./data/`. To supply your own data directory, just use the `--data` flag i.e; `python pipeline --data DATA_DIRECTORY_HERE`. By default, the results directory is `./results/`, but that can be changed using the `--result_dir` flag. The pretrained model used for CRAFT can be configured as well, using the `--trained_str_model` flag. Many more configurations as well - just check them out in `pipeline.py`.
## Examples

To play around, just pick some text detection datasets from this cool repository: https://github.com/cs-chan/Total-Text-Dataset.

## Contact

If you have any questions or concerns, feel free to contact me at danielenricocahall@gmail.com, or just submit a git issue.

## Contribution

This was a bit hastily thrown together but I wanted to get it out there in case it could benefit others. Feel free to submit PRs!

## TODO

- Refactor to permit supplying individual images
- Refactor for performance
- Refactor to follow nice development practices rather than a hacky script 
- Maybe dockerize this and enable it to be used as service? :)
