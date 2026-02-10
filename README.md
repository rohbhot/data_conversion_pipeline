Installation:
Recommended python virtual env
1. python -m venv data_env
2. cd data_conversion_pipeline
3. pip install requirements.txt

Usage:
python split_data.py --input-folder <path_to_input_folder> --output-folder <path_to_output_folder> --copy-images --split-percentage <train><val><test>


Example:
1. python split_data.py --input-folder /home/thor/Documents/Anand/data_conversion_pipeline/img/ --output-folder /home/thor/Documents/Anand/data_conversion_pipeline/out/ --copy-images --split-percentages 0.8 0.1 0.1 
2. python split_data.py --input-folder /home/thor/Documents/Anand/data_conversion_pipeline/img/ --output-folder /home/thor/Documents/Anand/data_conversion_pipeline/out/ --copy-images --split-percentages 0.8 0.1 0.1 --aug-type
3. python split_data.py --input-folder /home/thor/Documents/khundmeer/light_scratches/light_scratches/ --output-folder /home/thor/Documents/khundmeer/light_scratches/coco1 --copy-images --split-percentages 0.8 0.1 0.1 --aug-type --augment-size 5
