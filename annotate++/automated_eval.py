import json
import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm 
import utils
from demo import *


class Evaluate():
    def __init__(self, config):
        self.config = config
        self.mode = self.config['mode']
        if self.mode not in ['auto_annotate', 'generate_ground_truth', 'evaluate']:
            raise Exception('This mode is not supported!')
        self.framework = self.config['framework']
        self.model_config = self.config['model_config']
        
        # if self.mode in ['auto_annotate', 'evaluate']:
        self.thresholds = self.config['thresholds']
        if len(self.thresholds) == 0:
            raise Exception('Please provide thresholds in the configuration file')

        if self.framework not in ['nanodet', 'mmdetection']:
            raise Exception('This framework is not supported yet')

        self.model, self.cfg = utils.load_model(self.framework, self.model_config, self.thresholds)
        if not utils.check_thresholds_and_classes(self.model, self.framework, self.thresholds, self.cfg):
            raise Exception('The number of thresholds and classes from model\'s configuration file do not match!!')
            
        if self.framework == 'nanodet':
            self._classes = self.cfg.class_names
        # elif self.framework == 'mmdetection':
        #     self._classes = list(self.model._classes)
            
        
        self.dataset_path = self.config['dataset_path']
        self.output_path = self.config['output_path']
        
    def run(self):
        if self.mode == 'generate_ground_truth':
            df = utils.create_ground_truths(self.dataset_path, self._classes)
            df = df.drop(['dummy'], axis = 1)
            df.to_csv(os.path.join(self.output_path, 'ground_truths.csv'), index = None)

        if self.mode in ['auto_annotate', 'evaluate']:
            scores = list()
            for _file in tqdm(os.listdir(self.dataset_path)):
                if _file.endswith(('.jpg', '.bmp', '.BMP')):
                    #print(f'Processing file : {_file}')
                    if self.framework == 'nanodet':
                        meta, res = self.model.inference(os.path.join(self.dataset_path, _file))
                        
                        out= dict(zip(self.cfg.class_names, list(res[0].values())))
                        result = utils.nanodet_filter_boxes(res, self.thresholds)
                        result1=result
                        sc= utils.append_score(out, self.cfg.class_names)
                        scores.append({_file:sc})
                        
                    # elif self.framework == 'mmdetection':
                    #     image = Image.open(os.path.join(self.dataset_path, _file))
                    #     output_image = np.array(image)
                    #     result = self.model.show_inference(np.array(image.convert("RGB")))
                    #     result_boxes = result['result_boxes']
                    #     scores.append({_file : [{list(result_boxes.values())[x] : [self.model.scores[x]]} for x in range(len(self.model.scores))]})
                    if self.framework == 'nanodet':
                        defect_list = utils.create_xmls(self.framework, result, os.path.join(self.dataset_path, _file), self.cfg.class_names, meta)
                    else:
                        defect_list = utils.create_xmls(self.framework, result_boxes, os.path.join(self.dataset_path, _file))
                    if self.mode == 'evaluate':
                        # if self.framework == 'mmdetection':
                        #     for boxes, defect in result_boxes.items():
                        #         x0, y0, x1, y1 = boxes
                        #         cv2.rectangle(output_image, (int(x0),int(y0)), (int(x1),int(y1)), (255,0,0), 4)
                        #         cv2.putText(output_image, str(defect), (int(x0),int(y0)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220),3)
                        #     cv2.imwrite(os.path.join(self.output_path, _file), cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
                        if self.framework == 'nanodet':
                            # count.append(len(result))
                            from nanodet.util import overlay_bbox_cv
                            result = overlay_bbox_cv(meta['raw_img'][0], result, self.cfg.class_names, score_thresh=0.01)
                            cv2.imwrite(os.path.join(self.output_path, _file), result)       
            # print(scores)
            if self.mode == 'evaluate':
                #Perform step 3
                output_df = utils.create_ground_truths(self.dataset_path, self._classes)
                output_df = output_df.drop(['dummy'], axis = 1)
                output_df.to_csv(os.path.join(self.output_path, 'output.csv'), index = None)

                converters = {defect : pd.eval for defect in self._classes}
                ground_truth_df = pd.read_csv(os.path.join(self.output_path, 'ground_truths.csv'), converters = converters)
                strawman_df, detailed_df, confmat_df, score_df = utils.compare(output_df, ground_truth_df, scores)

                # Save Reports
                strawman_df = pd.DataFrame(strawman_df)
                strawman_df.to_csv(os.path.join(self.output_path, 'strawman_report.csv'), index = False)
                detailed_df = pd.DataFrame(detailed_df)
                detailed_df.to_csv(os.path.join(self.output_path, 'detailed_report.csv'), index = False)
                detailed_df = pd.DataFrame(confmat_df)
                detailed_df.to_csv(os.path.join(self.output_path, 'ConfusionMatrix_report.csv'), index = False)
                
                score_df = pd.DataFrame(score_df)
                score_df.to_csv(os.path.join(self.output_path, 'ThresholdScores_report.csv'), index = False)
                

                
                
if __name__ == '__main__':
    file_name = open('/mnt/2tb/General/Radhika/annotate++/evaluation_config.json')
    config = json.load(file_name)
    evaluate = Evaluate(config)      
    evaluate.run()

