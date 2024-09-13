import argparse
import glob
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np
import pandas as pd
from PIL import Image
from collections import defaultdict
import numpy as np

def load_model(framework, model_config, thresholds):

    config_path = model_config['config_path']
    model_path = model_config['model_path']

    if framework == 'nanodet':
        import nanodet
        import torch
        from nanodet.util import Logger, cfg, load_config, overlay_bbox_cv
        from demo import Predictor
        load_config(cfg, config_path)
        logger = Logger(-1, use_tensorboard=False)
        device = torch.device('cuda')
        model = Predictor(cfg, model_path, logger, device=device)
        model.visualize
        return model, cfg
#     else:
#         from mmdet.apis import inference_detector
#         from defect_detection import DefectDetection
        
#         model = DefectDetection(model_path, config_path, device = "cuda", score_thresholds = thresholds)
#         return model, None
                    
def check_thresholds_and_classes(model, framework, thresholds, cfg = None):
    if framework == 'nanodet':
        classes = cfg.class_names
    elif framework == 'mmdetection':
        classes = model._classes
   
                    
    if len(thresholds) == len(classes):
        return True
        
    return False
    
def get_xmls(file_path, img_shape, defect_dict):
        annotation = ET.Element('annotation')
        folder = ET.SubElement(annotation, "folder")
        folder.text = file_path.split('/')[-2]
        filename = ET.SubElement(annotation, "filename")
        filename.text = file_path.strip().split('/')[-1]
        path = ET.SubElement(annotation, 'path')
        path.text = file_path
        source = ET.SubElement(annotation, 'source')
        database = ET.SubElement(source, 'database')
        database.text = "Unknown"
        size = ET.SubElement(annotation, 'size')
        width = ET.SubElement(size, 'width')
        width.text = str(img_shape[1])
        height = ET.SubElement(size, 'height')
        height.text = str(img_shape[0])
        depth = ET.SubElement(size, 'depth')
        depth.text = "1"
        segmented = ET.SubElement(annotation, 'segmented')
        segmented.text = "0"
        
        for dic in defect_dict:
            object1 = ET.SubElement(annotation, 'object')
            name = ET.SubElement(object1, 'name')
            name.text = dic["name"]
            pose = ET.SubElement(object1, 'pose')
            pose.text = "unspecified"
            truncated = ET.SubElement(object1, 'truncated')
            truncated.text = "0"
            difficult = ET.SubElement(object1, 'difficult')
            difficult.text = "0"
            bndbox = ET.SubElement(object1, 'bndbox')
            xmin = ET.SubElement(bndbox, "xmin")
            xmin.text = str(int(dic['xmin']))
            ymin = ET.SubElement(bndbox, "ymin")
            ymin.text = str(int(dic['ymin']))
            xmax = ET.SubElement(bndbox, "xmax")
            xmax.text = str(int(dic['xmax']))
            ymax = ET.SubElement(bndbox, "ymax")
            ymax.text = str(int(dic['ymax']))
        
        mydata = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="    ")
        saving_file_name = f"{file_path[:-4]}.xml"
        open(saving_file_name, "w").close()
        with open(saving_file_name, "w") as f:
            f.write(mydata)
                    
def nanodet_filter_boxes(dets, score_thresh):
        dets = dets[0]
        if isinstance(score_thresh, list):
            assert len(score_thresh)==len(dets.keys())
            
        elif isinstance(score_thresh, float):
            score_thresh = [score_thresh]*len(dets.keys())
        else:
            raise TypeError("Invalid format of score threshold")
            
        for i, (label, preds) in enumerate(dets.items()):
            dets[label] = list(filter(lambda x: x[-1]>float(score_thresh[i]), preds))
        return dets

def create_xmls(framework, result, image_path, classes = None, meta = None):
    
    defect_list = list()
    
    if framework == 'nanodet':
        
        for key, val in result.items():
            if not val == [] and len(val) == 0:
                val = val[0]
            for j in val:
                defect_list += [{"name": classes[key],
                                "xmin": j[0],
                                "ymin": j[1],
                                "xmax": j[2],
                                "ymax": j[3]}]
        # import pdb; pdb.set_trace()
        get_xmls(image_path, meta["raw_img"][0].shape, defect_list)
                    
    # elif framework == 'mmdetection':
    #     image = Image.open(image_path)
    #     defect_list = list(map(lambda key,val:{"name":key,
    #                             "xmin":val[0],
    #                             "xmax":val[2],
    #                             "ymin":val[1],
    #                             "ymax":val[3]},
    #             result.values(),
    #             result.keys()
    #             )
    #         )
    #     get_xmls(image_path, (image.size[1],image.size[0]), defect_list)
        
    return defect_list

def create_ground_truths(path, classes):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        if not os.path.exists(xml_file.replace('.xml', '.jpg')):
            print(f'Could not find the associated image file for file --> {xml_file}')
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            bndbox = member.find('bndbox')
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member.find('name').text,
                     int(float(bndbox[0].text)),
                     int(float(bndbox[1].text)),
                     int(float(bndbox[2].text)),
                     int(float(bndbox[3].text)))
            xml_list.append(value)
        if not root.findall('object'):
            value=(root.find('filename').text,
                  int(root.find('size')[0].text),
                   int(root.find('size')[1].text)
                  )
            xml_list.append(value)
    column_name = ['filename','width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    import pdb;pdb.set_trace()
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    xml_df['class']= xml_df['class'].replace(np.nan,'dummy')
    xml_df['xmin']= xml_df['xmin'].replace(np.nan,0).astype(int)
    xml_df['ymin']= xml_df['ymin'].replace(np.nan,0).astype(int)
    xml_df['xmax']= xml_df['xmax'].replace(np.nan,0).astype(int)
    xml_df['ymax']= xml_df['ymax'].replace(np.nan,0).astype(int)
    classes.append('dummy')
    result_df = {'filename' : list()}
    for _class in classes:
        result_df[_class] = list()
    filename_list = list()
    for index, row in xml_df.iterrows():
        if row['filename'] not in filename_list:
            filename_list.append(row['filename'])
            result_df['filename'].append(row['filename'])
            for _class in classes:
                result_df[_class].append(list())
            result_df[row['class']][len(filename_list) - 1].append([row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        else:
            file_index = result_df['filename'].index(row['filename'])
            result_df[row['class']][file_index].append([row['xmin'], row['ymin'], row['xmax'], row['ymax']])
    df = pd.DataFrame(result_df)
    return df

def fetch_image_score_dict(scores, filename):
    for _score in scores:
        if filename in list(_score.keys()):
            return _score

def fetch_defect_scores(scores, defect):
    result = list()
    for _, scores in scores.items():
        for _score in scores:
            if defect in list(_score.keys()):
                result.append(list(_score.values())[0])
    return result


def compare(output, ground_truth, scores):
    strawman_df = {x : list() for x in output.columns}
    detailed_df = {'filename' : list()}
    confmat_df = {'filename': list()}
    score_df = {'filename': list()}
    for column in output.columns:
        if column != 'filename':
            detailed_df[column] = list()
            detailed_df[column + '_count'] = list()
            detailed_df[column + '_overlap'] = list()
            detailed_df[column + '_overlap_mismatch'] = list()
            detailed_df[column + '_overlap_match'] = list()
            confmat_df[column + '_gtCount'] = list()
            confmat_df[column + '_predCount'] = list()
            confmat_df[column + '_truePositive'] = list()
            confmat_df[column + '_missed'] = list()
            confmat_df[column + '_falsePositive'] = list()
            score_df[column + '_fpscore' ] = list()
            score_df[column + '_tpscore'] = list()
            score_df[column + '_fpscore_avg' ] = list()
            score_df[column + '_tpscore_avg'] = list()

    # Cursory checks
    if len(output) != len(ground_truth):
        raise Exception('Mismatch of number of images between output and ground truth. Make sure you run evaluation on the same test dataset.')
        
    if len(output.columns) != len(ground_truth.columns):
        raise Exception('Mismatch of defect names between output and ground truth.  Make sure you run evaluation on the same test dataset.')
    
    # Iterate over each filename in ground truth
    for index, row in ground_truth.iterrows():
        strawman_df['filename'].append(row['filename'])
        detailed_df['filename'].append(row['filename'])
        confmat_df['filename'].append(row['filename'])
        score_df['filename'].append(row['filename'])
        # Iterate over each defect
        for defect in ground_truth.columns[1:]:
            detailed_df[defect + '_overlap_mismatch'].append(list())
            detailed_df[defect + '_overlap_match'].append(list())
            confmat_df[defect + '_truePositive'].append(0)
            confmat_df[defect + '_missed'].append(0)
            confmat_df[defect + '_falsePositive'].append(0)
            score_df[defect + '_fpscore'].append(list())
            score_df[defect + '_tpscore'].append(list())
            score_df[defect + '_fpscore_avg'].append(0)
            score_df[defect + '_tpscore_avg'].append(0)
        
            gt_defect_list = row[defect]
            output_defect_list = output.at[index,defect]
            image_score_dict = fetch_image_score_dict(scores, row['filename'])
            defect_scores = fetch_defect_scores(image_score_dict, defect)

            
            # Update count in detailed df
            if len(output_defect_list) != len(gt_defect_list):
                detailed_df[defect + '_count'].append(2)
            else:
                detailed_df[defect + '_count'].append(1)

            # Update count in confmat df
            confmat_df[defect + '_gtCount'].append(len(gt_defect_list))
            confmat_df[defect + '_predCount'].append(len(output_defect_list))
            
            # Sort both lists from left to right - can potentially lead to faster search
            #sorted(output_defect_list, key = lambda x : x[0])
            #sorted(gt_defect_list, key = lambda x : x[0])
            
            # Perform IoU comparison 
            for gt_box in gt_defect_list:
                overlap_flag = False
                for o_box in output_defect_list:
                    if intersection_over_union(gt_box, o_box) > 0.6:
                        overlap_flag = True
                        #score_df[defect + '_tpscore'][index].append(defect_scores[output_defect_list.index(o_box)])
                        break
                    #else:
                        #pass
                    #    score_df[defect + '_fpscore'][index].append(defect_scores[output_defect_list.index(o_box)])

                if overlap_flag:
                    # print('overlap flag is true')
                    detailed_df[defect + '_overlap_match'][index].append(gt_box)
                    # print(confmat_df[defect + '_truePositive'])
                    confmat_df[defect + '_truePositive'][index] += 1
                else:
                    detailed_df[defect + '_overlap_mismatch'][index].append(gt_box)
                    confmat_df[defect + '_missed'][index] += 1
            
            if len(gt_defect_list) == len(detailed_df[defect + '_overlap_match'][index]):
                detailed_df[defect + '_overlap'].append(1)
            else:
                detailed_df[defect + '_overlap'].append(2)
            
            # Update strawman
            if detailed_df[defect + '_count'][-1] == 1 and detailed_df[defect + '_overlap'][-1] == 1: 
                strawman_df[defect].append(1)
                detailed_df[defect].append(1)
            else:
                strawman_df[defect].append(2)
                detailed_df[defect].append(2)
                
            confmat_df[defect + '_falsePositive'][index] = confmat_df[defect + '_predCount'][index] - confmat_df[defect + '_truePositive'][index]
            
            # Populate score_df
            score_df = populate_scores(score_df, defect, index, defect_scores, gt_defect_list, output_defect_list)

    return strawman_df, detailed_df, confmat_df, score_df

def create_dict(list1, list2):
    temp=defaultdict(list)
    for x, y in zip(list1,list2):
        temp[x].append(y)
    return temp

def intersection_over_union(boxA, boxB):
    
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    
    return interArea / float(boxAArea + boxBArea - interArea)



def populate_scores(df, defect, index, scores, gt_defect_list, output_defect_list):
    # print('################## defect ###############',defect)
    # We have created this separate function because in here the loop would be in reference to the output defect list unlike the compare(). This solves the problem of repeated scores in fp and tp columns for a defect. The reason why compare() is in reference to the ground truth needs to be revisited. - sorted
    try:
        for o_box in output_defect_list:
            overlap_flag = False
            for gt_box in gt_defect_list:
                if intersection_over_union(gt_box, o_box) > 0.1:
                    overlap_flag = True
            if overlap_flag:
                df[defect + '_tpscore'][index].append(scores[output_defect_list.index(o_box)])
            else:
                df[defect + '_fpscore'][index].append(scores[output_defect_list.index(o_box)])
    except Exception as e:
        print("error", e)
        import pdb; pdb.set_trace()
    try:
        df[defect + '_tpscore_avg'][index] = np.mean(df[defect + '_tpscore'][index])
        # print('this tp stage',df[defect + '_tpscore_avg'][index])
    except:
        df[defect + '_tpscore_avg'][index] = 0
        
    
    try:
        df[defect + '_fpscore_avg'][index] = np.mean(df[defect + '_fpscore'][index])
        # print('this fp stage',df[defect + '_fpscore_avg'][index])
    except:
        df[defect + '_fpscore_avg'][index] = 0
        
    # print('#################df###########',df)
    return df

def append_score(out,defected_list):
    dd = []
    for key,val in out.items():
        #for k1,v1 in val.items():
         #   print(v1)
        if len(val) == 0:
            dd.append({key:[0]})
        else:
            for index in val:
                #for i in range(len(index)):
                dd.append({key:[index[4]]})
                #dd.append({key:[index[4]]})
    return dd


