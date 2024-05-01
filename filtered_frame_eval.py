import os
from multiprocessing import freeze_support
import shutil
import tempfile
import configparser
import json
import trackeval 

def filter_frames(input_file, t0, t1):
    """ 
    Filter frames in a file based on the frame number
    Args:
        input_file (str): The path to the input file
        t0 (int): The start frame number
        t1 (int): The end frame number
    """
    # Create a temporary file
    temp_fd, temp_path = tempfile.mkstemp()
    try:
        with open(input_file, 'r') as f, os.fdopen(temp_fd, 'w') as out:
            for line in f:
                frame_number = int(line.split(',')[0])
                if t0 <= frame_number <= t1:
                    out.write(line)
        
        # Replace the original file with the temporary file
        shutil.move(temp_path, input_file)
    except Exception as e:
        print("An error occurred:", e)
        os.remove(temp_path)
        raise
    finally:
        # Ensure that the temp file is removed if it wasn't moved
        if os.path.exists(temp_path):
            os.remove(temp_path)

def allowed_file(filename):
    """ Check if the uploaded file is allowed by its extension 
    Args:
        filename (str): The name of the file
    Returns:
        bool: True if the file is allowed, False otherwise
    """
    allowed_extensions = {'txt'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def run_evaluation(t0, t1, SEQ_INFO = 'MOT16-02', uploaded_txt_dir = 'data/trackers/mot_challenge/MOT16-train/MPNTrack/data/MOT16-02.txt'):
    """
    Run the evaluation process
    Args:
        t0 (int): The start frame number
        t1 (int): The end frame number
        uploaded_txt_dir (str): The path to the uploaded txt file
        SEQ_INFO (str): The sequence information
    Returns:
        dict: A dictionary containing the evaluation results
    """
    # default config
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['DISPLAY_LESS_PROGRESS'] = False

    # 2. configs include GT data, tracker data, benchmark, SPLIT_TO_EVAL, etc
    default_dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()

    # 3. configs include metrics, thresholds, etc
    default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}

     # Merge default configs
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}
    
    # Basic Argument
    BENCHMARK = 'MOT16'
    SPLIT_TO_EVAL = 'train'
    TRACKERS_TO_EVAL = ['MPNTrack']
    METRICS = ['HOTA', 'CLEAR', 'Identity', 'VACE']
    USE_PARALLEL = False
    NUM_PARALLEL_CORES = 1   
    
    # get default seq length
    temp_gt_seq_dir = os.path.join(default_dataset_config['GT_FOLDER'], f"{BENCHMARK}-{SPLIT_TO_EVAL}", f"{SEQ_INFO}")
    seqinfo_file = os.path.join(temp_gt_seq_dir, 'seqinfo.ini')
    seq_info_config = configparser.ConfigParser()
    seq_info_config.read(seqinfo_file)
    
    # modify SEQ info format for the config
    SEQ_INFO = {SEQ_INFO: None}
    
    arg_dic = {'BENCHMARK': BENCHMARK, 'SPLIT_TO_EVAL': SPLIT_TO_EVAL, 'TRACKERS_TO_EVAL': TRACKERS_TO_EVAL,
     'METRICS': METRICS, 'USE_PARALLEL': USE_PARALLEL, 'NUM_PARALLEL_CORES': NUM_PARALLEL_CORES, 'SEQ_INFO': SEQ_INFO}

    for key, item in arg_dic.items():
        if key in config:
            config[key] = item

    # update 3 config
    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

    # print current config
    print('==' * 10, 'eval config', '==' * 10)
    for key, value in eval_config.items():
        print(key, ':', value)
    print('==' * 10, 'dataset config', '==' * 10)
    for key, value in dataset_config.items():
        print(key, ':', value)
    print('==' * 10, 'metrics config', '==' * 10)
    for key, value in metrics_config.items():
        print(key, ':', value)
    print('==' * 36)
    
    # Filter frames if t0 and t1 are provided
    temp_dir_used = False
    if t0 is not None and t1 is not None:
        temp_dir_used = True
        # obtain information about the sequence and tracker to evaluate (only for a single sequence and tracker)
        seq_info = list(dataset_config['SEQ_INFO'].keys())[0] # e.g: MOT16-02
        tracker = dataset_config['TRACKERS_TO_EVAL'][0] # e,g: MPNTrack

        # =============Now we want to make a copy of desired GT folder with only the desired sequence of selected frames================
        # get original gt/ sequence folder
        original_MOT16train_GT_folder = os.path.join(dataset_config['GT_FOLDER'], f"{dataset_config['BENCHMARK']}-{dataset_config['SPLIT_TO_EVAL']}", seq_info)
        
        # copy the original gt folder to a temporary folder
        # empty the temp folder if it exists
        temp_gt_seq_dir = os.path.join(dataset_config['GT_FOLDER'], f"{dataset_config['BENCHMARK']}-{dataset_config['SPLIT_TO_EVAL']}", f"{seq_info}_temp")
        # if os.path.exists(temp_gt_seq_dir):
        #     shutil.rmtree(temp_gt_seq_dir)
        shutil.copytree(original_MOT16train_GT_folder, temp_gt_seq_dir)
        
        # filter frames in the desired sequence
        gt_file = os.path.join(temp_gt_seq_dir, 'gt', 'gt.txt')
        filter_frames(gt_file, t0, t1)

        # modify seqLength in data/gt/mot_challenge/MOT16-train_temp/MOT16-02/seqinfo.ini to reflect the new number of frames
        seqinfo_file = os.path.join(temp_gt_seq_dir, 'seqinfo.ini')
        seq_info_config = configparser.ConfigParser()
        seq_info_config.read(seqinfo_file)
        seq_info_config['Sequence']['seqLength'] = str(t1 - t0 + 1)

        with open(seqinfo_file, 'w') as f:
            seq_info_config.write(f)
        
        # =============Now we want to make a copy of desired tracker with only the selected frames================
        # get original tracker folder
        original_MOT16train_tracker_folder = os.path.join(dataset_config['TRACKERS_FOLDER'], f"{dataset_config['BENCHMARK']}-{dataset_config['SPLIT_TO_EVAL']}", tracker)
        
        print("Checking directory:", original_MOT16train_tracker_folder)
        if not os.path.exists(original_MOT16train_tracker_folder):
            print("Directory does not exist:", original_MOT16train_tracker_folder)
        else:
            print("Directory exists, proceeding to copy...")

        # copy the original tracker folder to a temporary folder, if temp folder exists, remove it
        temp_tracker_dir = os.path.join(dataset_config['TRACKERS_FOLDER'], f"{dataset_config['BENCHMARK']}-{dataset_config['SPLIT_TO_EVAL']}", f"{tracker}_temp")
        # if os.path.exists(temp_tracker_dir):
        #     shutil.rmtree(temp_tracker_dir)
        shutil.copytree(original_MOT16train_tracker_folder, temp_tracker_dir)
    
        # filter frames in the desired tracker
        tracker_file = os.path.join(temp_tracker_dir, 'data', f"{seq_info}.txt")
        
        # if upload txt file, replace the original tracker file in the temp diretory with the uploaded file (copy instead of remove uploaded_txt_dir) 
        if uploaded_txt_dir:
            shutil.copyfile(uploaded_txt_dir, tracker_file)

        filter_frames(tracker_file, t0, t1)

        # rename the txt
        os.rename(tracker_file, os.path.join(temp_tracker_dir, 'data', f"{seq_info}_temp.txt"))

        # modify the config to indicate what we are evaluating now
        dataset_config['TRACKERS_TO_EVAL'] = [f"{tracker}_temp"]
        dataset_config['SEQ_INFO'] = {f"{seq_info}_temp": None}
    
    # print current config
    print('==' * 10, 'eval config', '==' * 10)
    for key, value in eval_config.items():
        print(key, ':', value)
    print('==' * 10, 'dataset config', '==' * 10)
    for key, value in dataset_config.items():
        print(key, ':', value)
    print('==' * 10, 'metrics config', '==' * 10)
    for key, value in metrics_config.items():
        print(key, ':', value)
    print('==' * 20)

    # Run code
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric(metrics_config))
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')

    output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list)
    print('Eval Ends in Backend')
    
    # get the output: pedestrian_summary.txt in the tracker folder
    output_folder = temp_tracker_dir
    output_txt = os.path.join(output_folder, 'pedestrian_summary.txt')

    # read the output file, first lines as keys and second lines as values
    try:
        with open(output_txt, 'r') as f:
            keys = f.readline().strip().split()
            values = f.readline().strip().split()
    except FileNotFoundError:
        print(f"Error: The file {output_txt} does not exist.")
        exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)
    
    # remove the temporary directories
    if temp_dir_used:
        shutil.rmtree(temp_gt_seq_dir)
        shutil.rmtree(temp_tracker_dir)
        print("Temporary directories removed.")
    
    converted_values = converted_values = [float(value) if value.replace('.', '', 1).isdigit() else value for value in values]
    temp_dict = dict(zip(keys, converted_values))

    # Define keys for each category
    HOTA_keys = {'HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr', 'AssRe', 'AssPr', 'LocA', 'OWTA', 'HOTA(0)', 'LocA(0)', 'HOTALocA(0)'}
    CLEAR_keys = {'MOTA', 'MOTP', 'MODA', 'CLR_Re', 'CLR_Pr', 'MTR', 'PTR', 'MLR', 'sMOTA', 'CLR_TP', 'CLR_FN', 'CLR_FP', 'IDSW', 'MT', 'PT', 'ML', 'Frag'}
    Identity_keys = {'IDF1', 'IDR', 'IDP', 'IDTP', 'IDFN', 'IDFP'}
    VACE_keys = {'SFDA', 'ATA'}
    COUNT_keys = {'Dets', 'GT_Dets', 'IDs', 'GT_IDs'}

    # Define a dictionary of dictionaries
    categories = {
        'HOTA': HOTA_keys,
        'CLEAR': CLEAR_keys,
        'Identity': Identity_keys,
        'VACE': VACE_keys,
        'COUNT': COUNT_keys
    }

    # Create dictionaries dynamically
    category_dicts = {category: {} for category in categories}

    # Distribute key-value pairs to the respective category dictionaries
    for key, value in temp_dict.items():
        for category, keys in categories.items():
            if key in keys:
                category_dicts[category][key] = value
                break
    
    return category_dicts
    # # convert dict to json
    # json_eval_results = json.dumps(category_dicts, indent=4)
    # print('Received JSON Evaluation Results, ready to return...')

    # # # Return the results as a JSON response
    # return json_eval_results

if __name__ == '__main__':
    t0 = 1
    t1 = 100
    uploaded_txt_dir = 'data/trackers/mot_challenge/MOT16-train/MPNTrack/data/MOT16-02.txt'
    SEQ_INFO = 'MOT16-02'
    eval_results = run_evaluation(t0, t1, SEQ_INFO, uploaded_txt_dir)
    print('Evaluation Results:')
    print(eval_results)