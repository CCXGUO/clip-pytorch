# to see the first ten matched captions from 1000 for one single randomly choosed image
from PIL import Image
from random import choice
import json
from clip import CLIP
from utils.dataloader import ClipDataset

def create_val_dataset(datasets_path, datasets_val_json_path):

    model = CLIP()
    val_lines = json.load(open(datasets_val_json_path, mode = 'r', encoding = 'utf-8'))
    val_dataset = ClipDataset([model.config['input_resolution'], model.config['input_resolution']], val_lines, datasets_path, random = False)

    return val_dataset

def find_largest_elements_with_indices(lst, n):
    largest_elements_with_indices = sorted(enumerate(lst), key=lambda x: x[1], reverse=True)[:n]
    indices, elements = zip(*largest_elements_with_indices)
    return list(elements), list(indices)

def eval_predict(image_path, captions):
    clip = CLIP()
    image = Image.open(image_path)
    probs = clip.detect_image(image, captions)
    probs_lst = probs.tolist()[0]
    #largest_1, indices_1 = find_largest_elements_with_indices(probs_lst, 1)
    #largest_5, indices_5 = find_largest_elements_with_indices(probs_lst, 5)
    largest_10, indices_10 = find_largest_elements_with_indices(probs_lst, 10)

    #print("Label probs:", probs)
    return indices_10

def get_image(main_list, indices_list):
    selected_elements = [main_list[i] for i in indices_list]
    print(selected_elements)

if __name__ == "__main__":
    datasets_path = "datasets/"
    datasets_val_json_path = "datasets/en_train.json"

    eval_predict_dataset = create_val_dataset(datasets_path,datasets_val_json_path)
    image_path = 'datasets/' + choice(eval_predict_dataset.image)
    captions = eval_predict_dataset.text

    rank4ten = eval_predict(image_path, captions)
    get_image(captions, rank4ten)
    print(image_path)


