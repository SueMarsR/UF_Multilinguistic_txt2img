# import argparse
import glob
import os
from pyiqa import create_metric
from tqdm import tqdm
import csv
import sys
from easydict import EasyDict

# easydict
metric_list = ["clipiqa+", "maniqa", "dbcnn", "paq2piq", "hyperiqa", "nima", "nima-vgg16-ava", "cnniqa", "brisque", "ilniqe", "niqe", "musiq", "musiq-ava", "musiq-koniq", "musiq-paq2piq", "musiq-spaq", "tres"]
# metric_list = ["nima", "musiq-koniq", "dbcnn", "tres", "nima-vgg16-ava", "musiq-ava"]

##
device = "cuda:1"
picture_path = "model_output_pic/promptist_0"
model_name = picture_path.replace("/", "_").replace(".", "")
model_iqa_dict = {}
# output_writer_file = open("./log/iqa_log.csv", "w")
# writer = csv.writer(outputwriter)


def call(arg_metric, arg_picture_path, writer):
    """Inference demo for pyiqa."""
    # parser = argparse.ArgumentParser()
    args = EasyDict()
    args.input = arg_picture_path
    args.ref = None
    args.metric_mode = "NR"
    args.metric_name = arg_metric
    args.save_file = None
    print("args.metric_name: ", args.metric_name)
    print("args.input: ", args.input)

    metric_name = args.metric_name.lower()

    # set up IQA model
    iqa_model = create_metric(metric_name, metric_mode=args.metric_mode, device=device)
    metric_mode = iqa_model.metric_mode

    if os.path.isfile(args.input):
        input_paths = [args.input]
        if args.ref is not None:
            ref_paths = [args.ref]
    else:
        input_paths = sorted(glob.glob(os.path.join(args.input, "*")))
        if args.ref is not None:
            ref_paths = sorted(glob.glob(os.path.join(args.ref, "*")))

    if args.save_file:
        sf = open(args.save_file, "w")

    avg_score = 0
    test_img_num = len(input_paths)
    if metric_name != "fid":
        pbar = tqdm(total=test_img_num, unit="image")
        for idx, img_path in enumerate(input_paths):
            img_name = os.path.basename(img_path)
            if metric_mode == "FR":
                ref_img_path = ref_paths[idx]
            else:
                ref_img_path = None

            score = iqa_model(img_path, ref_img_path).cpu().item()
            avg_score += score
            pbar.update(1)
            # pbar.set_description(f"{metric_name} of {img_name}: {score}")
            # pbar.write(f"{metric_name} of {img_name}: {score}")
            # if args.save_file:
            #     sf.write(f"{img_name}\t{score}\n")
        pbar.close()
        avg_score /= test_img_num
    else:
        assert os.path.isdir(args.input), "input path must be a folder for FID."
        avg_score = iqa_model(args.input, args.ref)

    msg = f"Average {metric_name} score of {args.input} with {test_img_num} images is: {avg_score}"
    print(msg)
    if args.save_file:
        sf.write(msg + "\n")
        sf.close()

    if args.save_file:
        print(f"Done! Results are in {args.save_file}.")
    else:
        print(f"Done!")

    writer.writerow({"model": arg_picture_path, "iqa": arg_metric, "score": avg_score})
    model_iqa_dict[arg_metric] = avg_score
    # writer.flush()


def main(picture_path):
    # gpu_id = int(
    # device = f"cuda:{gpu_id}"
    # picture_path = sys.argv[2]
    model_name = picture_path.replace("/", "_").replace(".", "")
    model_iqa_dict["model"] = model_name

    ##
    writer_file = open(f"./metric_iqa_max_new_tokens/iqa_{model_name}_each_metric.csv", "w", newline="")
    writer_1 = csv.DictWriter(writer_file, fieldnames=["model", "iqa", "score"])
    writer_1.writeheader()

    for metric in metric_list:
        call(metric, picture_path, writer_1)
        writer_file.flush()

    # writer_1.close()

    writer_2 = csv.DictWriter(open(f"./metric_iqa_max_new_tokens/iqa_{model_name}_each_model.csv", "w", newline=""), fieldnames=["model"] + metric_list)
    writer_2.writeheader()
    writer_2.writerow(model_iqa_dict)


if __name__ == "__main__":
    for i in range(1, 31):
        picture_path = f"./model_output_pic_max_new_token/gpt2_word_length_{i}"
        main(picture_path)

    # writer_2.close()
