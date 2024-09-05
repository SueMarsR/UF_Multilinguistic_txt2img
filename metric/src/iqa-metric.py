import pyiqa
import torch

device = "cuda:0"
# list all available metrics
# ['ahiq', 'brisque', 'ckdn', 'clipiqa', 'clipiqa+', 'clipiqa+_rn50_512', 'clipiqa+_vitL14_512', 'clipscore', 'cnniqa', 'cw_ssim', 'dbcnn', 'dists', 'entropy', 'fid', 'fsim', 'gmsd', 'hyperiqa', 'ilniqe', 'lpips', 'lpips-vgg', 'mad', 'maniqa', 'maniqa-kadid', 'maniqa-koniq', 'maniqa-pipal', 'ms_ssim', 'musiq', 'musiq-ava', 'musiq-koniq', 'musiq-paq2piq', 'musiq-spaq', 'nima', 'nima-vgg16-ava', 'niqe', 'nlpd', 'nrqm', 'paq2piq', 'pi', 'pieapp', 'psnr', 'psnry', 'ssim', 'ssimc', 'tres', 'tres-flive', 'tres-koniq', 'uranker', 'vif', 'vsi']
if __name__ == "__main__":
    print(pyiqa.list_models())

    device = torch.device(device)

    # create metric with default setting
    iqa_metric = pyiqa.create_metric("lpips", device=device)
    # Note that gradient propagation is disabled by default. set as_loss=True to enable it as a loss function.
    iqa_loss = pyiqa.create_metric("lpips", device=device, as_loss=True)

    # create metric with custom setting
    iqa_metric = pyiqa.create_metric("psnr", test_y_channel=True, color_space="ycbcr").to(device)

    # check if lower better or higher better
    print(iqa_metric.lower_better)

    # example for iqa score inference
    # Tensor inputs, img_tensor_x/y: (N, 3, H, W), RGB, 0 ~ 1
    # score_fr = iqa_metric(img_tensor_x, img_tensor_y)
    # score_nr = iqa_metric(img_tensor_x)

    # img path as inputs.
    score_fr = iqa_metric("./model_output_pic/gpt2_0/000000.png", "./model_output_pic/gpt2_0/000001.png")

    # For FID metric, use directory or precomputed statistics as inputs
    # refer to clean-fid for more details: https://github.com/GaParmar/clean-fid
    # fid_metric = pyiqa.create_metric("fid")
    # score = fid_metric("./ResultsCalibra/dist_dir/", "./ResultsCalibra/ref_dir")
    # score = fid_metric("./ResultsCalibra/dist_dir/", dataset_name="FFHQ", dataset_res=1024, dataset_split="trainval70k")
